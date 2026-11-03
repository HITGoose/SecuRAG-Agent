"""
手写的 ReAct 式工具调用循环：模型决策 → 执行工具 → 把结果（观察）交还模型 → 再决策 …… 直到给出回答。

一句话版本：
    for 每一步:
        回复 = 模型(消息, 工具说明)
        没有工具调用 → 这就是最终回答，返回
        有工具调用   → 逐个：解析参数 → 校验 → 查重 → 查预算 → 执行(带超时、兜异常) → 结果追加进消息
    步数用完 → 兜底回复

这个文件不认识任何具体业务：工具说明、参数校验、工具执行都由调用方传进来，
所以本地工具（阶段1）和 MCP 工具（阶段2）都能用同一个循环。
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ToolTimeout
from dataclasses import dataclass, field

# 达到各种上限时的固定回复：不再让模型继续生成，避免越跑越偏
FALLBACK_ANSWERS = {
    "max_steps": "抱歉，这个问题我没能在有限的步骤内处理完。建议您转人工客服，我可以帮您生成工单。",
    "tool_budget": "抱歉，这个问题需要查询的信息较多，我没能处理完。建议您转人工客服，我可以帮您生成工单。",
    "repeated": "抱歉，我在重复查询同样的信息，没能得到新的结果。建议您转人工客服，我可以帮您生成工单。",
    "empty_answer": "抱歉，我暂时没能生成有效的回答，请换个说法再问一次，或转人工客服。",
    "llm_error": "抱歉，系统暂时无法处理您的请求，请稍后再试，或转人工客服。",
}

# 工具在线程里执行，才能设超时。注意：超时后线程不会被强制杀掉，只是循环不再等它
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class LoopResult:
    answer: str
    stop_reason: str                            # answered / max_steps / tool_budget / repeated / empty_answer / llm_error
    steps: list = field(default_factory=list)   # 每个工具调用一条记录（工具名、参数、状态、耗时）
    llm_calls: int = 0                          # 循环内的模型调用次数
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_complete: bool = True                # 有任何一次调用拿不到 usage 就为 False，不能当 0 算


def tool_error(code: str, message: str, retryable: bool = False) -> dict:
    """工具出问题时给模型看的观察结果。只有简短原因，不含异常栈、路径、密钥。"""
    return {"ok": False, "data": None, "error": {"code": code, "message": message, "retryable": retryable}}


def run_agent_loop(llm, messages: list, tool_schemas: list, prepare_args, execute_tool,
                   max_steps: int = 5, max_tool_calls: int = 5, max_calls_per_response: int = 3,
                   tool_timeout_s: float = 10) -> LoopResult:
    """
    llm:           有 chat(messages, tools) 方法的对象（llm_gateway.LLMGateway）
    messages:      [系统提示, 历史..., 用户问题]，循环会往里追加，调用方传副本
    prepare_args:  (工具名, 参数dict) -> (规范化参数, 错误信息)   —— 校验 + 规范化
    execute_tool:  (工具名, 规范化参数) -> 结果dict              —— 真正执行（用户身份已由调用方绑定）
    """
    result = LoopResult(answer="", stop_reason="")
    seen = {}             # 去重表：(工具名, 规范化参数) -> 第一次执行的结果
    executed = 0          # 真正执行过的工具次数（重复的、被拒的不算）
    repeat_streak = 0     # 连续重复调用的次数，到 2 就停

    for step in range(1, max_steps + 1):
        # ---- 1. 模型决策 ----
        try:
            reply = llm.chat(messages, tools=tool_schemas)
        except Exception:
            # 阶段5 的网关会先重试、降级；走到这里说明模型真的不可用
            return _finish(result, "llm_error")
        result.llm_calls += 1
        _add_usage(result, reply)
        tool_calls = reply.message.get("tool_calls") or []

        # ---- 2. 没有工具调用：这就是最终回答 ----
        if not tool_calls:
            answer = (reply.message.get("content") or "").strip()
            if not answer:
                return _finish(result, "empty_answer")
            result.answer, result.stop_reason = answer, "answered"
            return result

        # ---- 3. 有工具调用：先把模型的"调用请求"消息放进历史 ----
        # 接口要求：每条 tool 结果消息前面必须有对应的 assistant tool_calls 消息，
        # 而且每个 tool_call_id 都要有一条结果，哪怕我们没执行它
        messages.append(reply.message)
        stop_reason = None

        for i, call in enumerate(tool_calls):
            name = call["function"]["name"]
            start = time.perf_counter()
            args, status = None, "ok"

            if i >= max_calls_per_response:
                # 一次回复里请求太多工具：多出来的不执行，防止用一次回复绕过步数限制
                observation, status = tool_error("TOO_MANY_CALLS", f"一次最多执行 {max_calls_per_response} 个工具调用"), "skipped"
            else:
                # 3a. 解析参数：模型给的是 JSON 字符串，可能不合法
                try:
                    raw_args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    raw_args = None
                # 3b. 校验 + 规范化：工具名在白名单里、参数符合 schema
                args, error_msg = prepare_args(name, raw_args) if raw_args is not None else (None, "参数不是合法的 JSON")

                if error_msg:
                    observation, status = tool_error("BAD_ARGUMENTS", error_msg), "bad_args"
                else:
                    # 3c. 查重：同一工具 + 同样的规范化参数，不再执行，直接给之前的结果
                    key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                    if key in seen:
                        observation = {**seen[key], "note": "重复调用：这个工具已经用相同参数调用过，结果同上，请直接使用"}
                        status = "duplicate"
                        repeat_streak += 1
                    # 3d. 查预算：真正执行的工具次数有上限
                    elif executed >= max_tool_calls:
                        observation, status = tool_error("TOOL_BUDGET_EXCEEDED", "本轮工具调用次数已用完"), "budget"
                        stop_reason = "tool_budget"
                    # 3e. 执行：带超时，任何异常都变成一条错误观察，不让整个请求崩掉
                    else:
                        observation, status = _execute_with_timeout(execute_tool, name, args, tool_timeout_s)
                        seen[key] = observation
                        executed += 1
                        repeat_streak = 0

            # 3f. 记录这一步，并把结果作为 tool 消息交还给模型
            result.steps.append({"step": step, "tool": name, "args": args, "status": status,
                                 "error_code": (observation.get("error") or {}).get("code"),
                                 "latency_ms": int((time.perf_counter() - start) * 1000)})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(observation, ensure_ascii=False)})

        # ---- 4. 这一批工具处理完，检查是否该停 ----
        if repeat_streak >= 2:
            return _finish(result, "repeated")
        if stop_reason:
            return _finish(result, stop_reason)

    # ---- 5. 步数用完还没回答 ----
    return _finish(result, "max_steps")


def _execute_with_timeout(execute_tool, name, args, timeout_s) -> tuple:
    future = _executor.submit(execute_tool, name, args)
    try:
        observation = future.result(timeout=timeout_s)
        return observation, ("ok" if observation.get("ok") else "tool_failed")
    except ToolTimeout:
        return tool_error("TIMEOUT", f"工具执行超过 {timeout_s} 秒", retryable=True), "timeout"
    except Exception as e:
        # 只告诉模型异常类型，具体信息可能含路径等内部细节
        return tool_error("TOOL_ERROR", f"工具内部错误（{type(e).__name__}）"), "exception"


def _add_usage(result: LoopResult, reply) -> None:
    if reply.input_tokens is None or reply.output_tokens is None:
        result.tokens_complete = False
        return
    result.input_tokens += reply.input_tokens
    result.output_tokens += reply.output_tokens


def _finish(result: LoopResult, reason: str) -> LoopResult:
    result.answer, result.stop_reason = FALLBACK_ANSWERS[reason], reason
    return result
