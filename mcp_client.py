"""
MCP 客户端适配器：Agent 通过它调用 shop_mcp_server.py 暴露的订单类工具。

职责：
1. 启动 MCP Server 子进程（stdio），握手，list_tools 核对工具白名单和参数
2. 调用时注入可信参数 user_id / request_id（模型不可见），解析返回
3. 超时、断连：断开时自动重连一次再试；仍失败就抛异常，由 agent_loop 转成错误观察

为什么要同步包装：agent_loop 是同步代码（面试时好讲、好默写），而 SDK 是异步的。
这里用 anyio 的 BlockingPortal 在一个后台线程里跑事件循环；用 wrap_async_context_manager
保证连接的建立和关闭发生在同一个异步任务里（SDK 的要求）。
"""
import json
import sys
from pathlib import Path

from anyio import fail_after
from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters

import config
from shop_tools import SCHEMA_BY_NAME, ToolContext

ROOT = Path(__file__).parent
MCP_TOOLS = {"get_order", "track_shipment", "transfer_to_human"}  # 白名单：只接纳这几个
INJECTED = {"user_id", "request_id"}                               # 由适配器注入、对模型隐藏的参数


class MCPToolError(Exception):
    """MCP 调用失败（Server 断开、超时、协议错误）。"""


class ShopMCPClient:
    def __init__(self, timeout_s: float = config.TOOL_TIMEOUT_S):
        self.timeout_s = timeout_s
        self.params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "shop_mcp_server.py")],
                                            cwd=str(ROOT), env={"PYTHONIOENCODING": "utf-8"})
        self._portal_cm = self._portal = None
        self._client_cm = self._client = None
        self.reconnects = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._connect()

    def _connect(self) -> None:
        self._client_cm = self._portal.wrap_async_context_manager(Client(self.params))
        self._client = self._client_cm.__enter__()  # 启动子进程 + initialize 握手
        self._check_tools()

    def _disconnect(self) -> None:
        if self._client_cm is not None:
            try:
                self._client_cm.__exit__(None, None, None)
            except Exception:
                pass  # 子进程可能已经不在了，关闭时的报错不影响后续重连
            self._client_cm = self._client = None

    def close(self) -> None:
        self._disconnect()
        if self._portal_cm is not None:
            self._portal_cm.__exit__(None, None, None)
            self._portal_cm = self._portal = None

    # ---------- 工具核对 ----------

    def _check_tools(self) -> None:
        """Server 暴露的工具必须和本地 schema 一致（去掉注入参数后），否则拒绝使用，避免两边悄悄不一致。"""
        listed = self._portal.call(self._client.list_tools).tools
        found = {t.name: t for t in listed if t.name in MCP_TOOLS}
        missing = MCP_TOOLS - set(found)
        if missing:
            raise MCPToolError(f"MCP Server 缺少工具：{sorted(missing)}")
        for name, tool in found.items():
            schema = tool.input_schema
            server_props = set(schema.get("properties", {})) - INJECTED
            server_required = set(schema.get("required", [])) - INJECTED
            local = SCHEMA_BY_NAME[name]
            if server_props != set(local["properties"]) or server_required != set(local["required"]):
                raise MCPToolError(f"工具 {name} 的参数与本地 schema 不一致：{sorted(server_props)} / {sorted(server_required)}")
            if not INJECTED <= set(schema.get("required", [])):
                raise MCPToolError(f"工具 {name} 没有要求可信参数 {sorted(INJECTED)}")

    # ---------- 调用 ----------

    def call(self, name: str, args: dict, ctx: ToolContext) -> dict:
        """args 是模型给的、已经 prepare_args 规范化过的参数；这里补上可信参数后发给 Server。"""
        if name not in MCP_TOOLS:
            raise MCPToolError(f"{name} 不是 MCP 工具")
        full_args = {**args, "user_id": ctx.user_id, "request_id": ctx.request_id}
        try:
            return self._call_once(name, full_args)
        except TimeoutError:
            raise  # 超时不重试：可能只是慢，重试会叠加等待时间
        except Exception:
            # 断连等错误：重启 Server 再试一次。transfer_to_human 有幂等键，重试不会重复建单
            self.reconnects += 1
            self._disconnect()
            self._connect()
            return self._call_once(name, full_args)

    def _call_once(self, name: str, full_args: dict) -> dict:
        async def _call():
            with fail_after(self.timeout_s):
                return await self._client.call_tool(name, full_args)
        result = self._portal.call(_call)
        if result.is_error:
            text = result.content[0].text if result.content else ""
            raise MCPToolError(f"MCP 工具 {name} 执行失败：{text[:200]}")
        if isinstance(result.structured_content, dict):
            data = result.structured_content
            return data.get("result", data) if set(data) == {"result"} else data
        return json.loads(result.content[0].text)
