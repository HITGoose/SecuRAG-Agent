"""
v2 电商客服的演示页面（阶段5，最小实现）。v1 的 frontend.py / dashboard.py 保持不变。
运行：streamlit run shop_frontend.py --server.fileWatcherType none
（关掉文件监视：否则 streamlit 会逐个检查 transformers 的子模块，启动很慢、日志刷屏）

每条回复下面有一个可折叠的"执行过程"：工具名、脱敏后的参数、结果状态、耗时，以及每次模型调用。
所有商品、订单、物流都是虚构的演示数据。
"""
import uuid

import pandas as pd
import streamlit as st

from shop_agent import ShopAgent

st.set_page_config(page_title="SecuRAG 电商客服（演示）", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="正在加载模型和知识库……")
def get_agent(provider: str, backend: str, skills: str) -> ShopAgent:
    return ShopAgent(provider=provider, tool_backend=backend, skills=skills)


with st.sidebar:
    st.header("设置")
    user = st.selectbox("模拟登录用户", ["USR-DEMO-01", "USR-DEMO-02", "USR-DEMO-03", "USR-DEMO-04", "USR-DEMO-05"])
    provider = st.selectbox("模型", ["deepseek", "ollama"], help="deepseek 失败时自动降级到本地 qwen2.5:7b")
    backend = st.selectbox("订单类工具通道", ["local", "mcp"])
    skills = st.selectbox("业务技能", ["off", "ondemand", "full"])
    if st.button("新会话") or "sid" not in st.session_state or st.session_state.get("login") != (user, provider, backend, skills):
        st.session_state.sid = uuid.uuid4().hex
        st.session_state.login = (user, provider, backend, skills)
        st.session_state.messages = []
    st.caption("全部为虚构演示数据，不代表任何真实商家。")

agent = get_agent(provider, backend, skills)
if st.session_state.sid not in agent.sessions:
    agent.login(st.session_state.sid, user)

st.title("🛒 SecuRAG 电商客服（演示）")


def show_process(res: dict) -> None:
    flags = f"意图 {res['intent']} → 路由 {res['route']}，停止原因 {res['stop_reason']}，事实检查 {res.get('fact_check')}"
    with st.expander(f"执行过程（{len(res['steps'])} 次工具调用，{res['llm_calls']} 次模型调用，{res['latency_ms']} ms）"):
        st.caption(flags)
        if res["steps"]:
            st.dataframe(pd.DataFrame([{
                "步骤": s.get("step"), "工具": s["tool"], "参数（脱敏后）": str(s.get("args")),
                "状态": s["status"] + (f" / {s['error_code']}" if s.get("error_code") else ""),
                "通道": s.get("via", "local"), "耗时ms": s.get("latency_ms")} for s in res["steps"]]),
                hide_index=True, use_container_width=True)
        if res.get("llm"):
            st.dataframe(pd.DataFrame([{
                "用途": c["purpose"], "后端": f"{c['provider']}/{c['model']}", "耗时ms": c["latency_ms"],
                "token": f"{c['input_tokens']}+{c['output_tokens']}", "重试": c.get("retries", 0),
                "降级原因": c.get("fallback_reason") or ""} for c in res["llm"]]),
                hide_index=True, use_container_width=True)
        if res.get("trace_id"):
            st.caption(f"trace_id：{res['trace_id']}（命令行查看：python view_traces.py {res['trace_id']}）")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result"):
            show_process(m["result"])

if prompt := st.chat_input("请输入问题，例如：我的订单到哪了"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("处理中……"):
            res = agent.handle(st.session_state.sid, prompt)
        st.markdown(res["answer"])
        show_process(res)
    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "result": res})
