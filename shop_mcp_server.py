"""
订单 / 物流 / 模拟工单的 MCP Server（官方 mcp Python SDK 2.x 的 MCPServer，stdio 传输）。

- 由 mcp_client.py 以子进程方式启动，通过 stdin/stdout 说 MCP 协议；stdout 只能用于协议，日志一律写 stderr。
- 业务逻辑复用 shop_tools.ShopTools 的订单类方法（底层是 shop_repository），这里不重写，保证和本地模式结果一致。
- user_id、request_id 是"可信适配器"（mcp_client）注入的参数，模型看不到也传不了；
  服务端仍然逐次检查订单归属（在 shop_repository 里），并再校验一遍模型可传的参数。
- 演示用的身份传递方式不是生产鉴权：stdio 只在本机父子进程之间通信，不暴露到网络。
"""
import sys

from mcp.server.mcpserver import MCPServer

from shop_repository import ShopRepository
from shop_tools import ShopTools, ToolContext, err, prepare_args

server = MCPServer("securag-shop-orders", instructions="SecuRAG 电商客服的订单、物流和模拟工单工具（演示数据）")
repo = ShopRepository()
tools = ShopTools(repo)  # 不带知识库：这个进程只提供订单类工具，启动时不加载检索模型

TRUSTED = "由客户端适配器注入的可信参数，不是模型参数"


def _run(name: str, user_id: str, request_id: str, model_args: dict) -> dict:
    print(f"[shop-mcp] call {name} user={user_id} args={model_args}", file=sys.stderr, flush=True)
    if user_id not in repo.users:
        return err("UNAUTHORIZED", "未知用户")
    args, error_msg = prepare_args(name, model_args)  # 服务端再校验一次，不完全信任客户端
    if error_msg:
        return err("BAD_ARGUMENTS", error_msg)
    return tools.execute(name, args, ToolContext(user_id=user_id, request_id=request_id))


@server.tool(description="查询当前用户的订单。传 order_id 查单个订单详情（含签收日期和退换货期限）；不传则列出最近的订单。")
def get_order(user_id: str, request_id: str, order_id: str = "") -> dict:
    """user_id / request_id：由客户端适配器注入的可信参数，不是模型参数。"""
    return _run("get_order", user_id, request_id, {"order_id": order_id} if order_id else {})


@server.tool(description="查询当前用户某个订单的物流：快递公司、运单号、最新轨迹、是否停滞。")
def track_shipment(user_id: str, request_id: str, order_id: str) -> dict:
    """user_id / request_id：由客户端适配器注入的可信参数，不是模型参数。"""
    return _run("track_shipment", user_id, request_id, {"order_id": order_id})


@server.tool(description="生成模拟人工客服工单。用于：用户要求人工、投诉、需要人工审核的质量问题/退款/催件、或自己无法解决的问题。")
def transfer_to_human(user_id: str, request_id: str, reason_code: str, summary: str) -> dict:
    """user_id / request_id：由客户端适配器注入的可信参数，不是模型参数。request_id 用于工单幂等。"""
    return _run("transfer_to_human", user_id, request_id, {"reason_code": reason_code, "summary": summary})


if __name__ == "__main__":
    print("[shop-mcp] server starting (stdio)", file=sys.stderr, flush=True)
    server.run("stdio")
