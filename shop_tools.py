"""
电商客服的工具：给模型看的说明（schema）、参数校验、具体实现。

约定：
- 每个工具返回 {"ok": bool, "data": ..., "error": {"code", "message", "retryable"}, "source_ids": [...]}
- 模型能传的参数只有 schema 里列出的；user_id、工单幂等键由程序通过 ToolContext 注入
- 返回给模型之前统一过一遍 PII 脱敏
"""
import re
from dataclasses import dataclass

import config
from security_guard import redact_pii
from shop_repository import ShopRepository

ORDER_ID_PATTERN = r"ORD-DEMO-\d{4}"
REASON_CODES = ["USER_REQUEST", "COMPLAINT", "QUALITY_ISSUE", "LOGISTICS_ISSUE", "REFUND_ISSUE", "OTHER"]


@dataclass
class ToolContext:
    """程序注入给工具的可信信息，模型看不到也改不了。"""
    user_id: str
    request_id: str  # 每个用户轮一个，用来生成工单幂等键


def ok(data, source_ids=None) -> dict:
    return {"ok": True, "data": data, "error": None, "source_ids": source_ids or []}


def err(code: str, message: str, retryable: bool = False) -> dict:
    return {"ok": False, "data": None, "error": {"code": code, "message": message, "retryable": retryable},
            "source_ids": []}


def _schema(name, description, properties, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required),
                       "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _schema("search_products", "在商品说明和商品常见问题里检索，回答商品功能、参数、价格、库存、使用方法等问题。",
            {"query": {"type": "string", "maxLength": 100, "description": "检索用的问题或关键词"}}, ["query"]),
    _schema("search_policy", "检索售后政策：退货、换货、质量问题、保修、运费、退款时效、价保、发票、投诉、物流异常等规则。",
            {"query": {"type": "string", "maxLength": 100, "description": "检索用的问题或关键词"}}, ["query"]),
    _schema("get_order", "查询当前用户的订单。传 order_id 查单个订单详情（含签收日期和退换货期限）；不传则列出最近的订单。",
            {"order_id": {"type": "string", "maxLength": 20, "pattern": ORDER_ID_PATTERN,
                          "description": "订单号，格式如 ORD-DEMO-0001"}}),
    _schema("track_shipment", "查询当前用户某个订单的物流：快递公司、运单号、最新轨迹、是否停滞。",
            {"order_id": {"type": "string", "maxLength": 20, "pattern": ORDER_ID_PATTERN,
                          "description": "订单号，格式如 ORD-DEMO-0001"}}, ["order_id"]),
    _schema("transfer_to_human", "生成模拟人工客服工单。用于：用户要求人工、投诉、需要人工审核的质量问题/退款/催件、或自己无法解决的问题。",
            {"reason_code": {"type": "string", "enum": REASON_CODES, "description": "转人工原因"},
             "summary": {"type": "string", "maxLength": 200, "description": "给人工客服的问题摘要，不要写用户隐私"}},
            ["reason_code", "summary"]),
]
SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}


def register_schema(schema: dict) -> None:
    """登记额外工具（如阶段3 的 load_skill）的参数 schema，让 prepare_args 也能校验它。"""
    SCHEMA_BY_NAME[schema["function"]["name"]] = schema["function"]["parameters"]


def prepare_args(name: str, args) -> tuple:
    """
    校验 + 规范化参数，返回 (规范化后的参数, 错误信息)。
    规范化（去首尾空格、订单号转大写、丢掉空的可选参数）之后的参数，也用来做重复调用检测。
    """
    if name not in SCHEMA_BY_NAME:
        return None, f"没有名为 {name} 的工具"
    if not isinstance(args, dict):
        return None, "参数必须是 JSON 对象"
    schema = SCHEMA_BY_NAME[name]
    props = schema["properties"]
    unknown = set(args) - set(props)
    if unknown:
        return None, f"不支持的参数：{sorted(unknown)}"

    clean = {}
    for key, value in args.items():
        if not isinstance(value, str):
            return None, f"参数 {key} 必须是字符串"
        value = value.strip()
        if key == "order_id":
            value = value.upper()
        if value == "" and key not in schema["required"]:
            continue  # 可选参数传了空串，当作没传
        spec = props[key]
        if len(value) > spec.get("maxLength", 10_000):
            return None, f"参数 {key} 太长（最多 {spec['maxLength']} 字）"
        if "enum" in spec and value not in spec["enum"]:
            return None, f"参数 {key} 只能是 {spec['enum']} 之一"
        if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
            return None, f"参数 {key} 格式不对，应类似 ORD-DEMO-0001"
        clean[key] = value
    missing = [k for k in schema["required"] if k not in clean]
    if missing:
        return None, f"缺少必填参数：{missing}"
    return clean, None


def redact_obj(obj):
    """递归脱敏工具返回的所有字符串。"""
    if isinstance(obj, str):
        return redact_pii(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    return obj


def yuan(cents: int) -> str:
    return f"¥{cents / 100:.2f}"


class ShopKnowledge:
    """商品、政策两个知识集合的检索：混合检索取 20 条候选 → CrossEncoder 重排取前几条。"""

    def __init__(self, embed_model: str = config.SHOP_EMBED_MODEL, rerank_model: str = config.SHOP_RERANK_MODEL):
        # 检索相关的库放到这里才导入：MCP Server 只用订单类工具，启动时不用加载 chromadb 和模型
        import chromadb
        from build_shop_kb import collection_names, embedding_function
        from hybrid_retriever import HybridRetriever
        from reranker import Reranker

        client = chromadb.PersistentClient(path=config.SHOP_DB_PATH)
        ef = embedding_function(embed_model)
        prefix = "" if embed_model == "default" else config.SHOP_EMBED_QUERY_PREFIX
        self.retrievers = {
            kind: HybridRetriever(client.get_collection(name, embedding_function=ef), query_prefix=prefix)
            for kind, name in collection_names(embed_model).items()}
        self.reranker = Reranker(rerank_model)

    def search(self, kind: str, query: str, top_k: int = 3, mode: str = "hybrid", rerank: bool = True) -> list:
        candidates = self.retrievers[kind].retrieve_with_ids(query, n_results=20, mode=mode)
        if not rerank:
            return candidates[:top_k]
        return self.reranker.rerank_with_scores(query, candidates, top_k=top_k)


class ShopTools:
    def __init__(self, repo: ShopRepository, knowledge: ShopKnowledge = None):
        """knowledge 为 None 时只能用订单类工具（MCP Server 就是这样用的）。"""
        self.repo = repo
        self.knowledge = knowledge
        self.schemas = TOOL_SCHEMAS
        self._impl = {
            "search_products": self.search_products, "search_policy": self.search_policy,
            "get_order": self.get_order, "track_shipment": self.track_shipment,
            "transfer_to_human": self.transfer_to_human,
        }

    def execute(self, name: str, args: dict, ctx: ToolContext) -> dict:
        """args 必须是 prepare_args 规范化过的。异常不在这里吞，交给 agent_loop 统一兜住。"""
        return redact_obj(self._impl[name](ctx, **args))

    # ---------- 知识检索 ----------

    def search_products(self, ctx, query):
        hits = self.knowledge.search("product", query, top_k=3)
        results, seen = [], set()
        for h in hits:
            pid = h["metadata"]["source_id"]
            p = self.repo.products[pid]
            entry = {"chunk_id": h["id"], "相关度": round(h["rerank_score"], 3), "内容": h["text"]}
            if pid not in seen:  # 同一商品的价格库存只列一次，价格库存来自结构化数据
                seen.add(pid)
                entry.update({"product_id": pid, "商品": p["name"], "价格": yuan(p["price_cents"]),
                              "库存": f"有货（{p['stock']}件）" if p["stock"] > 0 else "缺货",
                              "支持七天无理由": p["no_reason_return"], "保修": f"{p['warranty_months']}个月"})
            results.append(entry)
        return ok({"results": results}, [h["id"] for h in hits])

    def search_policy(self, ctx, query):
        hits = self.knowledge.search("policy", query, top_k=3)
        results = [{"chunk_id": h["id"], "政策": h["metadata"]["title"], "小节": h["metadata"]["section"],
                    "相关度": round(h["rerank_score"], 3), "内容": h["text"]} for h in hits]
        return ok({"results": results}, [h["id"] for h in hits])

    # ---------- 订单与物流（阶段2 改为经 MCP 调用）----------

    def get_order(self, ctx, order_id=None):
        if order_id is None:
            orders = self.repo.list_orders(ctx.user_id)
            return ok({"最近订单": [{"order_id": o["order_id"], "状态": o["status"], "下单时间": o["created_at"][:10],
                                    "商品": "、".join(it["name"] for it in o["items"]), "实付": yuan(o["total_cents"])}
                                   for o in orders]})
        order = self.repo.get_order(ctx.user_id, order_id)
        if order is None:
            return err("NOT_FOUND", f"当前账号下没有找到订单 {order_id}")
        return ok({
            "order_id": order["order_id"], "状态": order["status"],
            "商品": [{"product_id": it["product_id"], "名称": it["name"], "数量": it["qty"],
                     "单价": yuan(it["unit_price_cents"])} for it in order["items"]],
            "商品金额": yuan(order["goods_cents"]), "运费": yuan(order["shipping_cents"]), "实付": yuan(order["total_cents"]),
            "下单时间": order["created_at"], "付款时间": order["paid_at"], "发货时间": order["shipped_at"],
            "签收时间": order["signed_at"], "售后期限": self.repo.after_sales_windows(order),
        }, [order["order_id"]])

    def track_shipment(self, ctx, order_id):
        order, ship = self.repo.get_shipment(ctx.user_id, order_id)
        if order is None:
            return err("NOT_FOUND", f"当前账号下没有找到订单 {order_id}")
        if ship is None:
            return ok({"order_id": order_id, "订单状态": order["status"], "物流": "该订单还没有物流信息（未发货或已取消）"})
        return ok({"order_id": order_id, "快递公司": ship["carrier"], "运单号": ship["tracking_no"],
                   "物流状态": ship["status"], "是否停滞超过72小时": self.repo.is_stalled(ship),
                   "最新轨迹": ship["events"][-1], "近期轨迹": ship["events"][-3:]}, [order_id])

    # ---------- 模拟转人工 ----------

    def transfer_to_human(self, ctx, reason_code, summary):
        # 幂等键由程序生成：同一用户轮内无论模型调用几次，只建一张单
        ticket = self.repo.create_ticket(ctx.user_id, f"{ctx.user_id}:{ctx.request_id}", reason_code,
                                         redact_pii(summary))
        return ok({"工单号": ticket["ticket_id"], "状态": ticket["status"], "是否重复提交": ticket["duplicate"],
                   "说明": "这是模拟工单，用于演示转人工流程"}, [ticket["ticket_id"]])
