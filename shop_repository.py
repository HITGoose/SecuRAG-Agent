"""
模拟业务数据层：读取订单、物流，创建模拟工单。

权限在这一层强制执行：每个函数都要传 user_id（由程序从会话注入，模型传不了），
订单不存在和不属于当前用户返回同样的 None，不暴露别人的订单是否存在。
阶段2 的 MCP Server 也直接复用这个文件，不复制业务逻辑。
"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import config

DATA_DIR = Path(config.SHOP_DATA_DIR)
NOW = datetime.fromisoformat(config.SIMULATION_NOW)
STALL_HOURS = 72  # 与 POL-LOGISTICS 一致：超过72小时没更新视为停滞


def _load(name: str, key: str) -> list:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))[key]


class ShopRepository:
    def __init__(self, runtime_dir: str = config.SHOP_RUNTIME_DIR):
        self.products = {p["product_id"]: p for p in _load("products.json", "products")}
        self.orders = {o["order_id"]: o for o in _load("orders.json", "orders")}
        self.shipments = {s["order_id"]: s for s in _load("shipments.json", "shipments")}
        self.users = {u["user_id"]: u for u in _load("users.json", "users")}
        self.ticket_file = Path(runtime_dir) / "tickets.json"
        self._lock = threading.Lock()

    # ---------- 订单 ----------

    def get_order(self, user_id: str, order_id: str):
        order = self.orders.get(order_id.strip().upper())
        if order is None or order["user_id"] != user_id:
            return None  # 不存在 / 别人的订单，一律当作找不到
        return order

    def list_orders(self, user_id: str, limit: int = 5) -> list:
        mine = [o for o in self.orders.values() if o["user_id"] == user_id]
        return sorted(mine, key=lambda o: o["created_at"], reverse=True)[:limit]

    def after_sales_windows(self, order: dict) -> dict:
        """
        用代码算退换货期限，不让模型自己数日子。
        规则（POL-RETURN-7D / POL-EXCHANGE）：从签收次日零点起算，满 N 天当天 24 点截止。
        只算"时间上是否还在期限内"；商品是否完好、是否质量问题，工具不知道，需要问用户或人工审核。
        """
        if not order.get("signed_at"):
            return {"已签收": False}
        signed_day = datetime.fromisoformat(order["signed_at"]).date()

        def deadline(days):
            end = datetime.combine(signed_day + timedelta(days=days), datetime.max.time(), tzinfo=NOW.tzinfo)
            return {"截止时间": end.strftime("%Y-%m-%d 24:00前"), "是否在期限内": NOW <= end}

        items = [{"product_id": it["product_id"], "name": it["name"],
                  "商品是否支持七天无理由": self.products[it["product_id"]]["no_reason_return"],
                  "类目": self.products[it["product_id"]]["category"]} for it in order["items"]]
        return {"已签收": True, "签收日期": signed_day.isoformat(),
                "七天无理由退货": deadline(7), "质量问题退货": deadline(7), "质量问题换货": deadline(15),
                "商品": items}

    # ---------- 物流 ----------

    def get_shipment(self, user_id: str, order_id: str):
        """先查订单归属，再查物流；返回 (订单, 物流)，订单不可见时返回 (None, None)。"""
        order = self.get_order(user_id, order_id)
        if order is None:
            return None, None
        return order, self.shipments.get(order["order_id"])

    @staticmethod
    def is_stalled(shipment: dict) -> bool:
        if shipment["status"] != "运输中":
            return False
        last = datetime.fromisoformat(shipment["events"][-1]["time"])
        return NOW - last > timedelta(hours=STALL_HOURS)

    # ---------- 模拟工单 ----------

    def create_ticket(self, user_id: str, idem_key: str, reason_code: str, summary: str) -> dict:
        """
        幂等：同一个 idem_key（程序生成，模型传不了）只建一张单，重复调用返回原工单。
        这样模型重复调用、重试、降级都不会重复建单。
        """
        with self._lock:
            tickets = json.loads(self.ticket_file.read_text(encoding="utf-8")) if self.ticket_file.exists() else []
            for t in tickets:
                if t["idem_key"] == idem_key:
                    return {**t, "duplicate": True}
            ticket = {"ticket_id": f"TKT-DEMO-{len(tickets) + 1:05d}", "user_id": user_id, "idem_key": idem_key,
                      "reason_code": reason_code, "summary": summary, "status": "等待模拟人工处理",
                      "created_at": NOW.isoformat(timespec="seconds")}
            tickets.append(ticket)
            self.ticket_file.parent.mkdir(parents=True, exist_ok=True)
            self.ticket_file.write_text(json.dumps(tickets, ensure_ascii=False, indent=1), encoding="utf-8")
            return {**ticket, "duplicate": False}
