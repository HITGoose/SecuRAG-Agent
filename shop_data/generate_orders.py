"""
用固定随机种子生成模拟用户、订单和物流数据（全部虚构）。

运行：python shop_data/generate_orders.py
输出：shop_data/users.json、orders.json、shipments.json

边界样本是显式写死的（签收第7天/第8天、物流停滞、定制商品、个护商品、只有一笔订单的用户），
随机数只用来挑商品、数量和具体时刻，保证每次生成结果完全一样。
"""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 20261103
TZ = timezone(timedelta(hours=8))  # Asia/Shanghai
NOW = datetime(2026, 11, 3, 12, 0, 0, tzinfo=TZ)  # 与 config.SIMULATION_NOW 一致
FREE_SHIPPING_CENTS = 9900
SHIPPING_FEE_CENTS = 800

USERS = [
    {"user_id": "USR-DEMO-01", "nickname": "演示用户小林", "phone": "演示手机号-01", "address": "演示省演示市演示路1号（虚构）"},
    {"user_id": "USR-DEMO-02", "nickname": "演示用户阿青", "phone": "演示手机号-02", "address": "演示省演示市演示路2号（虚构）"},
    {"user_id": "USR-DEMO-03", "nickname": "演示用户大周", "phone": "演示手机号-03", "address": "演示省演示市演示路3号（虚构）"},
    {"user_id": "USR-DEMO-04", "nickname": "演示用户小满", "phone": "演示手机号-04", "address": "演示省演示市演示路4号（虚构）"},
    {"user_id": "USR-DEMO-05", "nickname": "演示用户老陈", "phone": "演示手机号-05", "address": "演示省演示市演示路5号（虚构）"},
]

CARRIERS = [("SFX", "顺丰演示快递"), ("YTX", "圆通演示快递"), ("ZTX", "中通演示快递")]

# (状态, 参数, 指定商品) —— 参数含义：签收类为"签收距今天数"，运输类为"发货距今天数"
SCENARIOS = (
    [("待付款", 0, None)] * 2 + [("待付款", 1, None)]
    + [("待发货", 0, None), ("待发货", 1, None), ("待发货", 1, None), ("待发货", 2, None), ("待发货", 2, None)]
    + [("已发货", 1, None), ("已发货", 2, None), ("已发货", 2, None), ("已发货", 3, None)]
    + [("物流停滞", 5, None), ("物流停滞", 6, None), ("物流停滞", 6, None)]
    + [("已签收", 1, None), ("已签收", 2, "PRD-DEMO-008"), ("已签收", 3, "PRD-DEMO-301"),
       ("已签收", 5, None), ("已签收", 6, "PRD-DEMO-108"), ("已签收", 7, None), ("已签收", 7, "PRD-DEMO-205"),
       ("已签收", 8, None), ("已签收", 9, None), ("已签收", 12, None), ("已签收", 14, None),
       ("已签收", 15, None), ("已签收", 16, None), ("已签收", 20, None), ("已签收", 30, None)]
    + [("退款中", 4, None), ("退款中", 6, None), ("退款中", 0, None)]
    + [("已完成", 35, None), ("已完成", 40, None), ("已完成", 50, None), ("已完成", 60, None)]
    + [("已取消", 2, None), ("已取消", 5, None), ("已取消", 10, None)]
)
assert len(SCENARIOS) == 40

# 只有一笔订单的用户：USR-DEMO-05 只拿"签收5天"这一单，用来测"不用澄清就能确定订单"
SINGLE_ORDER_USER = "USR-DEMO-05"
SINGLE_ORDER_INDEX = SCENARIOS.index(("已签收", 5, None))


def fmt(dt):
    return dt.isoformat(timespec="seconds")


def make_items(rng, products, fixed_pid):
    if fixed_pid:
        chosen = [next(p for p in products if p["product_id"] == fixed_pid)]
    else:
        chosen = rng.sample(products, rng.choice([1, 1, 1, 2]))
    return [{"product_id": p["product_id"], "name": p["name"], "qty": rng.choice([1, 1, 2]),
             "unit_price_cents": p["price_cents"]} for p in chosen]


def make_shipment(rng, order_id, shipped_at, signed_at, stalled):
    code, carrier = rng.choice(CARRIERS)
    events = [{"time": fmt(shipped_at), "desc": "商家已发货，快递已揽收", "location": "演示仓"}]
    t = shipped_at
    hops = ["演示转运中心A", "演示转运中心B", "演示目的城市网点"]
    for hop in hops[: (1 if stalled else 3)]:
        t = t + timedelta(hours=rng.randint(10, 20))
        if signed_at and t >= signed_at:
            break
        if not signed_at and t >= NOW:
            break
        events.append({"time": fmt(t), "desc": f"快件到达{hop}", "location": hop})
    if signed_at:
        events.append({"time": fmt(signed_at), "desc": "已签收，签收人：本人", "location": "收件地址"})
        status = "已签收"
    else:
        status = "运输中"
    return {"order_id": order_id, "tracking_no": f"{code}-DEMO-{rng.randint(0, 99999999):08d}",
            "carrier": carrier, "status": status, "events": events}


def main():
    rng = random.Random(SEED)
    products = json.loads((DATA_DIR / "products.json").read_text(encoding="utf-8"))["products"]
    in_stock = [p for p in products if p["stock"] > 0]
    multi_users = [u["user_id"] for u in USERS if u["user_id"] != SINGLE_ORDER_USER]

    orders, shipments = [], []
    for i, (status, days, fixed_pid) in enumerate(SCENARIOS):
        order_id = f"ORD-DEMO-{i + 1:04d}"
        user_id = SINGLE_ORDER_USER if i == SINGLE_ORDER_INDEX else multi_users[i % len(multi_users)]
        items = make_items(rng, in_stock, fixed_pid)
        goods = sum(it["qty"] * it["unit_price_cents"] for it in items)
        shipping = 0 if goods >= FREE_SHIPPING_CENTS else SHIPPING_FEE_CENTS

        created = paid = shipped = signed = None
        if status == "待付款":
            created = NOW - timedelta(days=days, hours=rng.randint(1, 10))
        elif status == "待发货":
            created = NOW - timedelta(days=days, hours=rng.randint(1, 10))
            paid = created + timedelta(minutes=5)
        elif status in ("已发货", "物流停滞"):
            shipped = NOW - timedelta(days=days, hours=rng.randint(1, 6))
            created = shipped - timedelta(days=1)
            paid = created + timedelta(minutes=5)
        elif status in ("已签收", "退款中", "已完成"):
            signed = (NOW - timedelta(days=days)).replace(hour=rng.randint(9, 20), minute=rng.randint(0, 59))
            shipped = signed - timedelta(days=2, hours=rng.randint(0, 8))
            created = shipped - timedelta(days=1)
            paid = created + timedelta(minutes=5)
        elif status == "已取消":
            created = NOW - timedelta(days=days, hours=rng.randint(1, 10))

        order = {
            "order_id": order_id, "user_id": user_id, "status": "已发货" if status == "物流停滞" else status,
            "items": items, "goods_cents": goods, "shipping_cents": shipping, "total_cents": goods + shipping,
            "created_at": fmt(created), "paid_at": fmt(paid) if paid else None,
            "shipped_at": fmt(shipped) if shipped else None, "signed_at": fmt(signed) if signed else None,
        }
        orders.append(order)
        if shipped:
            shipments.append(make_shipment(rng, order_id, shipped, signed, stalled=(status == "物流停滞")))

    meta = {"version": "orders-v1", "seed": SEED, "simulation_now": fmt(NOW), "note": "全部为虚构数据"}
    (DATA_DIR / "users.json").write_text(json.dumps({**meta, "users": USERS}, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA_DIR / "orders.json").write_text(json.dumps({**meta, "orders": orders}, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA_DIR / "shipments.json").write_text(json.dumps({**meta, "shipments": shipments}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"生成 {len(USERS)} 个用户、{len(orders)} 笔订单、{len(shipments)} 条物流")


if __name__ == "__main__":
    main()
