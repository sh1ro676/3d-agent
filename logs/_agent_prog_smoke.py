"""零成本冒烟程序：验执行器 + 工具库 + submit 契约（不调 LLM）。

它刻意写成「模型应当写出来的样子」：一律关键字传参、先判 ok、带证据、以 submit 结束。
"""

objs = list_objects()
if not objs.ok:
    submit("unknown", evidence=["list_objects 失败：" + str(objs.error.code)])

items = objs.value
if len(items) < 2:
    submit("unknown", evidence=["场景里物体少于两个，无法算两两距离"])

labels = sorted(set(o["label"] for o in items))
a = items[0]["object_id"]
b = items[1]["object_id"]

pair = calculate_distance(a=a, b=b)
if not pair.ok:
    submit("unknown", evidence=["calculate_distance 失败：" + str(pair.error.code)])

extent = get_3d_extent(object_id=a)
pos = get_3d_position(object_id=b)

submit(
    pair.value,
    target_ids=[a, b],
    evidence=[
        "labels = " + ",".join(labels),
        "calculate_distance(a=%s, b=%s) = %.4f m" % (a, b, pair.value),
        "formula = " + str(pair.evidence.get("formula")),
        "get_3d_extent(%s) = %s" % (a, extent.value if extent.ok else extent.error.code),
        "get_3d_position(%s) = %s" % (b, pos.value if pos.ok else pos.error.code),
    ],
)
