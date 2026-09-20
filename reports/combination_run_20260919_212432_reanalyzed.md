# 组合题真跑诊断 —— points_probe

真值来自 `scripts/probe_combination.py` 的同一份 SPECS（工具算出来的，不是脚本算的）。

## 归因汇总

- 题数 10 ｜ 与 GT 一致 **10** ｜ 成本 ¥0.0907
- 结局分布：`{"C": 3, "OK": 5, "OK*": 2}`
- 状态分布：`{"ok": 10}`
- 用了组合（推导式 / 带 key 算子 / `key=`）：**9/10** ｜ 出现 `key=`：1 ｜ 出现推导式：9
- ★ **答对却被证据契约判 `unsupported`：3 题**（与准确率无关）
- 工具调用：实际 106 次 vs 最少 70 次 ｜ 冗余倍数中位数 **5.0**

`class` 的含义：`OK/OK*` = 答对（证据 supported / weak）；`C` = 答对但证据被拒收；
`A` = 答错且没用组合；`B` = 答错但用了组合。**A 与 B 只描述失败**，
答对的题不归到 A/B —— 「工具已经覆盖这个组合」不该被统计成「模型不会组合」。

`冗余倍数` = 实际调用次数 / 完美程序调用次数。>1 说明**模型没利用返回值里已有的字段**，
而不是「工具不够用」—— 这一列把「功能单薄」与「没看清返回值」分开了。

## 逐题

| 题号 | class | 状态 | 答案 | GT | 证据 | 组合 | 算子 | 调用(实际/最少) | 倍数 |
|---|---|---|---|---|---|---|---|---|---|
| C1 | **OK** | ok | `picture_1` | `picture_1` | supported | 是 | sorted( | 5/1 | 5.0 |
| C2 | **OK** | ok | `picture_2` | `picture_2` | supported | **否** | — | 3/1 | 3.0 |
| C3 | **OK** | ok | `mirror_1` | `mirror_1` | supported | 是 | len( | 10/1 | 10.0 |
| C4 | **OK** | ok | `4` | `4` | supported | 是 | len( | 10/1 | 10.0 |
| C5 | **C** | ok | `0.4445499926805496` | `0.4445` | unsupported | 是 | mean(,len( | 5/1 | 5.0 |
| C6 | **OK*** | ok | `picture_4,picture_2,picture_3,picture_1` | `picture_4,picture_2,picture_3,picture_1` | weak | 是 | len( | 5/1 | 5.0 |
| C7 | **OK** | ok | `table_1` | `table_1` | supported | 是 | — | 10/9 | 1.11 |
| C8 | **C** | ok | `0` | `0` | unsupported | 是 | — | 11/9 | 1.22 |
| C8b | **C** | ok | `3` | `3` | unsupported | 是 | len( | 10/9 | 1.11 |
| C12 | **OK*** | ok | `table_1,picture_4` | `picture_4,table_1` | weak | 是 | len( | 37/37 | 1.0 |

### C1 —— 四幅画里哪一幅面积最大？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `picture_1` ｜ GT `picture_1` ｜ 证据 `supported` ｜ 工具调用序列 `find_object → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent`
- 组合 `是` ｜ 调用 5 次（完美程序 1 次，倍数 5.0）｜ 生成 1 次 ｜ 耗时 2.62 s

```python
from itertools import product

scene_id = "points_probe"

# 1) 找出所有 picture
res = find_object(scene_id=scene_id, label="picture")
if not res.ok:
    submit("unknown", evidence=["find_object(picture) failed: %s" % res.error.code])
    raise SystemExit

pics = res.value or []
if not pics:
    submit("unknown", evidence=["no picture objects found in scene"])
    raise SystemExit

# 2) 逐个取三维尺寸，算面积（取最大的两个维度之积作为画面面积）
best_id = None
best_area = -1.0
evidence = []

for p in pics:
    oid = p.get("object_id")
    if not oid:
        continue
    er = get_3d_extent(scene_id=scene_id, object_id=oid)
    if not er.ok:
        evidence.append("get_3d_extent(%s) failed: %s" % (oid, er.error.code))
        continue
    ext = er.value
    dims = sorted([ext["w"], ext["h"], ext["l"]], reverse=True)
    area = dims[0] * dims[1]
    evidence.append("%s extent=(w=%.4f,h=%.4f,l=%.4f) area=%.6f m^2" % (oid, ext["w"], ext["h"], ext["l"], area))
    if area > best_area:
        best_area = area
        best_id = oid

if best_id is None:
    submit("unknown", evidence=["could not compute extent for any picture"] + evidence)
else:
    submit(best_id, target_ids=[best_id], evidence=evidence)
```

### C2 —— 哪幅画离镜子最近？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `picture_2` ｜ GT `picture_2` ｜ 证据 `supported` ｜ 工具调用序列 `single_object → find_object → find_nearest`
- 组合 `**否**` ｜ 调用 3 次（完美程序 1 次，倍数 3.0）｜ 生成 1 次 ｜ 耗时 1.683 s

```python
from itertools import count

scene_id = "points_probe"

# 1) 找到镜子
res = single_object(scene_id=scene_id, label="mirror")
if not res.ok:
    submit("unknown", evidence=["single_object(mirror) failed: %s" % res.error.code])
else:
    mirror = res.value
    mirror_id = mirror["object_id"]

    # 2) 找到所有画
    res2 = find_object(scene_id=scene_id, label="picture")
    if not res2.ok:
        submit("unknown", evidence=["find_object(picture) failed: %s" % res2.error.code])
    else:
        pictures = res2.value
        if not pictures:
            submit("unknown", evidence=["no picture objects in scene"])
        else:
            # 3) 用 find_nearest 直接取离镜子最近的画
            res3 = find_nearest(scene_id=scene_id, anchor=mirror_id, label="picture", k=1)
            if not res3.ok:
                submit("unknown", evidence=["find_nearest failed: %s" % res3.error.code])
            else:
                nearest = res3.value
                if not nearest:
                    submit("unknown", evidence=["find_nearest returned empty"])
                else:
                    best = nearest[0]
                    submit(
                        best["object_id"],
                        target_ids=[best["object_id"]],
                        evidence=[
                            "find_nearest(anchor=%s, label=picture, k=1) -> object_id=%s, distance_m=%s"
                            % (mirror_id, best["object_id"], best.get("distance_m"))
                        ],
                    )
```

### C3 —— 场景里最高的物体是哪个？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `mirror_1` ｜ GT `mirror_1` ｜ 证据 `supported` ｜ 工具调用序列 `list_objects → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent`
- 组合 `是` ｜ 调用 10 次（完美程序 1 次，倍数 10.0）｜ 生成 1 次 ｜ 耗时 1.729 s

```python
from itertools import groupby

scene_id = "points_probe"

# 1) 列出场景中所有物体
res = list_objects(scene_id=scene_id)
if not res.ok:
    submit("unknown", evidence=[f"list_objects failed: {res.error.code}"])

objs = res.value

# 2) 逐个取三维尺寸，找最高的（h 最大）
best_id = None
best_h = None
best_label = None
details = []

for obj in objs:
    oid = obj.get("object_id")
    if not oid:
        continue
    ext_res = get_3d_extent(scene_id=scene_id, object_id=oid)
    if not ext_res.ok:
        continue
    ext = ext_res.value
    h = ext.get("h")
    if h is None:
        continue
    details.append((oid, obj.get("label"), h))
    if best_h is None or h > best_h:
        best_h = h
        best_id = oid
        best_label = obj.get("label")

if best_id is None:
    submit("unknown", evidence=["no object with a valid 3D extent height was found"])

evidence = [
    f"list_objects returned {len(objs)} objects",
    f"get_3d_extent heights (m): " + ", ".join(f"{lbl}:{h:.3f}" for _, lbl, h in details),
    f"tallest = {best_label} ({best_id}) with h={best_h:.3f} m",
]

submit(best_id, target_ids=[best_id], evidence=evidence)
```

### C4 —— 有几个物体的高度超过 0.8 米？

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `4` ｜ GT `4` ｜ 证据 `supported` ｜ 工具调用序列 `list_objects → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent`
- 组合 `是` ｜ 调用 10 次（完美程序 1 次，倍数 10.0）｜ 生成 1 次 ｜ 耗时 1.795 s

```python
from itertools import chain

scene_id = "points_probe"

# 1) 列出场景中所有物体
res = list_objects(scene_id=scene_id)
if not res.ok:
    submit("unknown", evidence=["list_objects failed: %s" % res.error.code])
    raise SystemExit

objs = res.value
count = 0
tall_ids = []
details = []

for obj in objs:
    oid = obj["object_id"]
    ext = get_3d_extent(scene_id=scene_id, object_id=oid)
    if not ext.ok:
        continue
    e = ext.value
    # 高度取 h 分量
    h = e["h"] if isinstance(e, dict) else e[1]
    details.append("%s(%s)=%.3f" % (oid, obj.get("label"), h))
    if h > 0.8:
        count += 1
        tall_ids.append(oid)

submit(
    count,
    target_ids=tall_ids,
    evidence=[
        "list_objects 返回 %d 个物体" % len(objs),
        "对每个物体调用 get_3d_extent 取 h 分量，统计 h > 0.8 的个数",
        "各物体高度: " + "; ".join(details),
    ],
)
```

### C5 —— 四幅画的平均宽度是多少米？

- class **C**：答案与 GT 一致，但被契约判 unsupported —— 契约问题，与模型无关
- 答案 `0.4445499926805496` ｜ GT `0.4445` ｜ 证据 `unsupported` ｜ 工具调用序列 `find_object → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent`
- 组合 `是` ｜ 调用 5 次（完美程序 1 次，倍数 5.0）｜ 生成 1 次 ｜ 耗时 1.544 s

```python
from statistics import mean

scene_id = "points_probe"

res = find_object(scene_id=scene_id, label="picture")
if not res.ok:
    submit("unknown", evidence=["find_object(picture) failed: %s" % res.error.code])

pics = res.value
if not pics:
    submit("unknown", evidence=["no picture objects found in scene"])

widths = []
ids = []
for p in pics:
    oid = p["object_id"]
    ext = get_3d_extent(scene_id=scene_id, object_id=oid)
    if not ext.ok:
        continue
    e = ext.value
    widths.append(e["w"])
    ids.append(oid)

if not widths:
    submit("unknown", evidence=["could not read extent for any picture"])

avg = mean(widths)
submit(avg, target_ids=ids,
       evidence=["get_3d_extent(w) for %d pictures: %s" % (len(widths), widths),
                 "average width = mean(widths) = %.6f m" % avg])
```

### C6 —— 把四幅画按从左到右排序，给出 object_id 列表（逗号分隔）。

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `picture_4,picture_2,picture_3,picture_1` ｜ GT `picture_4,picture_2,picture_3,picture_1` ｜ 证据 `weak` ｜ 工具调用序列 `find_object → get_3d_position → get_3d_position → get_3d_position → get_3d_position`
- 组合 `是` ｜ 调用 5 次（完美程序 1 次，倍数 5.0）｜ 生成 1 次 ｜ 耗时 1.823 s

```python
from collections import defaultdict

scene_id = "points_probe"

# 1) 找出所有 picture
res = find_object(scene_id=scene_id, label="picture")
if not res.ok:
    submit("unknown", evidence=["find_object(picture) failed: " + str(res.error.code)])
    raise SystemExit

pics = res.value
if not pics or len(pics) < 2:
    submit("unknown", evidence=["fewer than 2 pictures found"])
    raise SystemExit

# 2) 取每幅画的相机系质心 x 坐标（左 = x 更小，相机系约定）
items = []
for p in pics:
    oid = p["object_id"]
    pos = get_3d_position(scene_id=scene_id, object_id=oid)
    if not pos.ok:
        submit("unknown", evidence=["get_3d_position failed for " + oid])
        raise SystemExit
    x = pos.value[0]
    items.append((x, oid))

# 3) 从左到右 = x 升序
items.sort(key=lambda t: t[0])
ordered = [oid for _, oid in items]

submit(
    ",".join(ordered),
    target_ids=ordered,
    evidence=[
        "find_object(label='picture') returned {} pictures".format(len(pics)),
        "get_3d_position x-coords: " + ", ".join(
            "{}={:.3f}".format(oid, x) for x, oid in items
        ),
        "sorted ascending by camera-frame x (left = smaller x)",
    ],
)
```

### C7 —— 离沙发最近的物体是哪个？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `table_1` ｜ GT `table_1` ｜ 证据 `supported` ｜ 工具调用序列 `single_object → list_objects → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance`
- 组合 `是` ｜ 调用 10 次（完美程序 9 次，倍数 1.11）｜ 生成 1 次 ｜ 耗时 2.102 s

```python
from itertools import chain

SCENE = "points_probe"

# 1) 找到沙发（唯一）
res = single_object(scene_id=SCENE, label="sofa")
if not res.ok:
    submit("unknown", evidence=["single_object(sofa) failed: %s" % res.error.code])
else:
    sofa = res.value
    sofa_id = sofa["object_id"]

    # 2) 列出场景中所有物体，排除沙发自身
    res_all = list_objects(scene_id=SCENE)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: %s" % res_all.error.code])
    else:
        objs = res_all.value
        others = [o for o in objs if o["object_id"] != sofa_id]

        if not others:
            submit("unknown", evidence=["no other objects besides sofa"])
        else:
            # 3) 用 calculate_distance 逐个算距离（几何量，不猜坐标）
            best_id = None
            best_d = None
            for o in others:
                r = calculate_distance(scene_id=SCENE, a=sofa_id, b=o["object_id"])
                if not r.ok:
                    continue
                d = r.value
                if best_d is None or d < best_d:
                    best_d = d
                    best_id = o["object_id"]

            if best_id is None:
                submit("unknown", evidence=["calculate_distance failed for all candidates"])
            else:
                submit(
                    best_id,
                    target_ids=[best_id],
                    evidence=[
                        "single_object(label='sofa') -> %s" % sofa_id,
                        "calculate_distance(sofa, each other object) minimized; min distance = %.4f m" % best_d,
                    ],
                )
```

### C8 —— 有多少个物体在桌子的前面？

- class **C**：答案与 GT 一致，但被契约判 unsupported —— 契约问题，与模型无关
- 答案 `0` ｜ GT `0` ｜ 证据 `unsupported` ｜ 工具调用序列 `single_object → list_objects → get_3d_position → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation`
- 组合 `是` ｜ 调用 11 次（完美程序 9 次，倍数 1.22）｜ 生成 2 次 ｜ 耗时 4.456 s

```python
scene_id = "points_probe"

# 1) 找到桌子
res = single_object(scene_id=scene_id, label="table")
if not res.ok:
    submit("unknown", evidence=["single_object(table) failed: %s" % res.error.code])
else:
    table = res.value
    table_id = table["object_id"]

    # 2) 列出所有物体
    res_all = list_objects(scene_id=scene_id)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: %s" % res_all.error.code])
    else:
        objs = res_all.value
        table_pos = get_3d_position(scene_id=scene_id, object_id=table_id)
        if not table_pos.ok:
            submit("unknown", evidence=["get_3d_position(table) failed"])
        else:
            tx, ty, tz = table_pos.value

            count = 0
            details = []
            for o in objs:
                oid = o["object_id"]
                if oid == table_id:
                    continue
                # 用 query_relation 判断是否在桌子前面
                rel = query_relation(scene_id=scene_id, relation="front_of",
                                     a=oid, b=table_id)
                if rel.ok:
                    val = rel.value
                    is_front = bool(val) if not isinstance(val, (int, float)) else (val > 0)
                    if is_front:
                        count += 1
                        details.append(oid)
                else:
                    # 回退：用相机系坐标判断（相机系中 z 越小越靠前）
                    p = get_3d_position(scene_id=scene_id, object_id=oid)
                    if p.ok:
                        oz = p.value[2]
                        if oz < tz:
                            count += 1
                            details.append(oid)

            submit(count,
                   target_ids=[table_id],
                   evidence=["query_relation(front_of) 对每个非桌子物体判定，"
                             "回退用相机系 z 坐标比较 (z 越小越靠前)；"
                             "桌子 z=%.3f，命中物体: %s" % (tz, details)])
```

### C8b —— 有多少个物体在沙发的前面？

- class **C**：答案与 GT 一致，但被契约判 unsupported —— 契约问题，与模型无关
- 答案 `3` ｜ GT `3` ｜ 证据 `unsupported` ｜ 工具调用序列 `single_object → list_objects → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation`
- 组合 `是` ｜ 调用 10 次（完美程序 9 次，倍数 1.11）｜ 生成 1 次 ｜ 耗时 1.523 s

```python
import math

scene_id = "points_probe"

# 1. 找到沙发
res = single_object(scene_id=scene_id, label="sofa")
if not res.ok:
    submit("unknown", evidence=["single_object(sofa) failed: %s" % res.error.code])
else:
    sofa = res.value
    sofa_id = sofa["object_id"]

    # 2. 列出所有物体
    res_all = list_objects(scene_id=scene_id)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: %s" % res_all.error.code])
    else:
        objs = res_all.value
        count = 0
        details = []
        for o in objs:
            oid = o["object_id"]
            if oid == sofa_id:
                continue
            # 用关系查询判断是否在沙发前面
            r = query_relation(scene_id=scene_id, relation="front_of", a=oid, b=sofa_id)
            if not r.ok:
                # 关系不可用则跳过
                continue
            val = r.value
            is_front = bool(val) if isinstance(val, bool) else (val > 0)
            if is_front:
                count += 1
                details.append(oid)

        submit(
            count,
            target_ids=[sofa_id],
            evidence=[
                "single_object(sofa) -> %s" % sofa_id,
                "query_relation(front_of, a=obj, b=sofa) over %d objects" % len(objs),
                "front objects: %s" % details,
            ],
        )
```

### C12 —— 场景里质心相距最远的是哪两个物体？给出两个 object_id（逗号分隔）。

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `table_1,picture_4` ｜ GT `picture_4,table_1` ｜ 证据 `weak` ｜ 工具调用序列 `list_objects → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance → calculate_distance`
- 组合 `是` ｜ 调用 37 次（完美程序 37 次，倍数 1.0）｜ 生成 1 次 ｜ 耗时 1.306 s

```python
import itertools

scene_id = "points_probe"

res = list_objects(scene_id=scene_id)
if not res.ok:
    submit("unknown", evidence=["list_objects failed: " + str(res.error.code)])
else:
    objs = res.value
    ids = [o["object_id"] for o in objs]

    best_pair = None
    best_dist = -1.0
    evidence = []

    for a, b in itertools.combinations(ids, 2):
        r = calculate_distance(scene_id=scene_id, a=a, b=b)
        if not r.ok:
            continue
        d = r.value
        if d > best_dist:
            best_dist = d
            best_pair = (a, b)

    if best_pair is None:
        submit("unknown", evidence=["no valid pairwise distances could be computed"])
    else:
        a, b = best_pair
        evidence.append(
            "calculate_distance over all %d pairs; max = %.6f m between %s and %s"
            % (len(ids) * (len(ids) - 1) // 2, best_dist, a, b)
        )
        submit("%s,%s" % (a, b), target_ids=[a, b], evidence=evidence)
```

