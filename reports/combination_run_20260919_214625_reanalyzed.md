# 组合题真跑诊断 —— points_probe

真值来自 `scripts/probe_combination.py` 的同一份 SPECS（工具算出来的，不是脚本算的）。

## 归因汇总

- 题数 10 ｜ 与 GT 一致 **10** ｜ 成本 ¥0.0879
- 结局分布：`{"OK": 5, "OK*": 5}`
- 状态分布：`{"ok": 10}`
- 用了组合（**含显式 `for` 循环** / 带 key 算子 / `key=`）：**9/10**
  - 其中出现过 `for … in`：9 题 ｜ **真推导式**：4 题 ｜ 出现 `key=`：1 题
  - ⚠ 差值是**有信息量**的那个数：它 = 「用显式循环写、没压成一句表达式」的题数。
    `has_iteration` 用的正是 `\bfor\b…in\b`，它**同时匹配普通 `for x in y:` 语句**；
    早先这个字段叫 `has_comprehension`，于是「9/10 里出现过 for」被读成了
    「9/10 用了推导式」—— 那句话是错的（2026-09-20 修正，见 `reports/glue_mining.md`）。
- ★ **答对却被证据契约判 `unsupported`：0 题**（与准确率无关）
- ★ 其中 **9 题的全部所需几何量，一次 `list_objects` 就够**（上限：18 次调用）
- 工具调用：实际 **32** 次 ｜ GT 路线 70 次 ｜ 最少 18 次 ｜ 冗余倍数中位数 **3.0**

⚠ 口径：「中位数」= **排序后取第 n/2 位**（见 `_summarize`）—— 偶数题数时取
偏大那一个，不是两数均值。同一份定义横跨各次运行所以可比，别与教科书中位数混用。
⚠ 分母（「最少几次」）取决于**当前**的 GT 与「一次 `list_objects` 够不够」的判定；
GT 口径一改，这一列就会跟着变 —— **模型没变而这一列变了**是可能的，
引用前先看 `reanalyzed_from` 与 `reverified` 两个字段。

`class` 的含义：`OK/OK*` = 答对（证据 supported / weak）；`C` = 答对但证据被拒收；
`A` = 答错且没用组合；`B` = 答错但用了组合。**A 与 B 只描述失败**，
答对的题不归到 A/B —— 「工具已经覆盖这个组合」不该被统计成「模型不会组合」。

`冗余倍数` = 实际调用次数 / 最少调用次数。>1 说明**模型没利用返回值里已有的字段**，
而不是「工具不够用」—— 这一列把「功能单薄」与「没看清返回值」分开了。

`组合` 一列 = 出现过 `for … in`（**含手写循环**）/ 带 key 算子 / `key=` 之一。
它是**宽判据**：手写 `for` + 累加变量也算组合。窄的那个数是首页的「真推导式」。

## 逐题

| 题号 | class | 状态 | 答案 | GT | 证据 | 组合 | 调用(实际/最少) | 倍数 |
|---|---|---|---|---|---|---|---|---|
| C1 | **OK** | ok | `picture_1` | `picture_1` | supported | 是 | 1/1 | 1.0 |
| C2 | **OK** | ok | `picture_2` | `picture_2` | supported | **否** | 3/1 | 3.0 |
| C3 | **OK** | ok | `mirror_1` | `mirror_1` | supported | 是 | 1/1 | 1.0 |
| C4 | **OK*** | ok | `4` | `4` | weak | 是 | 1/1 | 1.0 |
| C5 | **OK*** | ok | `0.4445499926805496` | `0.4445` | weak | 是 | 5/1 | 5.0 |
| C6 | **OK*** | ok | `picture_4,picture_2,picture_3,picture_1` | `picture_4,picture_2,picture_3,picture_1` | weak | 是 | 5/1 | 5.0 |
| C7 | **OK** | ok | `table_1` | `table_1` | supported | 是 | 3/1 | 3.0 |
| C8 | **OK** | ok | `1` | `1` | supported | 是 | 10/1 | 10.0 |
| C8b | **OK*** | ok | `3` | `3` | weak | 是 | 2/9 | 0.22 |
| C12 | **OK*** | ok | `table_1,picture_4` | `picture_4,table_1` | weak | 是 | 1/1 | 1.0 |

### C1 —— 四幅画里哪一幅面积最大？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `picture_1` ｜ GT `picture_1` ｜ 证据 `supported` ｜ 工具调用序列 `find_object`
- 组合 `是` ｜ 调用 1 次（最少 1 次，GT 路线 1 次，倍数 1.0）｜ 生成 1 次 ｜ 耗时 1.729 s

```python
from collections import defaultdict

# 1) 找出所有 picture
res = find_object(scene_id="points_probe", label="picture")
if not res.ok:
    submit("unknown", evidence=["find_object(picture) failed: %s" % res.error.code])

pics = res.value
if not pics:
    submit("unknown", evidence=["no picture objects found in scene"])

# 2) 用 extent_m 计算每幅画的面积（取最大的两个维度乘积作为画面面积）
best_id = None
best_area = -1.0
detail = []
for p in pics:
    oid = p["object_id"]
    ext = p["extent_m"]
    dims = sorted([ext["w"], ext["h"], ext["l"]], reverse=True)
    area = dims[0] * dims[1]  # 画面面积 = 最大的两个边长之积
    detail.append("%s: dims=%s area=%.4f" % (oid, dims, area))
    if area > best_area:
        best_area = area
        best_id = oid

submit(
    best_id,
    target_ids=[best_id],
    evidence=[
        "find_object(label='picture') returned %d pictures" % len(pics),
        "area computed as product of two largest extent_m dims: " + "; ".join(detail),
        "max area = %.4f m^2 -> %s" % (best_area, best_id),
    ],
)
```

### C2 —— 哪幅画离镜子最近？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `picture_2` ｜ GT `picture_2` ｜ 证据 `supported` ｜ 工具调用序列 `single_object → find_object → find_nearest`
- 组合 `**否**` ｜ 调用 3 次（最少 1 次，GT 路线 1 次，倍数 3.0）｜ 生成 1 次 ｜ 耗时 1.668 s

```python
from itertools import count

scene_id = "points_probe"

# 1) 找到镜子
mirror_res = single_object(scene_id=scene_id, label="mirror")
if not mirror_res.ok:
    submit("unknown", evidence=["single_object(mirror) failed: " + str(mirror_res.error.code)])
else:
    mirror_id = mirror_res.value["object_id"]

    # 2) 找到所有画
    pics_res = find_object(scene_id=scene_id, label="picture")
    if not pics_res.ok:
        submit("unknown", evidence=["find_object(picture) failed: " + str(pics_res.error.code)])
    else:
        pics = pics_res.value
        if not pics:
            submit("unknown", evidence=["no picture objects in scene"])
        else:
            # 3) 用 find_nearest 直接按距离排序（anchor=镜子，label=picture）
            near_res = find_nearest(scene_id=scene_id, anchor=mirror_id, label="picture", k=len(pics))
            if not near_res.ok:
                submit("unknown", evidence=["find_nearest failed: " + str(near_res.error.code)])
            else:
                nearest = near_res.value[0]
                submit(
                    nearest["object_id"],
                    target_ids=[nearest["object_id"]],
                    evidence=[
                        "find_nearest(anchor=%s, label=picture) -> %s at distance_m=%s"
                        % (mirror_id, nearest["object_id"], nearest.get("distance_m"))
                    ],
                )
```

### C3 —— 场景里最高的物体是哪个？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `mirror_1` ｜ GT `mirror_1` ｜ 证据 `supported` ｜ 工具调用序列 `list_objects`
- 组合 `是` ｜ 调用 1 次（最少 1 次，GT 路线 1 次，倍数 1.0）｜ 生成 1 次 ｜ 耗时 1.372 s

```python
from collections import defaultdict

res = list_objects(scene_id="points_probe")
if not res.ok:
    submit("unknown", evidence=["list_objects failed: %s" % res.error.code])

objs = res.value
best = None
best_h = None
for o in objs:
    ext = o.get("extent_m") or {}
    h = ext.get("h")
    if h is None:
        continue
    if best_h is None or h > best_h:
        best_h = h
        best = o

if best is None:
    submit("unknown", evidence=["no object with extent_m.h available"])

submit(
    best["object_id"],
    target_ids=[best["object_id"]],
    evidence=[
        "list_objects(points_probe) returned %d objects with extent_m" % len(objs),
        "max extent_m.h = %.4f m on object_id=%s (label=%s)"
        % (best_h, best["object_id"], best.get("label")),
    ],
)
```

### C4 —— 有几个物体的高度超过 0.8 米？

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `4` ｜ GT `4` ｜ 证据 `weak` ｜ 工具调用序列 `list_objects`
- 组合 `是` ｜ 调用 1 次（最少 1 次，GT 路线 1 次，倍数 1.0）｜ 生成 1 次 ｜ 耗时 1.493 s

```python
from collections import defaultdict

res = list_objects(scene_id="points_probe")
if not res.ok:
    submit("unknown", evidence=["list_objects failed: " + str(res.error.code)])

objs = res.value
count = 0
tall = []
for o in objs:
    ext = o.get("extent_m") or {}
    h = ext.get("h")
    if h is None:
        continue
    if h > 0.8:
        count += 1
        tall.append((o.get("object_id"), o.get("label"), h))

submit(
    count,
    target_ids=[t[0] for t in tall],
    evidence=[
        "list_objects(points_probe) 返回 %d 个物体，逐个读取 extent_m['h']" % len(objs),
        "高度 > 0.8 m 的物体: " + ", ".join("%s(%s)=%.3f" % t for t in tall) if tall else "无物体高度超过 0.8 m",
    ],
)
```

### C5 —— 四幅画的平均宽度是多少米？

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `0.4445499926805496` ｜ GT `0.4445` ｜ 证据 `weak` ｜ 工具调用序列 `list_objects → get_3d_extent → get_3d_extent → get_3d_extent → get_3d_extent`
- 组合 `是` ｜ 调用 5 次（最少 1 次，GT 路线 1 次，倍数 5.0）｜ 生成 1 次 ｜ 耗时 1.512 s

```python
from statistics import mean

res = list_objects(scene_id="points_probe", label="picture")
if not res.ok:
    submit("unknown", evidence=[f"list_objects failed: {res.error.code}"])

pics = res.value
if not pics:
    submit("unknown", evidence=["no picture objects found in scene"])

widths = []
for p in pics:
    oid = p["object_id"]
    ext = get_3d_extent(scene_id="points_probe", object_id=oid)
    if not ext.ok:
        submit("unknown", evidence=[f"get_3d_extent failed for {oid}: {ext.error.code}"])
    widths.append(ext.value["w"])

avg = mean(widths)
submit(avg, target_ids=[p["object_id"] for p in pics],
       evidence=[f"widths from get_3d_extent (w) for {len(widths)} pictures: {widths}",
                 f"mean = {avg}"])
```

### C6 —— 把四幅画按从左到右排序，给出 object_id 列表（逗号分隔）。

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `picture_4,picture_2,picture_3,picture_1` ｜ GT `picture_4,picture_2,picture_3,picture_1` ｜ 证据 `weak` ｜ 工具调用序列 `find_object → get_object → get_object → get_object → get_object`
- 组合 `是` ｜ 调用 5 次（最少 1 次，GT 路线 1 次，倍数 5.0）｜ 生成 1 次 ｜ 耗时 2.342 s

```python
from collections import defaultdict

# 1) 取出所有 picture 物体
res = find_object(scene_id="points_probe", label="picture")
if not res.ok:
    submit("unknown", evidence=["find_object(picture) failed: %s" % res.error.code])

pics = res.value
if not pics:
    submit("unknown", evidence=["no picture objects found in scene"])

# 2) 用相机系 x 坐标排序（相机系：x 向右为正 → 图像上从左到右）
#    同时用 bbox_2d 的 x 中心做交叉验证
items = []
for p in pics:
    oid = p["object_id"]
    cx = p["centroid_m"][0]
    items.append((cx, oid))

items.sort(key=lambda t: t[0])
ordered = [oid for _, oid in items]

# 3) 交叉验证：读 bbox_2d 的图像像素 x 中心
bbox_centers = {}
for oid in ordered:
    r = get_object(scene_id="points_probe", object_id=oid)
    if r.ok:
        bb = r.value.get("bbox_2d")
        if bb:
            bbox_centers[oid] = (bb[0] + bb[2]) / 2.0

bbox_order = sorted(bbox_centers.keys(), key=lambda o: bbox_centers[o]) if bbox_centers else []

evidence = [
    "find_object(label='picture') returned %d pictures" % len(pics),
    "sorted by centroid_m[0] (camera x, right-positive): " + ", ".join(
        "%s=%.3f" % (oid, cx) for cx, oid in items),
]
if bbox_order:
    evidence.append("bbox_2d x-center order: " + ", ".join(
        "%s=%.1f" % (o, bbox_centers[o]) for o in bbox_order))

submit(",".join(ordered), target_ids=ordered, evidence=evidence)
```

### C7 —— 离沙发最近的物体是哪个？给出它的 object_id。

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `table_1` ｜ GT `table_1` ｜ 证据 `supported` ｜ 工具调用序列 `single_object → list_objects → calculate_distance`
- 组合 `是` ｜ 调用 3 次（最少 1 次，GT 路线 9 次，倍数 3.0）｜ 生成 1 次 ｜ 耗时 2.138 s

```python
from itertools import chain

scene_id = "points_probe"

# 1) 找到沙发（唯一）
res = single_object(scene_id=scene_id, label="sofa")
if not res.ok:
    submit("unknown", evidence=["single_object(sofa) failed: " + str(res.error.code)])
else:
    sofa = res.value
    sofa_id = sofa["object_id"]

    # 2) 列出全部物体，用质心算距离（list_objects 一次给全 centroid_m）
    res_all = list_objects(scene_id=scene_id)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: " + str(res_all.error.code)])
    else:
        objs = res_all.value
        sx, sy, sz = sofa["centroid_m"]

        best = None
        best_d = None
        for o in objs:
            if o["object_id"] == sofa_id:
                continue
            ox, oy, oz = o["centroid_m"]
            d = ((ox - sx) ** 2 + (oy - sy) ** 2 + (oz - sz) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_d = d
                best = o

        if best is None:
            submit("unknown", evidence=["no other objects in scene"])
        else:
            # 用工具复核距离
            res_d = calculate_distance(scene_id=scene_id, a=sofa_id, b=best["object_id"])
            dist = res_d.value if res_d.ok else best_d
            submit(
                best["object_id"],
                target_ids=[best["object_id"]],
                evidence=[
                    "sofa_id=%s centroid=%s" % (sofa_id, sofa["centroid_m"]),
                    "nearest=%s label=%s centroid=%s" % (best["object_id"], best["label"], best["centroid_m"]),
                    "distance_m=%.4f (calculate_distance / euclidean on centroid_m)" % dist,
                ],
            )
```

### C8 —— 有多少个物体在桌子的前面？

- class **OK**：答案与 GT 一致，证据被支持
- 答案 `1` ｜ GT `1` ｜ 证据 `supported` ｜ 工具调用序列 `single_object → list_objects → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation → query_relation`
- 组合 `是` ｜ 调用 10 次（最少 1 次，GT 路线 9 次，倍数 10.0）｜ 生成 1 次 ｜ 耗时 1.979 s

```python
from collections import Counter

scene_id = "points_probe"

# 1) 找到桌子
res = single_object(scene_id=scene_id, label="table")
if not res.ok:
    submit("unknown", evidence=["single_object(table) failed: %s" % res.error.code])
else:
    table = res.value
    table_id = table["object_id"]
    table_centroid = table["centroid_m"]

    # 2) 列出所有物体
    res_all = list_objects(scene_id=scene_id)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: %s" % res_all.error.code])
    else:
        objs = res_all.value

        # 相机系：x 右, y 下, z 前(深度)。"桌子前面" = 比桌子更靠近相机 = z 更小。
        # 用 query_relation 的 front_of 关系来判定更稳妥。
        count = 0
        counted = []
        for o in objs:
            oid = o["object_id"]
            if oid == table_id:
                continue
            r = query_relation(scene_id=scene_id, relation="front_of", a=oid, b=table_id, tol=0)
            if r.ok and r.value is True:
                count += 1
                counted.append(oid)

        submit(
            count,
            target_ids=counted,
            evidence=[
                "table_id=%s centroid_m=%s" % (table_id, table_centroid),
                "counted objects with query_relation(relation='front_of', a=obj, b=table, tol=0) == True",
                "count=%d, ids=%s" % (count, counted),
            ],
        )
```

### C8b —— 有多少个物体在沙发的前面？

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `3` ｜ GT `3` ｜ 证据 `weak` ｜ 工具调用序列 `single_object → list_objects`
- 组合 `是` ｜ 调用 2 次（最少 9 次，GT 路线 9 次，倍数 0.22）｜ 生成 1 次 ｜ 耗时 1.986 s

```python
from collections import defaultdict

scene_id = "points_probe"

# 1) 找到沙发
res = single_object(scene_id=scene_id, label="sofa")
if not res.ok:
    submit("unknown", evidence=["single_object(sofa) failed: " + str(res.error.code)])
else:
    sofa = res.value
    sofa_id = sofa["object_id"]
    sofa_c = sofa["centroid_m"]

    # 2) 列出全部物体
    res_all = list_objects(scene_id=scene_id)
    if not res_all.ok:
        submit("unknown", evidence=["list_objects failed: " + str(res_all.error.code)])
    else:
        objs = res_all.value

        # 相机系：x 向右，y 向下，z 向前（深度）。
        # "在沙发前面" = 比沙发更靠近相机 = z 更小。
        count = 0
        counted = []
        for o in objs:
            if o["object_id"] == sofa_id:
                continue
            z = o["centroid_m"][2]
            if z < sofa_c[2]:
                count += 1
                counted.append(o["object_id"])

        submit(
            count,
            target_ids=[sofa_id] + counted,
            evidence=[
                "sofa centroid_m=%s (z=%.4f)" % (sofa_c, sofa_c[2]),
                "counted objects with centroid z < sofa z: %s" % counted,
                "rule: camera-frame z is depth; smaller z = closer to camera = in front of sofa",
            ],
        )
```

### C12 —— 场景里质心相距最远的是哪两个物体？给出两个 object_id（逗号分隔）。

- class **OK***：答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）
- 答案 `table_1,picture_4` ｜ GT `picture_4,table_1` ｜ 证据 `weak` ｜ 工具调用序列 `list_objects`
- 组合 `是` ｜ 调用 1 次（最少 1 次，GT 路线 37 次，倍数 1.0）｜ 生成 1 次 ｜ 耗时 1.507 s

```python
from itertools import combinations

res = list_objects(scene_id="points_probe")
if not res.ok:
    submit("unknown", evidence=["list_objects failed: " + str(res.error.code)])
else:
    objs = res.value
    best_pair = None
    best_dist = -1.0
    for a, b in combinations(objs, 2):
        ca = a["centroid_m"]
        cb = b["centroid_m"]
        d = sum((ca[i] - cb[i]) ** 2 for i in range(3)) ** 0.5
        if d > best_dist:
            best_dist = d
            best_pair = (a["object_id"], b["object_id"])
    if best_pair is None:
        submit("unknown", evidence=["fewer than 2 objects in scene"])
    else:
        submit(
            best_pair[0] + "," + best_pair[1],
            target_ids=[best_pair[0], best_pair[1]],
            evidence=[
                "list_objects returned centroids for %d objects" % len(objs),
                "max pairwise euclidean distance = %.4f m between %s and %s"
                % (best_dist, best_pair[0], best_pair[1]),
            ],
        )
```

