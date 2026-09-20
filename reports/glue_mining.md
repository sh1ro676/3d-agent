# 胶水形态挖掘（从真跑程序反推算子候选）

> 由 `scripts/mine_glue_patterns.py` 生成。**零成本**：只读已落盘的程序源码，不调模型。
> 每题程序**原文**另存 `reports/glue_programs.txt`（`--dump`）—— 结论要能被独立核对。

## 〇 读的是哪一份产物（口径）

- 文件：`combination_run_20260919_214625.json`
- 场景：`points_probe` ｜ 工具空间：12 个
- 题数：**10** ｜ 工具调用合计 **32** 次（平均 3.2 次/题）
- **循环扫描：10 个**，分布在 **9/10** 道题上
- 解析失败：**0** 题 ✓

⚠ 样本范围：**一个场景、一张图、10 道题**。所有「N 题用过」只在样本内成立，
跨场景未验证 —— 别当普适结论引用。

### 顺带修正一个被引用的数字

真实推导式（`[... for ...]` / `(... for ...)`）只出现在 **4/10** 道题上；
**9/10 道题是用显式 `for` 循环 + 累加变量**干活的。
而 `probe_combination_run.py` 里的 `has_comprehension` 用的正则是
`\bfor\b[^\n]*\b in \b` —— 它**同样匹配普通 `for x in y:` 语句**。
所以「9/10 用了推导式」这句话，实际含义是「9/10 里出现过 `for ... in`」。
⟹ 引用时必须写成「**9/10 写了显式循环扫描**」，而不是「用了推导式」。

## 一 按形态大类聚合（这张表是决策用的）

| 形态大类 | 语义 | 题数 | 用到它的题 | 建议算子 | 判定 |
|---|---|---|---|---|---|
| `scan_count_where` | 扫描一遍数满足条件的（筛选 + 计数） | **3** | C4、C8、C8b | `count_where` | **够格**（≥2 题） |
| `scan_extreme_by` | 扫描一遍按某个键留最优（argmax / argmin） | **3** | C1、C3、C7 | `extreme_by` | **够格**（≥2 题） |
| `scan_collect` | 扫描收集成列表，再在下游聚合或排序 | **2** | C5、C6 | `aggregate / rank_by` | **够格**（≥2 题） |
| `scan_extreme_pair` | 在两两组合上取极值（O(n²)） | **1** | C12 | `extreme_pair` | n=1，样本不足 |

⟹ **4 个形态大类里 3 个够格**（其余是单题偶发，不作为提升依据）。

## 二 子形态 × 可提升性（决定「能提多少」）

同一个大类里，键/谓词是不是**自由形式**，决定了能不能完整提升：

| 形态签名 | 题数 | 题 | 可提升性 | 建议算子 |
|---|---|---|---|---|
| `scan_collect｜key=n/a｜pred=tool_call` | **2** | C5、C6 | 要看下游 | `（看下游）` —— 配合下游 `aggregate` / `rank_by` 一起提 |
| `scan_collect｜key=n/a｜pred=none` | **1** | C6 | 要看下游 | `（看下游）` —— 配合下游 `aggregate` / `rank_by` 一起提 |
| `scan_count_where｜key=n/a｜pred=field_compare` | **1** | C8b | 只能提一半 | `count_where ⚠` —— `count_where(predicate)` —— 谓词是自由条件，只能提一半 |
| `scan_count_where｜key=n/a｜pred=threshold` | **1** | C4 | 只能提一半 | `count_where ⚠` —— `count_where(predicate)` —— 谓词是自由条件，只能提一半 |
| `scan_count_where｜key=n/a｜pred=tool_call` | **1** | C8 | **可完整提升** | `count_relation` —— `count_relation(relation, anchor, tol)` —— 谓词已是工具，整段循环可塌成一次调用 |
| `scan_extreme_by｜key=arithmetic｜pred=none` | **1** | C1 | 只能提一半 | `extreme_by ⚠` —— `extreme_by(field=…)` 覆盖不了 —— 键是算式（如「最大两边之积」） |
| `scan_extreme_by｜key=euclidean｜pred=guard` | **1** | C7 | **可完整提升** | `nearest` —— `nearest_to(anchor, metric='euclidean')`，或**加宽 `find_nearest`** |
| `scan_extreme_by｜key=field｜pred=guard` | **1** | C3 | **可完整提升** | `extreme_by` —— `extreme_by(how='max'|'min', field='extent_m.h')` —— **完全覆盖** |
| `scan_extreme_pair｜key=euclidean｜pred=none` | **1** | C12 | **可完整提升** | `extreme_pair` —— `extreme_pair(field='centroid_m', metric='euclidean')` —— **完全覆盖** |

**可完整提升**的签名 4 个；**只能提一半**的签名 3 个。

可完整提升的那些（键/谓词都是数据）：

- `scan_count_where｜key=n/a｜pred=tool_call`（C8）⟹ `count_relation`
  - `count_relation(relation, anchor, tol)` —— 谓词已是工具，整段循环可塌成一次调用
- `scan_extreme_by｜key=euclidean｜pred=guard`（C7）⟹ `nearest`
  - `nearest_to(anchor, metric='euclidean')`，或**加宽 `find_nearest`**
- `scan_extreme_by｜key=field｜pred=guard`（C3）⟹ `extreme_by`
  - `extreme_by(how='max'|'min', field='extent_m.h')` —— **完全覆盖**
- `scan_extreme_pair｜key=euclidean｜pred=none`（C12）⟹ `extreme_pair`
  - `extreme_pair(field='centroid_m', metric='euclidean')` —— **完全覆盖**

⟹ 「只能提一半」这一档**在真实 trace 上有实例**，不是假想：
- `scan_count_where｜key=n/a｜pred=field_compare`（C8b）
- `scan_count_where｜key=n/a｜pred=threshold`（C4）
- `scan_extreme_by｜key=arithmetic｜pred=none`（C1）

把这类结构提成算子，**只是把代码从题里挪到参数里**，自由形式并没有消失。
这就是「纯菜单不可行、混合才是终局」的实测依据。

### 最便宜的一步：加宽一个已有参数（**零新工具**）

这些题是「按距离取最近/最远」，但候选**不限类别**（C7）——
`find_nearest` / `find_farthest` **已经存在**，只是 `label` 参数实际上必填
（空串会走 `by_label("")` → `NOT_FOUND`），于是模型只能手写距离循环。
把它放宽成「`label` 省略 = 全部类别」，比新增任何算子都便宜，
而且**不改动作空间长度**（工具数不变，只改一个参数的可选性）。

### 附带发现：循环里逐个调工具（两种，**修法不同**）

**① 收集型循环里逐项调工具（2 题：C5、C6）—— 是冗余，但**文档已经写过**。**
`list_objects` 的返回值里**已经有那些字段**（`_brief` 带 `centroid_m` +
`extent_m{w,h,l}`），逐项再调一次属于重复取数。

⚠ 但它**不是**「文档没写」这一类。`list_objects` 的**第一段**（即渲染进提示词的那段）原文写着：

> **所有物体的质心与尺寸一次给全**，找极值/计数/求均值不必再逐个调工具。

而 `tools/spatial.py` 的最后一次修改（09-19 21:42）**早于**这次真跑（`..._214625`，21:46），
也就是说该次跑完之后这个文件再没被改过 ⟹ **这句话模型当时看得到，它没有照做**。

⟹ 与「返回值形状没写进第一段」（§教训 2）**不是同一类**：那次是**看不到**，
这次是**看到不照做**。所以修法是提示词纪律，**不是**再改一遍文档。

**② 计数型循环里调工具当谓词（1 题：C8）—— 该塌成一次调用。**
谓词本身已经是工具调用，所以整段「逐个调 + 计数」可以塌成**一次**
`count_relation(...)`。C8 的 **10 次**调用就是这么来的。
与 ① 的区别：① 省的是重复取数，② 省的是**调用次数本身**。

## 三 每题到底写了什么（配方，证据用）

| 题号 | 扫描 | 表达式级组合 | 工具调用 | 一次调用够不够 |
|---|---|---|---|---|
| C1 | `scan_extreme_by` | — | 1 | 够（省冗余） |
| C2 | — | `expr_len_local` | 3 | 够（省冗余） |
| C3 | `scan_extreme_by` | — | 1 | 够（省冗余） |
| C4 | `scan_count_where` | — | 1 | 够（省冗余） |
| C5 | `scan_collect` | `expr_aggregate_mean` | 5 | 够（省冗余） |
| C6 | `scan_collect` → `scan_collect` | `expr_rank` → `expr_project` → `expr_rank` | 5 | 够（省冗余） |
| C7 | `scan_extreme_by` | — | 3 | 够（省冗余） |
| C8 | `scan_count_where` | — | 10 | **不够（真需要组合）** |
| C8b | `scan_count_where` | — | 2 | **不够（真需要组合）** |
| C12 | `scan_extreme_pair` | — | 1 | 够（省冗余） |

**8/10 题「一次工具调用就够」**（`one_call_matches_gt`）。
⟹ 这些题上，胶水是**可避的冗余**而不是能力缺口：提升算子省的是调用次数与
出错面，**不增加可回答问题的集合**。要证明「加工具真的扩大了能力」，
得看剩下的那 2 题。

## 四 表达式级组合明细（非循环、非 submit）

| 类型 | 语义 | 题数 | 题 | 举例（骨架） |
|---|---|---|---|---|
| `expr_aggregate_mean` | 对收集到的列表求均值 | 1 | C5 | `mean(v)` |
| `expr_len_local` | len(本地变量) —— Python 层面的记账，不是场景级计数 | 1 | C2 | `len(v)` |
| `expr_project` | 推导式无 if —— 只是投影 | 1 | C6 | `[v for v, v in v]` |
| `expr_rank` | 对收集到的列表排序 | 1 | C6 | `v.sort(key=lambda v: v[0])` |

⚠ `expr_len_local` 刻意**不进决策表**：`len(本地列表)` 是 Python 层面的记账，
不是「场景里有几个东西」。第一版把它当成最大的桶，是那版报告最大的错。

另有 **4 条**赋值给记账变量（`evidence` / `detail` / `order` 之类）的表达式
**被排除在外、不上表** —— 它们是拼呈堂证供，不是把场景归约成答案。
排除名单只认**变量名**（`BOOKKEEPING_TARGETS`），换个名字就漏；
所以这里如实报出条数，让「排除掉了什么」能被核对。

## 五 没挖到的

所有题的程序都解析成功，没有静默漏掉的。

## 六 形态明细（每条带循环原文）

### `scan_collect｜key=n/a｜pred=tool_call` ｜ 2 题：C5、C6 ｜ 6 行

```python
for p in pics:
    oid = p["object_id"]
    ext = get_3d_extent(scene_id="points_probe", object_id=oid)
    if not ext.ok:
        submit("unknown", evidence=[f"get_3d_extent failed for {oid}: {ext.error.code}"])
    widths.append(ext.value["w"])
```

### `scan_collect｜key=n/a｜pred=none` ｜ 1 题：C6 ｜ 4 行

```python
for p in pics:
    oid = p["object_id"]
    cx = p["centroid_m"][0]
    items.append((cx, oid))
```

### `scan_count_where｜key=n/a｜pred=field_compare` ｜ 1 题：C8b ｜ 7 行

```python
for o in objs:
            if o["object_id"] == sofa_id:
                continue
            z = o["centroid_m"][2]
            if z < sofa_c[2]:
                count += 1
                counted.append(o["object_id"])
```

### `scan_count_where｜key=n/a｜pred=threshold` ｜ 1 题：C4 ｜ 8 行

```python
for o in objs:
    ext = o.get("extent_m") or {}
    h = ext.get("h")
    if h is None:
        continue
    if h > 0.8:
        count += 1
        tall.append((o.get("object_id"), o.get("label"), h))
```

### `scan_count_where｜key=n/a｜pred=tool_call` ｜ 1 题：C8 ｜ 8 行

```python
for o in objs:
            oid = o["object_id"]
            if oid == table_id:
                continue
            r = query_relation(scene_id=scene_id, relation="front_of", a=oid, b=table_id, tol=0)
            if r.ok and r.value is True:
                count += 1
                counted.append(oid)
```

### `scan_extreme_by｜key=arithmetic｜pred=none` ｜ 1 题：C1 ｜ 9 行

```python
for p in pics:
    oid = p["object_id"]
    ext = p["extent_m"]
    dims = sorted([ext["w"], ext["h"], ext["l"]], reverse=True)
    area = dims[0] * dims[1]  # 画面面积 = 最大的两个边长之积
    detail.append("%s: dims=%s area=%.4f" % (oid, dims, area))
    if area > best_area:
        best_area = area
        best_id = oid
```

### `scan_extreme_by｜key=euclidean｜pred=guard` ｜ 1 题：C7 ｜ 8 行

```python
for o in objs:
            if o["object_id"] == sofa_id:
                continue
            ox, oy, oz = o["centroid_m"]
            d = ((ox - sx) ** 2 + (oy - sy) ** 2 + (oz - sz) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_d = d
                best = o
```

### `scan_extreme_by｜key=field｜pred=guard` ｜ 1 题：C3 ｜ 8 行

```python
for o in objs:
    ext = o.get("extent_m") or {}
    h = ext.get("h")
    if h is None:
        continue
    if best_h is None or h > best_h:
        best_h = h
        best = o
```

### `scan_extreme_pair｜key=euclidean｜pred=none` ｜ 1 题：C12 ｜ 7 行

```python
for a, b in combinations(objs, 2):
        ca = a["centroid_m"]
        cb = b["centroid_m"]
        d = sum((ca[i] - cb[i]) ** 2 for i in range(3)) ** 0.5
        if d > best_dist:
            best_dist = d
            best_pair = (a["object_id"], b["object_id"])
```

## 七 本工具看不到什么（用了它的结论就必须一起说）

1. **只看到模型写了什么，看不到它需要什么。** 没被写出来的需求不在样本里 ——
   它**不能**回答「还缺哪些算子」，只能回答「哪些形态重复出现过」。
2. **n=1 不是「不需要」，是「没证据」。** 单题偶发的形态，缺的是样本不是价值。
3. **归一化会高估**（不同变量名被抹平）**也会低估**（等价写法并不起来）。
   两个方向都有，别把「9 个签名」当精确值。
4. **它不判断对错。** 程序写错但形态漂亮的照样进表 —— 正确性在
   `probe_combination_run.py` 的 `class` 字段里，不在这里。
5. **它不提议现在就加算子。** 判据只到「够不够格进入讨论」为止；
   真加之前还要过「加工具不静默改变已跑臂」那一关（`TOOLS_VERSION` + 消融臂）。
6. **形态归类是启发式**：`_best_update_if` 的判据是「条件读的变量正是体内写的」——
   把「留最优」与「计数」分开，但它对更绕的写法会失手。失手的样子是 `scan_other`，
   不会静默算进别的桶。
