"""从真跑产物里挖「胶水形态」——把「该提哪几个算子进菜单」交给数据说。

## 为什么要走观测路线

候选算子名单一旦由人写，写进去的就是「我以为会被问到什么」。而这份先验会
**变成实验条件的一部分**（工具文档逐字节进 prompt —— 项目已为此立过
`TOOLS_VERSION`）。所以这里不设计算子，只做一件事：读模型**实际写出来的**程序，
把它在工具调用之间写下的组合结构归并、计数，再用一条**事先写死的判据**决定
哪些够格被提升。

## 挖掘单位 = 扫描（scan），不是表达式

第一版把单位定成「推导式 / `max(` / `sorted(` 这类表达式」，结果**全错**：
实测这 10 道题里 **8 道用的是显式 `for` + 累加变量**（`best_id` / `best_h` /
`count += 1`），而不是推导式。第一版因此把 `evidence=[...]` 里记账用的
`len(...)` 报成了最大的桶，而真正干活的循环一个都没看见（C8 显示「0 个站点」）。

所以单位改成**循环累积扫描**：一个 `for` 循环 + 一个跨迭代累积的变量 = 一次
「把多个物体归约成一个答案」的动作。这正是该被提成算子的东西。

    · `if <比较旧最优>` 体内给「最优变量」赋值  ⟹ `scan_extreme_by`（argmax/argmin）
    · `count += 1`（带 `if`）                    ⟹ `scan_count_where`
    · `list.append(...)` / `d[k] = ...`          ⟹ `scan_collect`（收集，等下游聚合）
    · 迭代对象是 `combinations(...)`             ⟹ `scan_extreme_pair`

## 两个正交的「自由形式」轴 —— 这才是可提升性的判据

一个扫描能不能被**完整**提成算子，取决于它的**键/谓词是数据还是代码**：

| 轴 | 取值 | 含义 |
|---|---|---|
| `key_kind` | `field` | 比较的是现成字段投影 ⟹ **可完整提升** |
| | `euclidean` | 欧氏距离式（`**2` 求和再 `**0.5`）⟹ 提成 `metric=` 参数即可 |
| | `arithmetic` | 任意算式（如「最大两边之积」）⟹ **只能提一半** |
| `predicate` | `tool_call` | 谓词本身已是工具调用 ⟹ **可完整提升** |
| | `threshold` | 与常数比（`h > 0.8`）⟹ **只能提一半** |
| | `field_compare` | 两个字段互比 ⟹ 多半是某个关系的重新实现，**只能提一半** |

## 判据（写死在这里，不临时改）

    · 同一**形态大类**被 **≥2 道不同的题** 用到 ⟹ 够格进入提升讨论
    · 只被 1 道题用到 ⟹ **样本不足**，不作为提升依据
    · 键/谓词含自由形式 ⟹ 只能提「一半」（参数里必须保留表达式）

## 排除掉的，以及为什么

`submit(..., evidence=[...])` 里的表达式**不算胶水**：那是记账（把中间量写进
呈堂证供），不是「把场景归约成答案」。第一版没排掉它，于是 `("%s=%.3f" % (oid,cx)
for ...)` 这类**拼字符串**被当成了组合算子。同理 `len(本地列表)` 不是场景级计数，
单列一桶且**不进决策表**。

## 已知误差（两个方向都会偏）

    · **会高估**：不同变量名被抹平，结构相同但语义不同的两段代码会被并成一个形态
    · **会低估**：同一件事的两种等价写法（`max(key=)` vs 手写循环）并不起来

所以本工具的输出是**线索**，不是「重复次数」的精确统计。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
EXECUTOR_SRC = PROJECT_ROOT / "agents" / "executor.py"

#: **不是**工具名的名字，在骨架里保留原名。Python 内建 + `statistics` 的入口 ——
#: 后者也要保留，否则 `statistics.mean(widths)` 会被抹成 `f(v)`，例子读不出来。
BUILTIN_KEEP = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
    "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "zip",
    "mean", "median", "stdev", "sqrt", "hypot",
})

#: **形态大类** → (建议算子, 说明)。判定用的是大类，不是子形态。
SCAN_KIND_INFO: dict[str, tuple[str, str]] = {
    "scan_extreme_by": ("extreme_by", "扫描一遍按某个键留最优（argmax / argmin）"),
    "scan_count_where": ("count_where", "扫描一遍数满足条件的（筛选 + 计数）"),
    "scan_extreme_pair": ("extreme_pair", "在两两组合上取极值（O(n²)）"),
    "scan_collect": ("aggregate / rank_by", "扫描收集成列表，再在下游聚合或排序"),
    "scan_count_all": ("已被覆盖", "无条件计数 —— `list_objects` 已给 `returned`"),
    "scan_other": ("？", "未归类 —— 人工判读"),
}

#: 表达式级（非循环）的组合 —— 用来补齐 `scan_collect` 的下游那一步。
EXPR_KIND_INFO: dict[str, tuple[str, str]] = {
    "expr_aggregate_mean": ("aggregate", "对收集到的列表求均值"),
    "expr_aggregate_sum": ("aggregate", "对收集到的列表求和"),
    "expr_rank": ("rank_by", "对收集到的列表排序"),
    "expr_extreme": ("extreme_by", "对内建序列取极值（未走循环）"),
    "expr_filter": ("count_where ⚠", "推导式带 if —— 谓词是自由表达式"),
    "expr_project": ("不需要", "推导式无 if —— 只是投影"),
    "expr_count_generated": ("count_over", "对生成式计数"),
    "expr_len_local": ("不计入决策表", "len(本地变量) —— Python 层面的记账，不是场景级计数"),
    "expr_other": ("？", "未归类 —— 人工判读"),
}

#: `key_kind` / `predicate` 里哪些是**自由形式**（也就是「只能提一半」的证据面）。
FREE_KEY_KINDS = frozenset({"arithmetic"})
FREE_PREDICATES = frozenset({"threshold", "field_compare"})

_LAMBDA_DUP_RE = re.compile(r"lambda v, v\b")
_FIELD_RE = re.compile(r'\["([^"]+)"\]')


# ============================================================================
# 0. 读工具名单（AST 读源码，不 import）
# ============================================================================


def read_tool_names(src_path: Path = EXECUTOR_SRC) -> frozenset[str]:
    """从 `agents/executor.py` 里读出 `QA_TOOLSET` 的字符串常量。

    读不到就 `SystemExit` —— 退回硬编码会让「调了工具」与「调了自定义函数」
    静默混在一起，形态归类随之出错。
    """
    if not src_path.exists():
        raise SystemExit("[mine-glue] 找不到 %s，无法确认工具名单" % src_path)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "QA_TOOLSET" and node.value is not None:
                if isinstance(node.value, (ast.Tuple, ast.List)):
                    names = tuple(
                        e.value for e in node.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
                    if not names:
                        raise SystemExit("[mine-glue] QA_TOOLSET 是空的 —— 解析方式失效了")
                    return frozenset(names)
    raise SystemExit("[mine-glue] 在 %s 里没找到 QA_TOOLSET" % src_path)


# ============================================================================
# 1. 通用小工具
# ============================================================================


def _name(id_: str) -> ast.Name:
    return ast.Name(id=id_, ctx=ast.Load())


def tool_calls_of(node: ast.AST, tool_names: frozenset[str]) -> set[str]:
    """**只算工具调用**：`Name` 形式的函数名，且落在 `QA_TOOLSET` 里。

    刻意不收 `Attribute` 形式的调用（`r.ok`、`x.append`、`d.get`）——
    它们是 Python 层面的方法，混进来会让「谓词是不是工具调用」判断失真。
    """
    return {n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in tool_names}


def _called_names(node: ast.AST) -> set[str]:
    """所有被调用的函数名（含 `Attribute` 形式的方法名），用于分桶。"""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
    return out


#: **记账变量名** —— 赋值给这些名字的表达式不算胶水。
#:
#: 为什么需要这一手：模型常把 evidence **先拼进一个变量**、再交给 `submit`，
#: 例如 `evidence = ["...", ", ".join(f"{o}" for o in order)]`。只看「在不在
#: `submit(...)` 里面」会漏掉它，于是拼字符串的生成式被算成了组合算子 ——
#: 第一版报告里 C6 就凭空多出两个 `expr_project`。
#:
#: ⚠ 这是**按名字**判的启发式，和 `verifier` 的 `_CENSUS_KEYS` 同一手法：
#: 它只能挡住「名字在名单里」的那些。模型换个名字（`acc`、`buf`）就漏 ——
#: 所以报告里要**如实报出被挡掉了几条**，别让排除本身变成不可见的一步。
BOOKKEEPING_TARGETS = frozenset({
    "evidence", "ev", "evs", "notes", "note", "detail", "details",
    "reason", "reasons", "explanation", "explain", "why", "log", "logs",
    "debug", "msg", "msgs", "buf", "acc", "lines", "text", "texts", "summary",
})


def _excluded_ids(tree: ast.AST) -> tuple[set[int], int]:
    """不该被当成「胶水」的节点 id，以及**被挡掉的条数**。

    三处都排掉，理由各不相同：

      · `submit(...)` 内部 —— 记账（把中间量写进呈堂证供），不是归约成答案；
      · 赋值给 `BOOKKEEPING_TARGETS` 的表达式 —— 同上，只是模型分两步写；
      · **循环内部** —— 那是扫描自己的实现细节，单独摘出来一个扫描会报成
        三个表达式，决策表就被自己的实现噪声填满。

    返回条数是刻意的：排除名单一旦生效，被挡掉的东西就**看不见了**，
    于是「这一步排除了什么」变成一句转述，没法核对。
    """
    out: set[int] = set()
    n_book = 0
    for n in ast.walk(tree):
        inside_submit = (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == "submit")
        if inside_submit or isinstance(n, (ast.For, ast.While)):
            out.update(id(x) for x in ast.walk(n))
            continue
        targets: list[ast.expr] = []
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = list(n.targets) if isinstance(n, ast.Assign) else [n.target]
        if any(isinstance(t, ast.Name) and t.id in BOOKKEEPING_TARGETS for t in targets):
            out.update(id(x) for x in ast.walk(n))
            n_book += 1
            continue
        # `evidence.append(...)` / `detail.extend(...)` —— 记账变量的**方法调用**。
        # 只挡赋值会漏掉这一种：C6 就是先把 evidence 拼进变量、再 `append` 追加一条，
        # 于是那条里拼字符串的生成式仍被算成 `expr_project`。
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) \
                and n.func.value.id in BOOKKEEPING_TARGETS:
            out.update(id(x) for x in ast.walk(n))
            n_book += 1
    return out, n_book


# ============================================================================
# 2. 骨架归一化（只用于表达式级证据展示）
# ============================================================================


class Canon(ast.NodeTransformer):
    """把一段表达式抹成**结构骨架**。

    规则：所有标识符 → `v`（变量名跨程序不可比）；下标里的字符串与数字
    **原样保留**（`["extent_m"]` 与 `["centroid_m"]` 是两件不同的事）；
    其它字符串 → `s`，其它数字 → `n`；内建与工具名保留，未知函数 → `f`。
    """

    def __init__(self, keep: frozenset[str]) -> None:
        self.keep = keep

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return ast.Name(id="v", ctx=ast.Store() if isinstance(node.ctx, ast.Store) else ast.Load())

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = "v"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        node.value = self.visit(node.value)  # type: ignore[assignment]
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        node.value = self.visit(node.value)  # type: ignore[assignment]
        return node  # `node.slice` 不动 —— 字段名与索引必须保留

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        if isinstance(node.value, bool):
            return node
        if isinstance(node.value, (int, float)):
            return _name("n")
        if isinstance(node.value, str):
            return _name("s")
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        f = node.func
        if isinstance(f, ast.Name):
            node.func = ast.Name(id=(f.id if f.id in self.keep else "f"), ctx=ast.Load())
        else:
            node.func = self.visit(f)  # type: ignore[assignment]
        node.args = [self.visit(a) for a in node.args]  # type: ignore[misc]
        node.keywords = [self.visit(k) for k in node.keywords]  # type: ignore[misc]
        return node


def skeleton(node: ast.AST, keep: frozenset[str]) -> str:
    try:
        out = ast.unparse(Canon(keep).visit(ast.parse(ast.unparse(node))))
    except Exception:  # noqa: BLE001 —— 诊断工具，任何解析意外都退化成「看不清」
        return "?"
    return _LAMBDA_DUP_RE.sub("lambda v, _", out)


# ============================================================================
# 3. 扫描（循环累积）的归类
# ============================================================================


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


def _best_update_if(node: ast.For | ast.While) -> ast.If | None:
    """找出「留最优」的那个 `if`：它的条件里读的变量，正是它自己体内写的变量。

    这条判据把「留最优」与「计数」干净地分开：
      · `if area > best_area: best_area = area`  ⟹ 条件读 `best_area`，体内写 `best_area` ✓
      · `if h > 0.8: count += 1`                 ⟹ 条件不读 `count` ✗
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.If):
            continue
        written = _assigned_names(n.body)
        if not written:
            continue
        for t in ast.walk(n.test):
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load) and t.id in written:
                return n
    return None


def _is_counter_loop(node: ast.For | ast.While) -> bool:
    return any(isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
               and isinstance(n.target, ast.Name) for n in ast.walk(node))


def _is_collector_loop(node: ast.For | ast.While) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ("append", "add", "extend", "update"):
            return True
        # `d[k] = ...` 这种字典收集也算
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) for t in n.targets):
            return True
    return False


def _is_pairwise(node: ast.For | ast.While) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "combinations" for n in ast.walk(node))


def _key_kind(node: ast.For | ast.While, best_if: ast.If | None) -> str:
    """被比较的那个量，是**现成字段**还是**算出来的**？这决定能不能完整提升。"""
    if best_if is None:
        return "n/a"
    # 条件里读、且体内写的变量 = 保持最优的那个变量
    written = _assigned_names(best_if.body)
    watched = [t.id for t in ast.walk(best_if.test)
               if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load) and t.id in written]
    rhs_for: dict[str, ast.expr] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    rhs_for[t.id] = n.value

    for w in watched:
        # 候选值来自「别的变量」时，顺着找它的来源（如 C3 的 `h = ext.get("h")`）
        rhs = rhs_for.get(w)
        if rhs is not None and isinstance(rhs, ast.Name) and rhs.id in rhs_for:
            rhs = rhs_for[rhs.id]
        if rhs is None:
            continue
        src = ast.unparse(rhs)
        if "** 2" in src and ("** 0.5" in src or "sqrt" in src):
            return "euclidean"
        has_binop = any(isinstance(x, ast.BinOp) for x in ast.walk(rhs))
        if has_binop:
            return "arithmetic"
        if any(isinstance(x, (ast.Subscript, ast.Attribute)) for x in ast.walk(rhs)):
            return "field"
    return "unknown"


def _numeric_literal_outside_subscript(node: ast.AST) -> bool:
    """节点里有没有数字字面量 —— **下标整棵子树不算**。

    ⚠ 这条排除是必需的，而且它属于一个**反复出现的形状**：
    「一个不是量的数字漏进了判断」。

      · `verifier` 那边：`evidence["method"] = "geometry_v1"` 里的 `1`
        被数字正则抠出来，进了证据池（2026-09-19 抓到）；
      · 这里：`z < sofa_c[2]` 的**下标** `2` 被当成「与常数比较的阈值」，
        于是 C8b 被判成 `threshold`，而它其实是两个量互比（2026-09-20 抓到）。

    两次的形状一样：**判断「这是不是量」时，忘了问「它出现在哪个位置」**。
    """
    if isinstance(node, ast.Subscript):
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    return any(_numeric_literal_outside_subscript(c) for c in ast.iter_child_nodes(node))


def _cmp_kind(test: ast.expr) -> str:
    """把一个条件式归成「与常数比」（threshold）还是「两个量互比」（field_compare）。"""
    for c in ast.walk(test):
        if isinstance(c, ast.Compare):
            operands = [c.left, *c.comparators]
            if any(_numeric_literal_outside_subscript(o) for o in operands):
                return "threshold"
            return "field_compare"
    return "none"


def _guard_ifs(node: ast.For | ast.While) -> list[ast.If]:
    """**只做排除**的 `if`：体内只有 `continue` / `pass` / `raise`。

    这类条件不是「数据谓词」—— 典型是 `if o["object_id"] == anchor: continue`
    （排除自己）。把它算成谓词，会让 `extreme_by` 看起来「有自由谓词」，
    从而把一个**本可完整提升**的形态误判成「只能提一半」。
    """
    out: list[ast.If] = []
    for n in ast.walk(node):
        if not isinstance(n, ast.If) or len(n.body) != 1:
            continue
        only = n.body[0]
        if isinstance(only, (ast.Continue, ast.Pass, ast.Break)):
            out.append(n)
        elif isinstance(only, ast.Raise):
            out.append(n)
    return out


def _center_if(node: ast.For | ast.While) -> ast.If | None:
    """**承载累加**的那个 `if`：体内含 `x += …` 或 `list.append(...)` / `d[k] = …`。

    众数谓词必须是**它**的条件 —— 而不是「循环里第一个 `ast.Compare`」。
    第一版就是这么错的：C4 的真谓词是 `if h > 0.8`（阈值），却被前面那句
    `if h is None: continue` 抢先匹配成了 `field_compare`。
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.If):
            continue
        for s in n.body:
            for x in ast.walk(s):
                if isinstance(x, ast.AugAssign) and isinstance(x.op, ast.Add) \
                        and isinstance(x.target, ast.Name):
                    return n
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) \
                        and x.func.attr in ("append", "add", "extend", "update"):
                    return n
                if isinstance(x, ast.Assign) and any(
                        isinstance(t, ast.Subscript) for t in x.targets):
                    return n
    return None


def _predicate_kind(node: ast.For | ast.While, tool_names: frozenset[str],
                    center_if: ast.If | None) -> str:
    """循环的**数据谓词**属于哪一类。

    判据刻意用「**循环体内是否直接调用了工具**」而不是抠条件式：
    C8 把工具调用写在 `if` 之前（`r = query_relation(...)` 然后 `if r.ok and
    r.value is True`），只看条件式会把它误判成「与常数比较」。

    没有数据谓词的循环返回 `none`（不筛，全量归约）或 `guard`（只排除自己）。
    """
    if center_if is not None:
        return "tool_call" if tool_calls_of(node, tool_names) else _cmp_kind(center_if.test)
    if _guard_ifs(node):
        return "guard"
    return "none"


def classify_scan(node: ast.For | ast.While, tool_names: frozenset[str]) -> dict[str, Any]:
    best_if = _best_update_if(node)
    center_if = _center_if(node)
    pred = _predicate_kind(node, tool_names, center_if)
    tools = sorted(tool_calls_of(node, tool_names))
    if _is_pairwise(node):
        kind = "scan_extreme_pair"
    elif best_if is not None:
        kind = "scan_extreme_by"
    elif _is_counter_loop(node):
        kind = "scan_count_where" if pred not in ("none", "guard") else "scan_count_all"
    elif _is_collector_loop(node):
        kind = "scan_collect"
    else:
        kind = "scan_other"
    if kind == "scan_collect":
        # 收集型循环的「谓词」栏统一成「有没有在循环里逐项调工具」。
        # 否则这一栏取决于「那个 `if` 里恰好有没有 `append`」——
        # C5 的 `if not ext.ok: submit(...)` 里没有，就被判 `none`；
        # C6 的 `if r.ok:` 里有下标赋值，就被判 `tool_call`。
        # 两者其实都在循环里调了工具。同一个观测点被实现细节左右，就不可比了。
        pred = "tool_call" if tools else "none"
    return {
        "kind": kind,
        "key_kind": _key_kind(node, best_if),
        "predicate": pred,
        "tools_called": tools,
        "n_lines": len(ast.unparse(node).splitlines()),
    }


def scan_signature(scan: dict[str, Any]) -> str:
    """形态签名 —— **分组用的键**。刻意只保留决策相关的三个维度。"""
    return "%s｜key=%s｜pred=%s" % (scan["kind"], scan["key_kind"], scan["predicate"])


# ============================================================================
# 4. 表达式级（非循环、非 submit）组合
# ============================================================================

_EXPR_FUNCS = frozenset({"max", "min", "sum", "mean", "median", "sorted", "len", "filter", "map"})


def classify_expr(node: ast.AST) -> str:
    called = _called_names(node)
    is_comp = any(isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
                  for n in ast.walk(node))
    comp_if = any(isinstance(n, ast.comprehension) and bool(n.ifs) for n in ast.walk(node))
    has_div = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(node))
    is_sort = ("sort" in called)

    if ("mean" in called or "median" in called) or ("sum" in called and ("len" in called or has_div)):
        return "expr_aggregate_mean"
    if "sum" in called:
        return "expr_aggregate_sum"
    if is_sort or "sorted" in called:
        return "expr_rank"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "len" and node.args:
        arg = node.args[0]
        if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
            return "expr_len_local"
        if is_comp:
            return "expr_count_generated"
    if ("max" in called or "min" in called) and any(
            isinstance(n, ast.Call) and any(k.arg == "key" for k in n.keywords)
            for n in ast.walk(node)):
        return "expr_extreme"
    if comp_if:
        return "expr_filter"
    if is_comp:
        return "expr_project"
    return "expr_other"


def collect_expr_sites(tree: ast.AST) -> list[ast.AST]:
    """取最外层（不再往里钻）的表达式级组合节点，跳过记账与循环内部。"""
    excluded, _n = _excluded_ids(tree)
    out: list[ast.AST] = []

    def walk(n: ast.AST) -> None:
        if id(n) in excluded:
            return
        if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            out.append(n)
            return
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in _EXPR_FUNCS:
                out.append(n)
                return
            if isinstance(n.func, ast.Attribute) and n.func.attr == "sort":
                out.append(n)
                return
        for c in ast.iter_child_nodes(n):
            walk(c)

    walk(tree)
    out.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    return out


# ============================================================================
# 5. 对一份产物做挖掘
# ============================================================================


def _default_run() -> Path:
    """默认取**最新一次真跑**。

    只认不带 `_reanalyzed` 的（命名规则：真跑永不被覆盖，复算重算即覆盖）。
    正因为真跑不会被覆盖，这一子集里的 `mtime` 才等于「什么时候跑的」——
    对 `_reanalyzed` 绝不能这么用（见 `reports/README.md`）。
    """
    cands = [p for p in REPORTS.glob("combination_run_*.json") if "_reanalyzed" not in p.name]
    if not cands:
        raise SystemExit("[mine-glue] reports/ 下没有真跑产物（combination_run_*.json）")
    return max(cands, key=lambda p: p.stat().st_mtime)


def mine(run_path: Path, tool_names: frozenset[str]) -> dict[str, Any]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = list(payload.get("runs") or [])
    keep = tool_names | BUILTIN_KEEP

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    sig_qids: dict[str, set[str]] = {}
    sig_info: dict[str, dict[str, Any]] = {}
    kind_qids: dict[str, set[str]] = {}
    expr_qids: dict[str, set[str]] = {}
    expr_examples: dict[str, str] = {}
    per_item_tool_qids: set[str] = set()
    collect_per_item_tool_qids: set[str] = set()
    count_where_tool_qids: set[str] = set()
    loop_qids: set[str] = set()
    n_bookkeeping = 0

    for r in runs:
        qid = str(r.get("qid") or "?")
        src = str(r.get("program") or "")
        row: dict[str, Any] = {
            "qid": qid, "category": r.get("category"), "question": r.get("question"),
            "program": src, "tool_calls": r.get("tool_calls"),
            "one_call_matches_gt": bool(r.get("one_call_matches_gt")),
            "class": r.get("class"), "answer": r.get("answer"), "gt": r.get("gt"),
            "scans": [], "exprs": [], "has_real_comprehension": False, "parse_error": None,
        }
        if not src.strip():
            row["parse_error"] = "程序为空（跑失败或未提交）"
            failed.append({"qid": qid, "why": row["parse_error"]})
            rows.append(row)
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            row["parse_error"] = "SyntaxError: %s" % exc
            failed.append({"qid": qid, "why": row["parse_error"]})
            rows.append(row)
            continue

        row["has_real_comprehension"] = any(
            isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
            for n in ast.walk(tree))

        inner_loops = _inner_loop_ids(tree)
        for loop in [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]:
            if id(loop) in inner_loops:  # 嵌套循环只报最外层
                continue
            scan = classify_scan(loop, tool_names)
            row["scans"].append(scan)
            loop_qids.add(qid)
            # 两种「循环里调工具」是**不同**的问题，修法也不同，必须分开记：
            #   · 收集型循环里逐项调工具 ⟹ list_objects 已经给了那些字段，是**冗余**
            #   · 计数型循环里调工具做谓词   ⟹ 该塌成**一次** count_relation 调用
            if scan["tools_called"]:
                if scan["kind"] == "scan_collect":
                    collect_per_item_tool_qids.add(qid)
                elif scan["kind"] == "scan_count_where":
                    count_where_tool_qids.add(qid)
            sig = scan_signature(scan)
            sig_qids.setdefault(sig, set()).add(qid)
            sig_info.setdefault(sig, {
                "signature": sig, "kind": scan["kind"],
                "key_kind": scan["key_kind"], "predicate": scan["predicate"],
                "n_lines": scan["n_lines"], "example_qid": qid,
                "example_source": (ast.get_source_segment(src, loop) or "")[:600],
            })
            kind_qids.setdefault(scan["kind"], set()).add(qid)

        for node in collect_expr_sites(tree):
            kind = classify_expr(node)
            sk = skeleton(node, keep)
            row["exprs"].append({
                "kind": kind, "skeleton": sk,
                "lineno": getattr(node, "lineno", None),
                "fields": sorted(set(_FIELD_RE.findall(sk))),
            })
            expr_qids.setdefault(kind, set()).add(qid)
            expr_examples.setdefault(kind, sk)

        # 被「记账变量名」挡掉的条数 —— 排除名单必须可核对（见 _excluded_ids）
        for n in ast.walk(tree):
            hit = (isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in BOOKKEEPING_TARGETS
                           for t in n.targets))
            hit = hit or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and isinstance(n.func.value, ast.Name)
                          and n.func.value.id in BOOKKEEPING_TARGETS)
            if hit:
                n_bookkeeping += 1

        rows.append(row)

    return {
        "run_path": str(run_path), "scene": payload.get("scene"),
        "toolset": list(payload.get("toolset") or []),
        "n_runs": len(runs), "failed": failed,
        "rows": rows,
        "sig_qids": {k: sorted(v) for k, v in sig_qids.items()},
        "sig_info": sig_info,
        "kind_qids": {k: sorted(v) for k, v in kind_qids.items()},
        "expr_qids": {k: sorted(v) for k, v in expr_qids.items()},
        "expr_examples": expr_examples,
        "n_loop_qids": len(loop_qids),
        "n_bookkeeping": n_bookkeeping,
        "n_per_item_tool_qids": len(per_item_tool_qids),
        "per_item_tool_qids": sorted(per_item_tool_qids),
        "collect_per_item_tool_qids": sorted(collect_per_item_tool_qids),
        "count_where_tool_qids": sorted(count_where_tool_qids),
        "n_real_comprehension_qids": sum(1 for r in rows if r["has_real_comprehension"]),
        "tool_names": sorted(tool_names),
    }


def _inner_loop_ids(tree: ast.AST) -> set[int]:
    """**嵌套循环的内层**的节点 id —— 用于只报最外层循环，避免一个双层循环报两次。"""
    out: set[int] = set()
    for n in ast.walk(tree):
        if not isinstance(n, (ast.For, ast.While)):
            continue
        for c in ast.walk(n):
            if isinstance(c, (ast.For, ast.While)) and c is not n:
                out.update(id(x) for x in ast.walk(c))
    return out


# ============================================================================
# 6. 出报告
# ============================================================================


def _promotability(scan: dict[str, Any]) -> tuple[str, str]:
    """一个扫描能提到什么程度。**这是本报告最核心的一列。**"""
    kind = scan["kind"]
    if kind in ("scan_count_all", "scan_other"):
        return "—", SCAN_KIND_INFO.get(kind, ("？", ""))[1]
    if kind == "scan_collect":
        return "要看下游", "单看循环提不出算子，得连同下游的 aggregate/rank 一起提"
    free = []
    if scan["key_kind"] in FREE_KEY_KINDS:
        free.append("键是算式")
    if scan["predicate"] in FREE_PREDICATES:
        free.append("谓词是自由条件")
    if free:
        return "只能提一半", "＋".join(free) + " ⟹ 参数里必须保留表达式"
    if scan["key_kind"] == "euclidean":
        return "**可完整提升**", "距离是标准度量 ⟹ 提成 `metric=` 参数即可完整表达"
    return "**可完整提升**", "键/谓词都是数据（字段投影 / 已是工具调用 / 只排除自己）"


def _suggest(scan: dict[str, Any]) -> tuple[str, str]:
    """这个形态最该被提成什么。**比大类那一层的建议精确一档。**"""
    kind, key, pred = scan["kind"], scan["key_kind"], scan["predicate"]
    if kind == "scan_extreme_by":
        if key == "field":
            return "extreme_by", "`extreme_by(how='max'|'min', field='extent_m.h')` —— **完全覆盖**"
        if key == "euclidean":
            return "nearest", "`nearest_to(anchor, metric='euclidean')`，或**加宽 `find_nearest`**"
        return "extreme_by ⚠", "`extreme_by(field=…)` 覆盖不了 —— 键是算式（如「最大两边之积」）"
    if kind == "scan_extreme_pair":
        return "extreme_pair", "`extreme_pair(field='centroid_m', metric='euclidean')` —— **完全覆盖**"
    if kind == "scan_count_where":
        if pred == "tool_call":
            return "count_relation", "`count_relation(relation, anchor, tol)` —— 谓词已是工具，整段循环可塌成一次调用"
        return "count_where ⚠", "`count_where(predicate)` —— 谓词是自由条件，只能提一半"
    if kind == "scan_count_all":
        return "已被覆盖", "`list_objects` 已给 `returned`"
    if kind == "scan_collect":
        return "（看下游）", "配合下游 `aggregate` / `rank_by` 一起提"
    return "？", "人工判读"


def build_report(m: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = m["rows"]
    n = m["n_runs"]
    total_calls = sum(int(r.get("tool_calls") or 0) for r in rows)
    n_scan = sum(len(r["scans"]) for r in rows)
    L: list[str] = []

    L.append("# 胶水形态挖掘（从真跑程序反推算子候选）")
    L.append("")
    L.append("> 由 `scripts/mine_glue_patterns.py` 生成。**零成本**：只读已落盘的程序源码，不调模型。")
    L.append("> 每题程序**原文**另存 `reports/glue_programs.txt`（`--dump`）—— 结论要能被独立核对。")
    L.append("")

    L.append("## 〇 读的是哪一份产物（口径）")
    L.append("")
    L.append("- 文件：`%s`" % Path(m["run_path"]).name)
    L.append("- 场景：`%s` ｜ 工具空间：%d 个" % (m["scene"], len(m["toolset"])))
    L.append("- 题数：**%d** ｜ 工具调用合计 **%d** 次（平均 %.1f 次/题）" % (n, total_calls, (total_calls / n) if n else 0))
    L.append("- **循环扫描：%d 个**，分布在 **%d/%d** 道题上" % (n_scan, m["n_loop_qids"], n))
    L.append("- 解析失败：**%d** 题%s" % (len(m["failed"]), "（见第五节）" if m["failed"] else " ✓"))
    L.append("")
    L.append("⚠ 样本范围：**一个场景、一张图、%d 道题**。所有「N 题用过」只在样本内成立，" % n)
    L.append("跨场景未验证 —— 别当普适结论引用。")
    L.append("")

    # ---- 修正条：第一版错的 ----
    L.append("### 顺带修正一个被引用的数字")
    L.append("")
    L.append("真实推导式（`[... for ...]` / `(... for ...)`）只出现在 **%d/%d** 道题上；"
             % (m["n_real_comprehension_qids"], n))
    L.append("**%d/%d 道题是用显式 `for` 循环 + 累加变量**干活的。"
             % (m["n_loop_qids"], n))
    L.append("而 `probe_combination_run.py` 里的 `has_comprehension` 用的正则是")
    L.append("`\\bfor\\b[^\\n]*\\b in \\b` —— 它**同样匹配普通 `for x in y:` 语句**。")
    L.append("所以「9/10 用了推导式」这句话，实际含义是「9/10 里出现过 `for ... in`」。")
    L.append("⟹ 引用时必须写成「**%d/%d 写了显式循环扫描**」，而不是「用了推导式」。"
             % (m["n_loop_qids"], n))
    L.append("")

    # ---- 决策表：形态大类 ----
    L.append("## 一 按形态大类聚合（这张表是决策用的）")
    L.append("")
    L.append("| 形态大类 | 语义 | 题数 | 用到它的题 | 建议算子 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    ranked = sorted(m["kind_qids"].items(), key=lambda kv: (-len(kv[1]), kv[0]))
    qualify = [(k, q) for k, q in ranked if len(q) >= 2]
    for kind, qids in ranked:
        op, desc = SCAN_KIND_INFO.get(kind, ("？", ""))
        verdict = "**够格**（≥2 题）" if len(qids) >= 2 else "n=1，样本不足"
        L.append("| `%s` | %s | **%d** | %s | `%s` | %s |"
                 % (kind, desc, len(qids), "、".join(qids), op, verdict))
    L.append("")
    L.append("⟹ **%d 个形态大类里 %d 个够格**（其余是单题偶发，不作为提升依据）。"
             % (len(ranked), len(qualify)))
    L.append("")

    # ---- 子形态 × 可提升性 ----
    L.append("## 二 子形态 × 可提升性（决定「能提多少」）")
    L.append("")
    L.append("同一个大类里，键/谓词是不是**自由形式**，决定了能不能完整提升：")
    L.append("")
    L.append("| 形态签名 | 题数 | 题 | 可提升性 | 建议算子 |")
    L.append("|---|---|---|---|---|")
    for sig, qids in sorted(m["sig_qids"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        info = m["sig_info"][sig]
        level, _why = _promotability(info)
        opname, opnote = _suggest(info)
        L.append("| `%s` | **%d** | %s | %s | `%s` —— %s |"
                 % (sig, len(qids), "、".join(qids), level, opname, opnote))
    L.append("")

    free_sigs = [s for s in m["sig_qids"] if _promotability(m["sig_info"][s])[0] == "只能提一半"]
    full_sigs = [s for s in m["sig_qids"]
                 if _promotability(m["sig_info"][s])[0] == "**可完整提升**"]
    L.append("**可完整提升**的签名 %d 个；**只能提一半**的签名 %d 个。"
             % (len(full_sigs), len(free_sigs)))
    L.append("")
    if full_sigs:
        L.append("可完整提升的那些（键/谓词都是数据）：")
        L.append("")
        for s in sorted(full_sigs):
            info = m["sig_info"][s]
            opname, opnote = _suggest(info)
            L.append("- `%s`（%s）⟹ `%s`" % (s, "、".join(m["sig_qids"][s]), opname))
            L.append("  - %s" % opnote)
        L.append("")
    if free_sigs:
        L.append("⟹ 「只能提一半」这一档**在真实 trace 上有实例**，不是假想：")
        for s in sorted(free_sigs):
            L.append("- `%s`（%s）" % (s, "、".join(m["sig_qids"][s])))
        L.append("")
        L.append("把这类结构提成算子，**只是把代码从题里挪到参数里**，自由形式并没有消失。")
        L.append("这就是「纯菜单不可行、混合才是终局」的实测依据。")
        L.append("")
    else:
        L.append("⚠ 本次样本里没有出现自由形式的键/谓词 —— **但别据此下结论**，")
        L.append("样本只有 %d 题，且判据是「键/谓词是否含算式与阈值」这种结构特征。" % n)
        L.append("")

    # ---- 最便宜的一步 ----
    widen = [s for s in m["sig_qids"]
             if m["sig_info"][s]["kind"] == "scan_extreme_by"
             and m["sig_info"][s]["key_kind"] == "euclidean"]
    if widen:
        L.append("### 最便宜的一步：加宽一个已有参数（**零新工具**）")
        L.append("")
        L.append("这些题是「按距离取最近/最远」，但候选**不限类别**（%s）——"
                 % "、".join(sorted({q for s in widen for q in m["sig_qids"][s]})))
        L.append("`find_nearest` / `find_farthest` **已经存在**，只是 `label` 参数实际上必填")
        L.append("（空串会走 `by_label(\"\")` → `NOT_FOUND`），于是模型只能手写距离循环。")
        L.append("把它放宽成「`label` 省略 = 全部类别」，比新增任何算子都便宜，")
        L.append("而且**不改动作空间长度**（工具数不变，只改一个参数的可选性）。")
        L.append("")

    # ---- 循环里逐个调工具 ----
    coll = m["collect_per_item_tool_qids"]
    cnt = m["count_where_tool_qids"]
    if coll or cnt:
        L.append("### 附带发现：循环里逐个调工具（两种，**修法不同**）")
        L.append("")
        if coll:
            L.append("**① 收集型循环里逐项调工具（%d 题：%s）—— 是冗余，但**文档已经写过**。**"
                     % (len(coll), "、".join(coll)))
            L.append("`list_objects` 的返回值里**已经有那些字段**（`_brief` 带 `centroid_m` +")
            L.append("`extent_m{w,h,l}`），逐项再调一次属于重复取数。")
            L.append("")
            L.append("⚠ 但它**不是**「文档没写」这一类。`list_objects` 的**第一段**"
                     "（即渲染进提示词的那段）原文写着：")
            L.append("")
            L.append("> **所有物体的质心与尺寸一次给全**，找极值/计数/求均值不必再逐个调工具。")
            L.append("")
            L.append("而 `tools/spatial.py` 的最后一次修改（09-19 21:42）**早于**这次真跑"
                     "（`..._214625`，21:46），")
            L.append("也就是说该次跑完之后这个文件再没被改过 ⟹ **这句话模型当时看得到，它没有照做**。")
            L.append("")
            L.append("⟹ 与「返回值形状没写进第一段」（§教训 2）**不是同一类**：那次是**看不到**，")
            L.append("这次是**看到不照做**。所以修法是提示词纪律，**不是**再改一遍文档。")
            L.append("")
        if cnt:
            L.append("**② 计数型循环里调工具当谓词（%d 题：%s）—— 该塌成一次调用。**"
                     % (len(cnt), "、".join(cnt)))
            L.append("谓词本身已经是工具调用，所以整段「逐个调 + 计数」可以塌成**一次**")
            L.append("`count_relation(...)`。C8 的 **10 次**调用就是这么来的。")
            L.append("与 ① 的区别：① 省的是重复取数，② 省的是**调用次数本身**。")
            L.append("")

    # ---- 每题配方 ----
    L.append("## 三 每题到底写了什么（配方，证据用）")
    L.append("")
    L.append("| 题号 | 扫描 | 表达式级组合 | 工具调用 | 一次调用够不够 |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        scans = " → ".join("`%s`" % s["kind"] for s in r["scans"]) or "—"
        exprs = " → ".join("`%s`" % e["kind"] for e in r["exprs"]) or "—"
        oc = "够（省冗余）" if r["one_call_matches_gt"] else "**不够（真需要组合）**"
        L.append("| %s | %s | %s | %s | %s |" % (r["qid"], scans, exprs, r.get("tool_calls"), oc))
    L.append("")
    n_avoid = sum(1 for r in rows if r["one_call_matches_gt"])
    L.append("**%d/%d 题「一次工具调用就够」**（`one_call_matches_gt`）。" % (n_avoid, n))
    L.append("⟹ 这些题上，胶水是**可避的冗余**而不是能力缺口：提升算子省的是调用次数与")
    L.append("出错面，**不增加可回答问题的集合**。要证明「加工具真的扩大了能力」，")
    L.append("得看剩下的那 %d 题。" % (n - n_avoid))
    L.append("")

    # ---- 表达式级明细 ----
    L.append("## 四 表达式级组合明细（非循环、非 submit）")
    L.append("")
    L.append("| 类型 | 语义 | 题数 | 题 | 举例（骨架） |")
    L.append("|---|---|---|---|---|")
    for kind, qids in sorted(m["expr_qids"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        label, desc = EXPR_KIND_INFO.get(kind, ("？", ""))
        L.append("| `%s` | %s | %d | %s | `%s` |"
                 % (kind, desc, len(qids), "、".join(qids),
                    m["expr_examples"].get(kind, "")[:90]))
    L.append("")
    L.append("⚠ `expr_len_local` 刻意**不进决策表**：`len(本地列表)` 是 Python 层面的记账，")
    L.append("不是「场景里有几个东西」。第一版把它当成最大的桶，是那版报告最大的错。")
    L.append("")
    L.append("另有 **%d 条**赋值给记账变量（`evidence` / `detail` / `order` 之类）的表达式"
             % m["n_bookkeeping"])
    L.append("**被排除在外、不上表** —— 它们是拼呈堂证供，不是把场景归约成答案。")
    L.append("排除名单只认**变量名**（`BOOKKEEPING_TARGETS`），换个名字就漏；")
    L.append("所以这里如实报出条数，让「排除掉了什么」能被核对。")
    L.append("")

    # ---- 没挖到的 ----
    L.append("## 五 没挖到的")
    L.append("")
    if m["failed"]:
        L.append("这些题**不进统计**（不是「没有组合」，是「看不见」）：")
        L.append("")
        for f in m["failed"]:
            L.append("- `%s`：%s" % (f["qid"], f["why"]))
    else:
        L.append("所有题的程序都解析成功，没有静默漏掉的。")
    L.append("")

    # ---- 形态明细 ----
    L.append("## 六 形态明细（每条带循环原文）")
    L.append("")
    for sig, qids in sorted(m["sig_qids"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        info = m["sig_info"][sig]
        L.append("### `%s` ｜ %d 题：%s ｜ %d 行" % (sig, len(qids), "、".join(qids), info["n_lines"]))
        L.append("")
        L.append("```python")
        L.append(info["example_source"].rstrip())
        L.append("```")
        L.append("")

    L.append("## 七 本工具看不到什么（用了它的结论就必须一起说）")
    L.append("")
    L.append("1. **只看到模型写了什么，看不到它需要什么。** 没被写出来的需求不在样本里 ——")
    L.append("   它**不能**回答「还缺哪些算子」，只能回答「哪些形态重复出现过」。")
    L.append("2. **n=1 不是「不需要」，是「没证据」。** 单题偶发的形态，缺的是样本不是价值。")
    L.append("3. **归一化会高估**（不同变量名被抹平）**也会低估**（等价写法并不起来）。")
    L.append("   两个方向都有，别把「%d 个签名」当精确值。" % len(m["sig_qids"]))
    L.append("4. **它不判断对错。** 程序写错但形态漂亮的照样进表 —— 正确性在")
    L.append("   `probe_combination_run.py` 的 `class` 字段里，不在这里。")
    L.append("5. **它不提议现在就加算子。** 判据只到「够不够格进入讨论」为止；")
    L.append("   真加之前还要过「加工具不静默改变已跑臂」那一关（`TOOLS_VERSION` + 消融臂）。")
    L.append("6. **形态归类是启发式**：`_best_update_if` 的判据是「条件读的变量正是体内写的」——")
    L.append("   把「留最优」与「计数」分开，但它对更绕的写法会失手。失手的样子是 `scan_other`，")
    L.append("   不会静默算进别的桶。")
    L.append("")
    return "\n".join(L)


# ============================================================================
# 7. 自检 —— 冻结住这把尺子
# ============================================================================


def selftest() -> int:
    """把**归类的性质**钉死。

    为什么钉性质而不是钉字符串：钉字符串的话 `ast.unparse` 换版本就全线报错，
    而那并不是缺陷。这里要保的是真正影响结论的那几条。
    """
    tool_names = frozenset({"list_objects", "find_object", "query_relation"})
    keep = tool_names | BUILTIN_KEEP
    fails: list[str] = []

    def scan_of(src: str) -> dict[str, Any]:
        tree = ast.parse(src)
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        return classify_scan(loops[0], tool_names)

    best = scan_of("for o in objs:\n    if h > 0.8:\n        count += 1\n")
    if best["kind"] != "scan_count_where":
        fails.append("`if h > 0.8: count += 1` 应归 scan_count_where，实际 %s" % best["kind"])
    if best["predicate"] != "threshold":
        fails.append("`if h > 0.8` 的谓词应判 threshold，实际 %s" % best["predicate"])

    argmax = scan_of("for p in pics:\n    if a > b:\n        b = a\n        best = p\n")
    if argmax["kind"] != "scan_extreme_by":
        fails.append("「条件读的变量 == 体内写的变量」应归 scan_extreme_by，实际 %s" % argmax["kind"])

    tool_pred = scan_of("for o in objs:\n    r = query_relation(a=o, b=t)\n    if r.ok:\n        count += 1\n")
    if tool_pred["predicate"] != "tool_call":
        fails.append("循环体内调了工具，谓词应判 tool_call，实际 %s" % tool_pred["predicate"])

    pair = scan_of("for a, b in combinations(objs, 2):\n    if d > best:\n        best = d\n")
    if pair["kind"] != "scan_extreme_pair":
        fails.append("`combinations` 上的扫描应归 scan_extreme_pair，实际 %s" % pair["kind"])

    # 归一化：变量名不同要并，字段名不同不能并
    a = ast.parse('max(x1, key=lambda p: p["extent_m"][0])').body[0].value
    b = ast.parse('max(y2, key=lambda q: q["extent_m"][0])').body[0].value
    c = ast.parse('max(x1, key=lambda p: p["centroid_m"][0])').body[0].value
    if skeleton(a, keep) != skeleton(b, keep):
        fails.append("变量名不同但结构相同的两段代码没有归成一个骨架（会低估重复度）")
    if skeleton(a, keep) == skeleton(c, keep):
        fails.append("字段名不同被归成了同一个骨架（会造出假候选）")

    # submit 内部不算胶水
    tree = ast.parse('submit(x, evidence=[len(y)])\n')
    if collect_expr_sites(tree):
        fails.append("`submit(..., evidence=[len(y)])` 里的 len 不该被当成胶水站点")

    # 循环内部不再重复报表达式
    tree = ast.parse("for o in objs:\n    items.append((o['a'], 1))\n")
    kinds = [classify_expr(x) for x in collect_expr_sites(tree)]
    if kinds:
        fails.append("循环内部的表达式不该被单独报出来，却报出了 %s" % kinds)

    # ★ 回归用例：这一条正是第一版错的地方 —— 真谓词在第二个 `if` 里，
    #   而前面那个 `if h is None: continue` 是守卫。第一版取「第一个 Compare」，
    #   把 C4 判成了 field_compare（真值是 threshold）。
    guarded = scan_of(
        "for o in objs:\n"
        "    h = o.get('h')\n"
        "    if h is None:\n"
        "        continue\n"
        "    if h > 0.8:\n"
        "        count += 1\n")
    if guarded["kind"] != "scan_count_where":
        fails.append("带守卫的计数循环应归 scan_count_where，实际 %s" % guarded["kind"])
    if guarded["predicate"] != "threshold":
        fails.append("「守卫在前、阈值在后」应判 threshold，实际 %s（第一版就是错在这里）"
                     % guarded["predicate"])

    # ★ 回归用例：**下标里的数字不是阈值**。`z < sofa_c[2]` 是两个量互比，
    #   但 `[2]` 里的 `2` 是 Constant. 不排掉下标，它就会被判成 threshold ——
    #   与 `geometry_v1` 里的 `1` 进证据池是同一个形状的错。
    subscript = scan_of("for o in objs:\n    if z < sofa_c[2]:\n        count += 1\n")
    if subscript["predicate"] != "field_compare":
        fails.append("`z < sofa_c[2]` 应判 field_compare（下标数字不算阈值），实际 %s"
                     % subscript["predicate"])

    # ★ 回归用例：`continue` 排除自己是**守卫**，不是数据谓词。
    #   算成谓词会把「可完整提升」的 extreme_by 误判成「只能提一半」。
    excl = scan_of(
        "for o in objs:\n"
        "    if o['object_id'] == anchor:\n"
        "        continue\n"
        "    if best is None or d < best:\n"
        "        best = d\n")
    if excl["kind"] != "scan_extreme_by":
        fails.append("排除自己的扫描应归 scan_extreme_by，实际 %s" % excl["kind"])
    if excl["predicate"] != "guard":
        fails.append("`if … continue` 应判 guard，实际 %s" % excl["predicate"])

    if fails:
        print("[mine-glue] 自检失败：")
        for f in fails:
            print("  · %s" % f)
        return 2
    print("[mine-glue] 自检通过（11 项：扫描归类 4 + 谓词/键 5 + 归一化 2）")
    return 0


# ============================================================================
# 8. 入口
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从真跑程序里挖胶水形态，反推算子候选")
    ap.add_argument("--run", default=None, help="真跑产物 JSON；默认取最新一次真跑")
    ap.add_argument("--out", default=None, help="输出 md 路径；同名 .json 一起写")
    ap.add_argument("--selftest", action="store_true", help="只跑归类自检，不读产物")
    ap.add_argument("--dump", default=None, help="把每题的程序原文写到这个文件")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    tool_names = read_tool_names()
    run_path = Path(args.run) if args.run else _default_run()
    if not run_path.exists():
        raise SystemExit("[mine-glue] 找不到 %s" % run_path)

    m = mine(run_path, tool_names)

    if args.dump:
        chunks: list[str] = ["# 真跑程序原文（%s）\n" % run_path.name]
        for r in m["rows"]:
            chunks.append("=" * 78)
            chunks.append("### %s ｜ %s ｜ 工具调用 %s ｜ 一次调用够不够 %s ｜ class %s"
                          % (r["qid"], r.get("category"), r.get("tool_calls"),
                             r.get("one_call_matches_gt"), r.get("class")))
            chunks.append("题：%s" % (r.get("question") or ""))
            chunks.append("-" * 78)
            chunks.append(str(r.get("program") or ""))
            chunks.append("")
        Path(args.dump).write_text("\n".join(chunks), encoding="utf-8")
        print("[mine-glue] 程序原文 → %s" % args.dump)

    md_path = Path(args.out) if args.out else REPORTS / "glue_mining.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_report(m), encoding="utf-8")
    md_path.with_suffix(".json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[mine-glue] 读 %s" % run_path.name)
    print("[mine-glue] %s" % md_path)
    print("[mine-glue] 扫描 %d 个 / %d 题 ｜ 形态大类 %d ｜ 签名 %d ｜ 解析失败 %d"
          % (sum(len(r["scans"]) for r in m["rows"]), m["n_loop_qids"],
             len(m["kind_qids"]), len(m["sig_qids"]), len(m["failed"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
