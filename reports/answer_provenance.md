# 答案出处（provenance）—— `points_probe`

每题：跑一遍它的**完美工具路线**，再看答案命中了 trace 里哪些 `(键, 值)`。
「命中 `evidence` 桶且键名不是量」= 巧合命中 —— 这正是 `method` 那一类的形状。

## C1「类内 argmax（按尺寸）」

- 答案 `'picture_1'` ｜ 结论 `supported` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 28 ｜ evidence 1 ｜ measures 28 ｜ populations (4.0,)

## C2「类内 argmin（按距离）」

- 答案 `'picture_2'` ｜ 结论 `supported` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 8 ｜ evidence 8 ｜ measures 15 ｜ populations (4.0,)

## C3「全场景 argmax」

- 答案 `'mirror_1'` ｜ 结论 `supported` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 63 ｜ evidence 3 ｜ measures 63 ｜ populations (9.0, 9.0, 9.0)

## C4「筛选 + 计数」

- 答案 `4` ｜ 结论 `weak` ｜ `matched_from` = `derived:within`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 63 ｜ evidence 3 ｜ measures 63 ｜ populations (9.0, 9.0, 9.0)

## C5「聚合（均值）」

- 答案 `0.4445` ｜ 结论 `weak` ｜ `matched_from` = `derived:within`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 28 ｜ evidence 1 ｜ measures 28 ｜ populations (4.0,)

## C6「排序」

- 答案 `'picture_4,picture_2,picture_3,picture_1'` ｜ 结论 `weak` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 28 ｜ evidence 1 ｜ measures 28 ｜ populations (4.0,)

## C7「两跳组合（跨类别最近）」

- 答案 `'table_1'` ｜ 结论 `supported` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 71 ｜ evidence 91 ｜ measures 159 ｜ populations (9.0, 9.0, 9.0)

## C8「关系计数（口径敏感）」

- 答案 `1` ｜ 结论 `weak` ｜ `matched_from` = `derived:within`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- ⚠ **修复前会命中**：`method=1.0`, `method=1.0`, `method=1.0`, `method=1.0`, `method=1.0`, `method=1.0`, `method=1.0`, `method=1.0` ⟹ 这就是被挡掉的假出处
- 池子规模：value 63 ｜ evidence 27 ｜ measures 87 ｜ populations (9.0, 9.0, 9.0)

## C8b「关系计数（远离死区）」

- 答案 `3` ｜ 结论 `weak` ｜ `matched_from` = `derived:within`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 63 ｜ evidence 27 ｜ measures 87 ｜ populations (9.0, 9.0, 9.0)

## C12「全局极值对（O(n²)）」

- 答案 `'picture_4,table_1'` ｜ 结论 `weak` ｜ `matched_from` = `（非数值答案：走 `text_backed` 文本查找）`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 99 ｜ evidence 399 ｜ measures 495 ｜ populations (9.0, 9.0, 9.0)

## C10「体积」

- 答案 `2.416642` ｜ 结论 `weak` ｜ `matched_from` = `derived:within`
- 命中 `value` 桶：—
- 命中 `evidence` 桶：—
- 池子规模：value 63 ｜ evidence 3 ｜ measures 63 ｜ populations (9.0, 9.0, 9.0)

---

⚠ **修复前会命中答案的题**：C8

这些就是**已经在真实 trace 上验证过**的假出处 —— 不是构造出来的例子。
两个机制一起把它们挡住：① `_ID_RE` 现在会连 `_v1` 这种版本号后缀一起抹掉；
② `verifier.NON_QUANTITY_KEYS` 按**键名**把这些子树整棵挡在数字池外。
于是它们在上表「命中」一栏是 `—`，答案落到 `derived:*` → `weak`。

⚠ **这张表查不到什么**：它只重放**已经写进 `NON_QUANTITY_KEYS` 的键名**。
一个用了新键名的同类泄漏，在这张表上同样是 `—`。
要往下走一步，得换一种问法：不是「这个键是不是量」，
而是「答案有没有**独立于 trace** 的出处」——那需要契约层面的改动，本轮不做。
