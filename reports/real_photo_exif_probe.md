# 真机照片 EXIF 实测：微信传图会把 EXIF 丢光（2026-09-23）

**一句话**：两张 iPhone 照片（一张实拍、一张截图）经微信送达后，**文件里连一个 EXIF 段都没有**。
于是内参必然退化为模型预测 —— 横向米制尺度不可信（实测 118 倍误差）。
本轮把「用户手填等效焦距」这条兜底补上了：原来要求填 4 个**像素**焦距，而手机里查不到，
等于兜底是空的。

---

## 一、输入

| 文件 | 大小 | 尺寸 | 说明 |
|---|---|---|---|
| `uploads/real_photo_probe/iphone_real.jpg` | 217,924 B | 1280×1707 | iPhone 实拍，经微信送达 |
| `uploads/real_photo_probe/iphone_screenshot.jpg` | 301,351 B | 1179×2556 | iPhone 截图，经微信送达 |

两张都**不是相册原图**：① 是 4:3 竖拍（3024×4032）等比缩小 2.36 倍的结果；
② 的 1179×2556 正是 iPhone 15/16 Pro 的屏幕分辨率。

⚠ 图片放在 `uploads/` 下，该目录已被 `.gitignore` 第 48 行覆盖，**不会被提交**。

## 二、证据链（三层，每层都带对照）

### 第 1 层 · PIL 读 EXIF → 两张都是 0 标签

```
python scripts/inspect_exif.py --image uploads/real_photo_probe/iphone_real.jpg \
                                      uploads/real_photo_probe/iphone_screenshot.jpg --raw
→ 两张都是「EXIF 顶层标签数 0」/「<EXIF 为空>」
```

⚠ **这一层不足以下结论**：`<EXIF 为空>` 对「真的没有」和「有但读不到」是同一个输出。

### 第 2 层 · 段级扫描 → APP1 段根本不存在

```
python scripts/inspect_jpeg_segments.py uploads/real_photo_probe/iphone_real.jpg \
       uploads/real_photo_probe/iphone_screenshot.jpg \
       .cache/exif_fixture/rgb_exif.jpg .cache/cross_source/sony.jpg .cache/cross_source/panasonic.jpg
```

| 文件 | 段数 | Exif 段 |
|---|---|---|
| `iphone_real.jpg` | 10 | **0** |
| `iphone_screenshot.jpg` | 10 | **0** |
| `rgb_exif.jpg`（对照） | 10 | 1（APP1 len=136） |
| `sony.jpg`（对照） | 9 | 1（APP1 len=39965） |
| `panasonic.jpg`（对照） | 9 | 1（APP1 len=29182） |

**对照 3/3 全部命中** ⟹ 扫描器有效 ⟹ 「没扫到」是有信息的 ⟹ 属于「**真的不在文件里**」，
而不是「在、但 PIL 读不到」。

两张图的段序列**逐项同构**：

```
JFIF(16) → APP2 = ICC_PROFILE(564) → DQT(67,67) → SOF0(17) → DHT(31,181,31,181) → SOS
```

`DHT` 长度 31/181 是 libjpeg 的**标准** Huffman 表（相机直出一般用按图优化的表），
再叠加 564 B 的小 ICC（重编码器写入的 sRGB）与 SOF0 里的 4:2:0 采样
⟹ 两张图都过了**同一套默认参数的重编码器**。

### 第 3 层 · 用**产品自己的**前端逻辑跑 → 它会如实告诉用户

```
node scripts/check_exif_parity.js uploads/real_photo_probe/iphone_real.jpg \
     uploads/real_photo_probe/iphone_screenshot.jpg .cache/exif_fixture/rgb_exif.jpg
```

| 文件 | 前端读到的 tag 数 | `UP.exif35` | 界面显示 |
|---|---|---|---|
| `iphone_real.jpg` | 0 | `false` | 没有 EXIF 等效焦距 →「会被标成『尺度未标定』」 |
| `iphone_screenshot.jpg` | 0 | `false` | 同上 |
| `rgb_exif.jpg` | 6（含 41989 = 0xA405） | `true` | 检测到 EXIF 等效焦距 |
| `sony.jpg` | 49（含 41989） | `true` | 同上 |

⟹ **产品的判断逻辑是正确的**（能识别对照、如实报警目标）。
问题不在产品逻辑，而在产品所面对的现实。

## 三、能确定什么 / 不能确定什么

**能确定**

1. 这两份**文件本身**没有任何 EXIF —— 三处独立检查（PIL / 段级 / 前端源码）结论一致。
2. 它们都经过重编码与缩放，**不是相册原图**。
3. 因此 `--intrinsics exif` 在这两张图上**必然**退化到模型预测。

**不能确定**

- **具体是哪一环剥掉的。** 链路是「iPhone → 微信 → 我」。iPhone 直出必然带 APP1，
  所以必然在某处被剥；但「微信剥的」与「用户导出方式剥的」这次分不开。
  ⚠ 要定位需要**同一张照片的非微信途径原图**（直接拷到 `uploads/`），一对比即可确定。

## 四、这暴露的产品缺口（本轮已补）

**缺口**：前端有「手动 —— 我填 fx,fy,cx,cy」，后端只认「文件 / `exif` / `auto` / 恰好 4 个数」。
而**手机里能查到的只有「等效焦距 24 mm」，查不到像素焦距** ⟹ 对目标用户，这条兜底是空的。

**补法**（硬约束：**公式只有一份**）

| 位置 | 改动 |
|---|---|
| `vision/exif.py` | `read_exif_intrinsics(..., focal_35mm_mm=)` 新入参，复用**同一段**换算；`source` 记 `user:35mm` |
| `scripts/build_scene.py` | `load_known_intrinsics` 认 `--intrinsics f35:24` |
| `demo/index.html` + `app.js` | 内参下拉新增「等效焦距」，提交 `f35:<mm>`；**换算交给后端，前端不自己算** |
| `scripts/inspect_exif.py` | 新增 `--assume-f35 24`，上传前先预览会得到什么 K |

**验证**（`tests/test_build_scene_intrinsics.py` 17 条 + `tests/test_vision_exif.py` +12 条）

- **逐位相等**：同一个 f_35，从文件读 vs 调用方给 ⟹ `np.array_equal` 通过。
- **坏输入必须响**：`f35:` / `abc` / `0` / `-5` / `nan` / `inf` ⟹ 全部 `SystemExit` / `ValueError`，
  **不许**静默退化成「这张图没有 EXIF」（那会让填错的数字伪装成「没有内参」）。
- **真实微信图上**：`exif` → 降级；`f35:24` → `fx=1138.0`（= 24/36 × 1707 ✓）。

## 五、遗留（明确未解决）

1. ⚠ **截图 / 裁剪过的图，「等效焦距」这条公式的前提不成立。**
   `f_px = f_35/36 × 当前长边` 隐含「长边 = 36 mm 画幅长边」。
   截图 1179×2556（1:2.17）不是相机比例，对它填 24 mm 得 HFoV **38.2°** ——
   落在 30–110° 可信窗口内，**`check_fov` 不会拦**。缺一条「宽高比像不像相机照片」的检查。
2. **两套 EXIF 解析存在分叉风险**（后端 `vision/exif.py` / 前端 `app.js::readExifTags`）。
   本轮在这 4 张样本上**未发现分叉**（`scripts/check_exif_parity.js`），但样本量小，
   只说明这条路上没出问题，不等于不会分叉。
3. 微信传图的**最优解是让用户重发原图**（保住 EXIF），手填是次优 —— 产品提示里已写。
   手填的主要误差是**镜头倍率报错**，而 `check_fov` 拦不住它
   （窗口 30–110° 等价于 f_35 ∈ [12.6, 67.2] mm，手机各档几乎都落在里面）。

## 六、复现

```bash
# ① EXIF 能不能用
python scripts/inspect_exif.py --image <图> --raw
# ② 若 ① 说读不到：EXIF 到底在不在文件里（必须带阳性对照）
python scripts/inspect_jpeg_segments.py <图> .cache/cross_source/sony.jpg
# ③ 前端会告诉用户什么 / 会不会与后端分叉
node scripts/check_exif_parity.js <图>
# ④ 没有 EXIF 时，填等效焦距会得到什么 K
python scripts/inspect_exif.py --image <图> --assume-f35 24
```
