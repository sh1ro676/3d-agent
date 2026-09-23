/* 用**产品前端自己的** `readExifTags` 判一批图 —— 用来核对前后端两套 EXIF 解析会不会分叉。
 *
 * ## 为什么这是个真实风险
 *
 * 产品里有**两套** EXIF 解析：
 *
 *     后端 `vision/exif.py`       完整解析，决定内参 K 从哪来（fx 对不对）
 *     前端 `demo/app.js`          只判「有没有 0xA405」，决定界面上告诉用户什么
 *
 * 两套实现一旦分叉，界面写着「检测到 EXIF 等效焦距」而建图时却退化成模型预测
 * （或反过来），用户**没有任何办法分辨** —— 而这两条路的横向误差差 118 倍。
 *
 * ## 做法：按源码抽取，不手抄
 *
 * 手抄一份函数来测，测的就不是用户会运行的代码了。所以这里读 `demo/app.js`，
 * 正则定位 `function readExifTags(buf) {` 并用括号配对取出整段源码求值。
 * 源码里找不到它就**报错退出**（产品改名了，本脚本作废，不许给旧结论）。
 *
 * ## 判据
 *
 * 判据是「两边对同一张图给出**同一个结论**」，不是「前端跑得通」。所以必须
 * 同时跑后端，然后人工/脚本比对：
 *
 *     node  scripts/check_exif_parity.js a.jpg b.jpg
 *     python scripts/inspect_exif.py --image a.jpg --image b.jpg --raw
 *
 * 2026-09-23 实测：`.cache/exif_fixture/rgb_exif.jpg`（6 tag，含 41989）
 * 与 `.cache/cross_source/sony.jpg`（49 tag，含 41989）两边都判「有」；
 * 两张微信转存的 iPhone 照片（0 tag）两边都判「没有」⟹ **未发现分叉**。
 * 样本量小，只说明这两条路上没出问题，不等于不会分叉。
 *
 * ## 用法
 *
 *     node scripts/check_exif_parity.js <图...>
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(ROOT, 'demo/app.js'), 'utf8');

const start = src.indexOf('function readExifTags(buf) {');
if (start < 0) {
  console.error('[FAIL] 找不到 readExifTags —— 产品源码已变，本脚本作废，不要相信它的输出');
  process.exit(3);
}
let i = src.indexOf('{', start), depth = 0, end = -1;
for (let k = i; k < src.length; k++) {
  if (src[k] === '{') depth++;
  else if (src[k] === '}') { depth--; if (depth === 0) { end = k + 1; break; } }
}
const fnSrc = src.slice(start, end);
const startLine = src.slice(0, start).split('\n').length;
console.log('抽取自 demo/app.js:' + startLine + '（' + fnSrc.length + ' 字符）\n');

const readExifTags = new Function('return (' + fnSrc + ')')();

const files = process.argv.slice(2);
if (!files.length) {
  console.error('用法: node scripts/check_exif_parity.js <图...>');
  process.exit(2);
}
let ok = 0;
for (const f of files) {
  const b = fs.readFileSync(f);
  const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  let tags;
  try {
    tags = readExifTags(ab);
  } catch (e) {
    console.log('=== ' + f);
    console.log('    [异常] ' + e.message + '  ← 前端会 catch 并按「无 EXIF」处理\n');
    continue;
  }
  const has = tags.has(0xA405);
  if (has) ok++;
  console.log('=== ' + f);
  console.log('    整个 JPEG 里读到的 EXIF tag 数 : ' + tags.size);
  console.log('    tag 号                          : ' + JSON.stringify([...tags]));
  console.log('    UP.exif35 (= tags.has(0xA405))  : ' + has);
  console.log('    预览行显示                      : ' +
    (has ? '检测到 EXIF 等效焦距' : '没有 EXIF 等效焦距'));
  console.log('    状态行显示                      : ' + (has
    ? '有 EXIF 焦距 → 内参可从 EXIF 换算（选出 auto 即可）'
    : '无 EXIF 焦距（微信/社交软件转发的图必丢）→ 会被标成「尺度未标定」。'
      + '两条出路：改用相册原图，或在下方内参选「等效焦距」手填一次（1× 主摄通常 24 mm）'));
  console.log();
}
console.log('小结：' + ok + '/' + files.length + ' 张会被前端判为「有 EXIF 等效焦距」');
console.log('对照后端：python scripts/inspect_exif.py --image <同一批图> --raw');
process.exit(0);
