/* ============================================================================
 * demo/app.js —— 3D 空间智能 Agent 演示台（前端逻辑）
 *
 * 三件事必须写在最前面，因为它们决定了这个界面**长什么样**：
 *
 * 1. **两种模式**：`回放` 读已落盘的 runs.json（零成本、零联网）；
 *    `现场提问` 打 /api/ask 真跑一次 AgentLoop。它们共用同一套渲染。
 *    探不到后端就自动锁进回放模式 —— 拔网线也能演，只是不能现场提问。
 *
 * 2. **诚实性优先于好看**。这份场景图 `scale_calibrated=false`，
 *    所以每个盒子都有**正确的位置**和**不可采信的尺寸**（沙发 extent 6.7 m）；
 *    `up_axis_reliable=false`，所以**不画地面网格** —— 画了就等于宣称重力已知。
 *    界面上那两条黄色横幅不是装饰，是这次演示最该被看见的东西。
 *
 * 3. **一张图只被"看"一次**。所有空间数字来自离线建好的场景图，
 *    在线只做「自然语言 → 工具调用」的映射。所以界面上任何米数都必须
 *    能追到某个工具返回值 —— 这正是右侧"轨迹"面板存在的理由。
 * ==========================================================================*/

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

/* ---------------------------------------------------------------- 小工具 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const fmt = (v, d = 3) => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
    return v.toFixed(d).replace(/0+$/, '').replace(/\.$/, '');
  }
  return String(v);
};

const jstr = (v, max = 400) => {
  let s;
  try { s = JSON.stringify(v); } catch { s = String(v); }
  if (s === undefined) s = String(v);
  return s.length > max ? s.slice(0, max) + ' …' : s;
};

/** 状态 → 颜色类。五个 status 是 AgentRun 的契约取值，不是随手起的名。 */
const STATUS_CLASS = {
  ok: 't-ok', abstained: 't-warn', static_failed: 't-err',
  exec_failed: 't-err', llm_error: 't-err',
};
const VERDICT_CLASS = {
  supported: 't-ok', weak: 't-warn', unsupported: 't-err', abstained: 't-dim',
};
const VERDICT_LABEL = {
  supported: '证据支持', weak: '证据偏弱', unsupported: '证据不成立', abstained: '弃答',
};

/** 同 label 同色。用**排序后的标签表**建索引，保证每次刷新配色一致。 */
const PALETTE = ['#4cc9f0', '#f72585', '#ffbe0b', '#06d6a0', '#b5179e',
  '#3a86ff', '#fb5607', '#8ac926', '#ef476f', '#9d4edd'];
let labelColor = new Map();

function buildPalette(nodes) {
  labelColor = new Map();
  const labels = [...new Set(nodes.map((n) => n.label))].sort();
  labels.forEach((l, i) => labelColor.set(l, PALETTE[i % PALETTE.length]));
}
const colorFor = (label) => labelColor.get(label) || '#8b98a8';

/* ------------------------------------------------------------------ 状态 */

const S = {
  index: null,
  runs: null,
  sceneId: null,
  scene: null,
  report: null,
  nodes: [],
  selected: new Set(),
  backend: { online: false, ready: false, info: null, toolsVersion: null },
  mode: 'replay',
  replay: { batchIdx: 0, runIdx: -1 },
  lastLive: null,
  busy: false,
};

/** 演示时可以直接用 URL 定格：`?scene=living_room_gt&select=sofa_1&mode=live`。
 *  `&layout=focus` 定布局、`&upload=1` 直接打开上传面板 —— 演示时少点几下，
 *  也少一次"当着评委的面找按钮"。 */
const QS = new URLSearchParams(location.search);
const URL_SCENE = QS.get('scene');
const URL_SELECT = QS.get('select');
const URL_MODE = QS.get('mode');
const URL_LAYOUT = QS.get('layout');
const URL_UPLOAD = QS.get('upload');

/* ================================================================== 3D 视图 */

const V = {
  ok: false, renderer: null, scene: null, camera: null, controls: null,
  hull: null, links: null, meshes: new Map(), labels: new Map(),
  linkLabels: [],
  raycaster: new THREE.Raycaster(), ptr: new THREE.Vector2(),
  box: new THREE.Box3(),
};

/**
 * 坐标系换算 —— 这一步错了整个 3D 视图就是镜像的，而且**看起来很正常**。
 *
 * 场景图里 `frame: "camera"`，即 **x 右 / y 下 / z 前**（图像坐标系的自然延伸）。
 * three.js 是 **x 右 / y 上 / z 朝观察者**。所以：
 *
 *     three.x =  x_data
 *     three.y = -y_data        ← 翻转：图像里"往下"在世界里是"往下"
 *     three.z = -z_data        ← 翻转：让相机前方落在 three 的 -z（默认视向）
 *
 * 两次取反 = 绕 x 轴 180°，仍是右手系，法线不会翻。
 */
const to3 = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);

function init3D() {
  const host = $('#viewport');
  if (!host) return;
  try {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0b0e14, 1);
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 2000);
    camera.position.set(4, 3, 9);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 0.9;
    controls.screenSpacePanning = true;

    // 光照只对"有明暗的面"有用；这里盒子用半透明纯色，所以只需要环境光
    // 让标准材质的默认灰不要发黑。改回 Basic 也行，但标准材质在
    // 半透明叠加时层次更清楚。
    scene.add(new THREE.AmbientLight(0xffffff, 2.2));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(6, 10, 6);
    scene.add(dir);

    // 原点处一个小十字 —— 提示这是**相机坐标系的原点**（光心），不是地面。
    const axes = new THREE.AxesHelper(0.6);
    axes.material.opacity = 0.55;
    axes.material.transparent = true;
    scene.add(axes);

    const hull = new THREE.Group();
    const links = new THREE.Group();
    scene.add(hull, links);

    V.renderer = renderer; V.scene = scene; V.camera = camera;
    V.controls = controls; V.hull = hull; V.links = links; V.ok = true;

    new ResizeObserver(resize3D).observe(host);
    resize3D();

    // 点击拾取：射线只打盒子 mesh，不打标签（标签是 HTML，天然不参与）
    renderer.domElement.addEventListener('pointerdown', (e) => {
      const r = renderer.domElement.getBoundingClientRect();
      V.ptr.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      V.ptr.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      V.raycaster.setFromCamera(V.ptr, V.camera);
      const hits = V.raycaster.intersectObjects([...V.meshes.values()], false);
      if (!hits.length) return;
      toggleSelect(hits[0].object.userData.nodeId, e.shiftKey || e.ctrlKey || e.metaKey);
    });

    const loop = () => {
      requestAnimationFrame(loop);
      V.controls.update();
      V.renderer.render(V.scene, V.camera);
      updateLabels();
    };
    loop();
  } catch (exc) {
    V.ok = false;
    $('#viewport').innerHTML =
      '<div class="empty">WebGL 初始化失败：' + esc(exc.message) + '<br>换一个支持 WebGL 的浏览器即可。</div>';
  }
}

function resize3D() {
  if (!V.ok) return;
  const host = $('#viewport');
  const w = Math.max(host.clientWidth, 10);
  const h = Math.max(host.clientHeight, 10);
  V.renderer.setSize(w, h, false);
  V.camera.aspect = w / h;
  V.camera.updateProjectionMatrix();
}

/** 清空 3D 内容（换场景时调用，否则上一个场景的盒子会残留）。
 *
 * ⚠ 清理必须遍历 `hull.children` 而不是 `V.meshes` —— 每个物体往组里放了
 * **两个**对象（半透明面 + 棱线），只按 meshes 清会留下棱线幽灵。
 */
function clear3D() {
  if (!V.ok) return;
  while (V.hull.children.length) {
    const c = V.hull.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  V.meshes.clear();
  while (V.links.children.length) {
    const c = V.links.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  $('#labels').innerHTML = '';
  V.labels.clear();
  V.linkLabels = [];
}

function build3D() {
  if (!V.ok) return;
  clear3D();
  const nodes = S.nodes;
  if (!nodes.length) return;

  const gBox = new THREE.BoxGeometry(1, 1, 1);
  const eBox = new THREE.EdgesGeometry(gBox);

  for (const n of nodes) {
    const e = (n.extent_3d || [1, 1, 1]).map((v) => Math.max(Math.abs(v), 1e-3));
    const c = to3(n.centroid_3d || [0, 0, 0]);
    const col = new THREE.Color(colorFor(n.label));

    const mesh = new THREE.Mesh(gBox.clone(), new THREE.MeshStandardMaterial({
      color: col, transparent: true, opacity: 0.1, depthWrite: false,
      roughness: 0.85, metalness: 0,
    }));
    mesh.scale.set(e[0], e[1], e[2]);
    mesh.position.copy(c);
    mesh.userData.nodeId = n.id;
    V.hull.add(mesh);

    const edge = new THREE.LineSegments(eBox.clone(), new THREE.LineBasicMaterial({
      color: col, transparent: true, opacity: 0.75,
    }));
    edge.scale.copy(mesh.scale);
    edge.position.copy(c);
    V.hull.add(edge);

    // 把面与棱当作一个可拾取单元 —— 只存面，棱不参与 raycast（棱太细打不中）
    mesh.userData.edge = edge;
    V.meshes.set(n.id, mesh);

    // HTML 标签（比 Sprite 清晰，且能直接吃 CSS）
    const el = document.createElement('div');
    el.className = 'lbl';
    el.textContent = n.id;
    el.style.borderColor = 'rgba(0,0,0,0)';
    el.style.color = colorFor(n.label);
    $('#labels').appendChild(el);
    V.labels.set(n.id, el);
  }
  eBox.dispose(); gBox.dispose();
  fit3D();
  refreshSelectionVisuals();
}

/** 自动取景：把相机拉到能看全整个包围盒的位置。
 *
 * ⚠ **刻意不把光心（原点）纳入包围盒**。物体全在相机前方 z≈2.8~3.7，
 * 一旦把原点也算进去，包围盒凭空大一倍，整个场景会被推到画面中央一小块 ——
 * 「东西离得挺远」这件事反而看不出来了。原点在哪不影响读图。
 */
function fit3D() {
  if (!V.ok || !S.nodes.length) return;
  const box = new THREE.Box3();
  for (const n of S.nodes) {
    const e = (n.extent_3d || [0, 0, 0]).map((v) => Math.abs(v) / 2);
    const c = to3(n.centroid_3d || [0, 0, 0]);
    box.expandByPoint(new THREE.Vector3(c.x - e[0], c.y - e[1], c.z - e[2]));
    box.expandByPoint(new THREE.Vector3(c.x + e[0], c.y + e[1], c.z + e[2]));
  }
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.35);

  // 第一近似：按包围球算一个"一定装得下"的距离（垂直与水平两个方向都要顾，
  // 窄面板 aspect<1 时水平方向才是限制项）。
  const fovV = (V.camera.fov * Math.PI) / 180;
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * Math.max(V.camera.aspect, 0.1));
  let dist = radius / Math.sin(Math.min(fovV, fovH) / 2) * 0.96;

  const dirv = new THREE.Vector3(0.5, 0.38, 1).normalize();

  // ★ 包围球只保证"装得下"，不保证"填得满"。一组又扁又散的物体，
  //   球半径由最远两点决定，但它们在屏幕上可能只占中间窄窄一条 ——
  //   于是画面一半是空白，看起来"物体缩在角落里"。
  //   所以再按**投影后的真实占比**收敛：把包围盒 8 个角投影到 NDC，
  //   取最外的那个，用它反推该退多远/该推多近。近大远小的透视下
  //   屏幕尺寸近似 ∝ 1/dist，所以一次缩放就够了，做两次是留余量。
  for (let iter = 0; iter < 2; iter++) {
    V.camera.position.copy(center).addScaledVector(dirv, dist);
    V.camera.lookAt(center);
    V.camera.updateMatrixWorld(true);
    let worst = 0;
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(
        (i & 1) ? box.max.x : box.min.x,
        (i & 2) ? box.max.y : box.min.y,
        (i & 4) ? box.max.z : box.min.z,
      ).project(V.camera);
      if (p.z > 1) { worst = Math.max(worst, 1.8); continue; }  // 有角跑到相机背后
      worst = Math.max(worst, Math.abs(p.x), Math.abs(p.y));
    }
    if (!Number.isFinite(worst) || worst <= 1e-6) break;
    const want = worst / 0.9;                 // 目标：最外角落在屏幕 90% 半径处
    if (want > 0.985 && want < 1.015) break;  // 已经贴到目标，不必再动
    dist *= want;
  }

  V.camera.position.copy(center).addScaledVector(dirv, dist);
  V.camera.near = Math.max(dist / 5000, 0.01);
  V.camera.far = dist * 50;
  V.camera.updateProjectionMatrix();
  V.controls.target.copy(center);
  V.controls.update();
}

const _p = new THREE.Vector3();
function _place(el, v3, w, h) {
  _p.copy(v3).project(V.camera);
  if (_p.z > 1) { el.style.opacity = '0'; return; }
  el.style.opacity = '1';
  el.style.left = ((_p.x * 0.5 + 0.5) * w).toFixed(1) + 'px';
  el.style.top = ((-_p.y * 0.5 + 0.5) * h).toFixed(1) + 'px';
}

function updateLabels() {
  if (!V.ok) return;
  const host = $('#viewport');
  const w = host.clientWidth, h = host.clientHeight;
  const show = $('#chk-labels')?.checked !== false;

  // 先全部算出屏幕位置，再统一做避让。
  // 边算边显示的话，"谁盖住谁"就取决于遍历顺序 —— 而遍历顺序是数组下标，
  // 不是重要程度。实测 picture_3 / picture_4 / chair_3 三个标签会叠成一团。
  const cand = [];
  for (const n of S.nodes) {
    const el = V.labels.get(n.id);
    if (!el) continue;
    if (!show) { el.style.opacity = '0'; continue; }
    _p.copy(to3(n.centroid_3d)).project(V.camera);
    if (_p.z > 1) { el.style.opacity = '0'; continue; }
    cand.push({
      el,
      x: (_p.x * 0.5 + 0.5) * w,
      y: (-_p.y * 0.5 + 0.5) * h,
      w: el.offsetWidth || 58,
      h: el.offsetHeight || 16,
      // 优先级：选中的绝对优先，其余按点数（点数≈它在这张图里占多大）
      pri: (S.selected.has(n.id) ? 1e9 : 0) + (n.n_points || 0),
    });
  }
  cand.sort((a, b) => b.pri - a.pri);
  const kept = [];
  for (const it of cand) {
    const clash = kept.some((k) =>
      Math.abs(k.x - it.x) < (k.w + it.w) / 2 + 3 &&
      Math.abs(k.y - it.y) < (k.h + it.h) / 2 + 2);
    if (clash) { it.el.style.opacity = '0'; continue; }   // 只是不显示，不是不存在
    it.el.style.opacity = '1';
    it.el.style.left = it.x.toFixed(1) + 'px';
    it.el.style.top = it.y.toFixed(1) + 'px';
    kept.push(it);
  }

  const showLinks = show && $('#chk-links')?.checked;
  for (const it of V.linkLabels) {
    if (!showLinks) { it.el.style.opacity = '0'; continue; }
    _place(it.el, it.mid, w, h);
  }
}

/** 选中态的可视化：面变亮、棱变粗（用 linewidth 在多数平台无效，故改用双线叠色）。 */
function refreshSelectionVisuals() {
  if (!V.ok) return;
  for (const [id, mesh] of V.meshes) {
    const on = S.selected.has(id);
    mesh.material.opacity = on ? 0.34 : 0.1;
    mesh.material.emissive = new THREE.Color(on ? colorFor(S.nodes.find((n) => n.id === id)?.label) : 0x000000);
    mesh.material.emissiveIntensity = on ? 0.6 : 0;
    mesh.userData.edge.material.opacity = on ? 1 : 0.75;
    const el = V.labels.get(id);
    if (el) el.classList.toggle('sel', on);
  }
  rebuildLinks();
  updateChips();
  drawImage();
}

/** 关系线：只画**选中节点**到其他节点的距离连线，避免 133 条边糊成一团。
 *
 * 距离数字用 HTML 标签而不是 3D 文字/精灵：一是本项目没有字体资源，
 * 二是 HTML 标签在任意缩放下都清晰。代价是每帧要自己投影一次（见 updateLabels）。
 */
function rebuildLinks() {
  if (!V.ok) return;
  while (V.links.children.length) {
    const c = V.links.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  for (const it of V.linkLabels) it.el.remove();
  V.linkLabels = [];
  if (!$('#chk-links')?.checked || S.selected.size === 0) return;

  for (const id of S.selected) {
    const a = S.nodes.find((n) => n.id === id);
    if (!a) continue;
    const p1 = to3(a.centroid_3d);
    for (const n of S.nodes) {
      if (n.id === id) continue;
      const p2 = to3(n.centroid_3d);
      const d = p1.distanceTo(p2);
      V.links.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([p1, p2]),
        new THREE.LineBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.3 }),
      ));

      const el = document.createElement('div');
      el.className = 'lbl';
      el.textContent = d.toFixed(2) + ' m';
      el.style.color = '#4cc9f0';
      el.style.fontSize = '9.5px';
      $('#labels').appendChild(el);
      V.linkLabels.push({ el, mid: p1.clone().add(p2).multiplyScalar(0.5) });
    }
  }
}

/* ================================================================ 图像视图 */

const I = { canvas: null, ctx: null, base: null, W: 640, H: 480, masks: new Map(), tmp: null };

function initImage() {
  I.canvas = $('#imgcanvas');
  I.ctx = I.canvas.getContext('2d', { willReadFrequently: true });

  I.canvas.addEventListener('pointerdown', (e) => {
    const r = I.canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (I.canvas.width / r.width);
    const y = (e.clientY - r.top) * (I.canvas.height / r.height);
    // 从后往前找：后画的在上层，命中就选它
    for (let i = S.nodes.length - 1; i >= 0; i--) {
      const b = S.nodes[i].bbox_2d;
      if (!b) continue;
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
        toggleSelect(S.nodes[i].id, e.shiftKey || e.ctrlKey || e.metaKey);
        return;
      }
    }
    if (!e.shiftKey && !e.ctrlKey && !e.metaKey) { S.selected.clear(); refreshSelectionVisuals(); }
  });
}

/** 把 PNG 掩码读成 0/1 位图。1-bit 灰度图，阈值 127 即可。 */
async function loadMaskBin(url) {
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  const d = cx.getImageData(0, 0, c.width, c.height).data;
  const bin = new Uint8Array(c.width * c.height);
  for (let i = 0, n = bin.length; i < n; i++) bin[i] = d[i * 4] > 127 ? 1 : 0;
  return bin;
}

async function loadImageForScene() {
  I.masks.clear();
  const url = S.scene.image_url || (S.index.scenes.find((s) => s.scene_id === S.sceneId) || {}).image_url;
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  I.base = img;
  const hw = S.scene.image_hw || [img.naturalWidth, img.naturalHeight];
  I.W = hw[0]; I.H = hw[1];
  I.canvas.width = I.W; I.canvas.height = I.H;
  I.tmp = document.createElement('canvas');
  I.tmp.width = I.W; I.tmp.height = I.H;

  // 这行字从"图下方的一块说明"变成了页脚的一行 —— 它重要，但不该在图下面
  // 再占三行高度（图片居中放大后，那三行会把它挤小）。完整说明放 title。
  // 内容里只有本地数字常量，没有外部数据，所以直接拼 innerHTML 是安全的。
  const note = $('#img-note');
  note.innerHTML = '原图 ' + I.W + '×' + I.H + ' px ｜ 掩码 = SAM2 在检测框内收紧的轮廓；' +
    '质心取<b>掩码内点云的中位数</b>，不是框内均值（差均值 83 mm，而容差只有 50 mm）。';
  note.title = '原图 ' + I.W + '×' + I.H + ' px\n' +
    '检测框与掩码均为像素坐标，直接叠在原图上，不做任何重采样。\n' +
    '掩码是 SAM2 在检测框内收紧出来的物体轮廓。\n' +
    '质心取掩码内点云的中位数，不是检测框内的中位数 —— ' +
    '两者差均值 83 mm、最大 208 mm，而题目容差只有 50 mm。';

  // 掩码逐张加载；缺一张不该让整屏挂掉
  const tasks = S.nodes.filter((n) => n.mask_url).map(async (n) => {
    try { I.masks.set(n.id, await loadMaskBin(n.mask_url)); }
    catch { /* 单张缺失 → 该物体没有掩码高亮，其余照常 */ }
  });
  await Promise.all(tasks);
  drawImage();
}

function drawImage() {
  const ctx = I.ctx;
  if (!ctx) return;
  ctx.clearRect(0, 0, I.W, I.H);
  if (I.base) ctx.drawImage(I.base, 0, 0, I.W, I.H);

  // --- 掩码叠色（只给选中的物体上色：全画会糊成一团，也看不出选中了谁）
  if ($('#chk-mask')?.checked && S.selected.size) {
    const tctx = I.tmp.getContext('2d');
    const im = tctx.createImageData(I.W, I.H);
    const data = im.data;
    const claimed = new Uint8Array(I.W * I.H);
    for (const id of S.selected) {
      const bin = I.masks.get(id);
      const node = S.nodes.find((n) => n.id === id);
      if (!bin || !node) continue;
      const c = new THREE.Color(colorFor(node.label));
      const r = Math.round(c.r * 255), g = Math.round(c.g * 255), b = Math.round(c.b * 255);
      for (let i = 0, n = bin.length; i < n; i++) {
        if (!bin[i] || claimed[i]) continue;
        claimed[i] = 1;
        const o = i * 4;
        data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
      }
    }
    tctx.putImageData(im, 0, 0);
    ctx.save();
    ctx.globalAlpha = 0.42;
    ctx.drawImage(I.tmp, 0, 0);
    ctx.restore();
  }

  // --- 检测框
  const showBox = $('#chk-box')?.checked !== false;
  ctx.lineWidth = 2;
  ctx.font = '600 13px ui-monospace, Menlo, Consolas, monospace';
  ctx.textBaseline = 'bottom';
  for (const n of S.nodes) {
    const b = n.bbox_2d;
    if (!b) continue;
    const on = S.selected.has(n.id);
    const col = colorFor(n.label);
    if (!showBox && !on) continue;
    ctx.strokeStyle = col;
    ctx.globalAlpha = on ? 1 : 0.5;
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    if (on) {
      const txt = n.id;
      const w = ctx.measureText(txt).width;
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = '#0b0e14';
      ctx.fillRect(b[0], b[1] - 18, w + 10, 18);
      ctx.globalAlpha = 1;
      ctx.fillStyle = col;
      ctx.fillText(txt, b[0] + 5, b[1] - 4);
    }
  }
  ctx.globalAlpha = 1;

  // --- 空选中时给一句提示：否则首屏看起来像"掩码功能坏了"，
  //     而实际上掩码是**按需上色**的（全画会糊成一团，也看不出选中了谁）。
  if (!S.selected.size) {
    const t = '点击画面中的物体（或下方色块）→ 高亮它的掩码与三维盒';
    ctx.font = '600 13px -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
    const w = ctx.measureText(t).width + 22;
    const x = Math.max((I.W - w) / 2, 8);
    ctx.fillStyle = 'rgba(11,14,20,.8)';
    ctx.fillRect(x, I.H - 40, w, 28);
    ctx.strokeStyle = 'rgba(76,201,240,.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + .5, I.H - 39.5, w - 1, 27);
    ctx.fillStyle = '#c6d3e0';
    ctx.textBaseline = 'middle';
    ctx.fillText(t, x + 11, I.H - 25);
  }
}

/* ================================================================== 选择 */

function toggleSelect(id, additive) {
  if (!id) return;
  if (additive) {
    if (S.selected.has(id)) S.selected.delete(id); else S.selected.add(id);
  } else {
    // 单选语义：已经只选了它 → 再点一次取消（方便清空）
    if (S.selected.size === 1 && S.selected.has(id)) S.selected.clear();
    else { S.selected.clear(); S.selected.add(id); }
  }
  refreshSelectionVisuals();
}

function updateChips() {
  const host = $('#obj-chips');
  if (!host) return;
  host.innerHTML = S.nodes.map((n) => {
    const on = S.selected.has(n.id) ? ' sel' : '';
    const nm = n.n_points ? ' title="点数 ' + n.n_points + '"' : '';
    return '<button class="chip' + on + '" data-id="' + esc(n.id) + '"' + nm + '>' +
      '<span class="sw" style="background:' + colorFor(n.label) + '"></span>' +
      esc(n.id) + '<span class="n">' + esc(fmt(n.score, 3)) + '</span></button>';
  }).join('');
  $$('.chip', host).forEach((el) => {
    el.addEventListener('click', (e) => toggleSelect(el.dataset.id, e.shiftKey || e.ctrlKey || e.metaKey));
  });
  const n = S.selected.size;
  $('#btn-clear').disabled = n === 0;
  $('#btn-cf').disabled = n === 0 || !S.backend.online;
  $('#btn-cf').title = S.backend.online
    ? '对选中的 ' + n + ' 个物体做反事实推理（纯图操作，不花 API 钱）'
    : '反事实需要后端在线（scripts/serve_demo.py）';
}

/* ================================================================== 报告 */

/**
 * 一条**可折叠**的诚实性横幅：默认只占一行。
 *
 * 为什么值得改：两条横幅全展开时顶部要吃掉 70 多 px，而且两条宽度差很多，
 * 视觉上是压在上面的两块砖。但折叠**不等于**淡化 —— 摘要行里必须写明
 * "这件事有多严重"（比如"相对位置可信、绝对尺寸不可采信"），
 * 否则它就从"诚实"退化成"眼不见为净"，那正好反了这个界面的初衷。
 */
function banner({ title, summary, detail }) {
  return '<div class="banner"><span class="ic">!</span><span>' +
    '<b>' + title + '</b> —— <span class="sm">' + summary + '</span> ' +
    '<button class="more">展开</button>' +
    '<div class="det">' + detail + '</div></span></div>';
}

function setBanners() {
  const host = $('#banners');
  const q = (S.report && S.report.quality) || S.scene || {};
  const meta = S.scene?.build_meta || {};
  const out = [];

  const calibrated = q.scale_calibrated ?? meta.scale_calibrated ?? false;
  if (!calibrated) {
    out.push(banner({
      title: '尺度未标定',
      summary: '相对位置可信，<b>绝对尺寸不可采信</b> —— 盒子比例不能当尺寸读',
      detail: '全部米制数字共享一个未知比例因子（scale_factor = 1.0）。' +
        '本场景 <code>sofa_1</code> 的输出尺寸是 6.7 m —— 这不是"沙发真有六米七"，' +
        '而是尺度没定。节点之间的<b>相对</b>位置是对的，三维距离就是这么算出来的。',
    }));
  }

  const upOk = q.up_axis_reliable ?? meta.up_axis_reliable ?? false;
  const tilt = q.up_axis_tilt_deg ?? meta.up_axis_tilt_deg;
  if (!upOk) {
    out.push(banner({
      title: '重力方向不可靠',
      summary: 'tilt ' + esc(fmt(tilt, 2)) + '° —— 所以本界面<b>不画地面网格</b>',
      detail: (meta.up_axis_reason ? '原因是 <code>' + esc(meta.up_axis_reason) + '</code>。' : '') +
        '没有地面线不是省事，是"画了就等于宣称重力已知"，而它其实只是估出来的。',
    }));
  }
  host.innerHTML = out.join('');
  $$('.banner', host).forEach((el) => {
    const btn = $('.more', el);
    btn.addEventListener('click', () => {
      el.classList.toggle('open');
      btn.textContent = el.classList.contains('open') ? '收起' : '展开';
    });
  });
}

function kv(rows) {
  return '<dl class="kv">' + rows.map(([k, v]) =>
    '<dt>' + esc(k) + '</dt><dd>' + (v === undefined || v === null ? '—' : String(v)) + '</dd>'
  ).join('') + '</dl>';
}

function renderReport() {
  const host = $('#report-body');
  const s = S.scene;
  if (!s) { host.innerHTML = '<div class="empty">没有场景数据。</div>'; return; }
  const meta = s.build_meta || {};
  const r = S.report;
  const q = (r && r.quality) || {};
  const parts = [];

  parts.push('<div class="sec"><h3>场景概览</h3>' + kv([
    ['scene_id', esc(s.scene_id)],
    ['图像', esc(s.image_id) + ' · ' + I.W + '×' + I.H],
    ['物体', s.nodes.length + ' 个'],
    ['关系边', (s.edges || []).length + ' 条'],
    ['坐标系', '相机系 (' + esc(s.up_axis || '?') + ' 为上)'],
  ]) + '</div>');

  const counts = meta.label_counts || {};
  const cnt = Object.entries(counts).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => '<span class="tag t-dim" style="margin:2px 4px 2px 0">' + esc(k) + ' × ' + v + '</span>')
    .join('');
  parts.push('<div class="sec"><h3>类别计数（这就是模型看到的 <code>scene_hint</code>）</h3>' +
    '<div>' + (cnt || '<span class="muted">—</span>') + '</div>' +
    '<div class="note" style="margin-top:6px">' +
    '提示词里只有这些计数与图像尺寸，<b>没有任何坐标</b>。' +
    '模型想知道"沙发离桌子多远"，唯一的路径是让工具去算 —— 这是信息约束，不是提示词要求。</div></div>');

  parts.push('<div class="sec"><h3>感知质量</h3>' + kv([
    ['检测原始 / 保留', (meta.n_detections_raw ?? '—') + ' / ' + (meta.n_detections_kept ?? '—')],
    ['丢弃 / 兜底', (meta.n_dropped ?? '—') + ' / ' + (meta.n_fallbacks ?? '—')],
    ['掩码-框覆盖率均值', fmt(meta.mask_box_coverage_mean, 3)],
    ['掩码-框覆盖率最低', fmt(meta.mask_box_coverage_min, 3)],
    ['深度范围 (m)', Array.isArray(meta.depth_range_m) ? meta.depth_range_m.map((v) => fmt(v, 2)).join(' ~ ') : '—'],
    ['内参来源', esc((r && r.quality && r.quality.intrinsics_source) || meta.config?.intrinsics_source || '未知')],
    ['低点数物体', q.n_low_point_objects ?? '—'],
  ]) + '</div>');

  if (r && Array.isArray(r.caveats) && r.caveats.length) {
    parts.push('<div class="sec"><h3>系统自述的注意事项（caveats）</h3>' +
      r.caveats.map((c) => '<div class="note" style="margin-bottom:6px">· ' +
        esc(c).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>') + '</div>').join('') + '</div>');
  }
  if (r && r.diagnosis && Array.isArray(r.diagnosis.findings) && r.diagnosis.findings.length) {
    parts.push('<div class="sec"><h3>诊断（stage: ' + esc(r.diagnosis.stage || '—') + '）</h3>' +
      r.diagnosis.findings.map((f) => '<div class="note" style="margin-bottom:6px">· ' +
        esc(typeof f === 'string' ? f : jstr(f, 300)) + '</div>').join('') + '</div>');
  }
  if (r && r.summary) {
    parts.push('<div class="sec"><h3>场景摘要</h3><div class="note" style="white-space:pre-wrap">' +
      esc(r.summary) + '</div></div>');
  }
  if (!r) {
    parts.push('<div class="sec"><h3>场景报告</h3><div class="note">' +
      '这个场景没有 <code>report.json</code>（未跑 L5 场景级报告）。' +
      '要生成：<code>python scripts/report_scene.py --scene ' + esc(S.sceneId) + '</code></div></div>');
  }

  const prov = r && r.provenance;
  parts.push('<div class="sec"><h3>溯源</h3>' + kv([
    ['工具版本', esc((prov && prov.tools_version) || S.backend.toolsVersion || '—')],
    ['几何方法', esc((prov && prov.method) || '—')],
    ['场景目录', esc((prov && prov.scene_json) || '—')],
    ['源图', esc(s.image_url || '—')],
  ]) + '<div class="note" style="margin-top:6px">' +
    '判据：退出码只反映"循环有没有交出答案"，不反映答得对不对 —— ' +
    '对错需要真值，而真值不在 <code>agents/</code> 这一层（测量器械不长在被测方身上）。</div></div>');

  host.innerHTML = parts.join('');
  $('#report-src').textContent = r ? 'report.json 已加载' : '无报告文件';
}

/* ================================================================== 问答 */

function statusTag(st) {
  return '<span class="tag ' + (STATUS_CLASS[st] || 't-dim') + '">' + esc(st || '—') + '</span>';
}
function verdictTag(v) {
  if (!v) return '<span class="tag t-dim">未校验</span>';
  const lv = v.level || (v.ok === true ? 'supported' : 'unsupported');
  return '<span class="tag ' + (VERDICT_CLASS[lv] || 't-dim') + '">' +
    esc(VERDICT_LABEL[lv] || lv) + '</span>';
}

function renderTrace(trace) {
  if (!trace || !trace.length) return '<div class="empty">没有工具调用 —— 这一题程序没有向场景提问。</div>';
  return '<div class="trace">' + trace.map((t, i) => {
    const r = t.result || {};
    const ok = r.ok !== false;
    return '<div class="tr">' +
      '<div class="tr-head">' +
      '<span class="tag t-dim">' + (i + 1) + '</span>' +
      '<span class="tr-name">' + esc(t.tool || '?') + '</span>' +
      '<span class="tag ' + (ok ? 't-ok' : 't-err') + '">' +
      (ok ? 'ok' : esc((r.error && r.error.code) || 'error')) + '</span>' +
      (r.error && r.error.recovery ? '<span class="muted" style="font-size:11px">recovery: ' +
        esc(r.error.recovery) + '</span>' : '') +
      '</div>' +
      '<div class="tr-args">' + esc(jstr(t.args || {}, 220)) + '</div>' +
      (r.value !== undefined
        ? '<div class="tr-val">' + esc(jstr(r.value, 700)) + '</div>'
        : (r.error ? '<div class="tr-val">' + esc(jstr(r.error, 400)) + '</div>' : '')) +
      (r.evidence && r.evidence.length
        ? '<div class="tr-args" style="color:#79c0ff">evidence: ' + esc(jstr(r.evidence, 260)) + '</div>' : '') +
      '</div>';
  }).join('') + '</div>';
}

function renderRunDetail(run, extra) {
  if (!run) return '<div class="empty">选一道题看详细轨迹。</div>';
  const p = [];
  const ans = run.answer;
  const ansTxt = ans === null || ans === undefined ? '—' : (typeof ans === 'object' ? jstr(ans) : String(ans));

  p.push('<div class="answer">' +
    '<div class="big">' + esc(ansTxt) + '</div>' +
    '<div class="meta">' +
    statusTag(run.status) + verdictTag(run.verdict) +
    '<span>答案类型 <b>' + esc(run.answer_type || '?') + '</b></span>' +
    '<span>耗时 <b>' + esc(fmt(run.elapsed_s, 2)) + ' s</b></span>' +
    '<span>尝试 <b>' + esc(run.attempts ?? '—') + '</b> 次</span>' +
    (run.usage ? '<span>成本 <b>¥' + esc(fmt(run.usage.cost_cny, 5)) + '</b></span>' : '') +
    (run.usage ? '<span>LLM 调用 <b>' + esc(run.usage.calls ?? '—') + '</b></span>' : '') +
    '</div></div>');

  if (run.question) {
    p.push('<div class="sec"><h3>问题</h3><div class="note" style="color:var(--fg2);font-size:12.5px">' +
      esc(run.question) + '</div></div>');
  }
  if (run.evidence && run.evidence.length) {
    p.push('<div class="sec"><h3>提交时给出的证据</h3>' +
      run.evidence.map((e) => '<div class="note" style="font-family:var(--mono);font-size:11.5px">· ' +
        esc(String(e)) + '</div>').join('') + '</div>');
  }
  if (run.verdict && run.verdict.checks) {
    // ⚠ `checks` 是**数组** `[{name, ok, detail, hard}]`，不是 {名: 布尔} 的字典。
    // 早期版本曾是字典，这里两种都接 —— 否则 check 名会渲染成 0/1/2/3 的下标。
    const checks = Array.isArray(run.verdict.checks)
      ? run.verdict.checks.map((c) => ({
        name: c.name, ok: !!c.ok, detail: c.detail, hard: c.hard,
      }))
      : Object.entries(run.verdict.checks).map(([k, v]) => ({ name: k, ok: !!v }));
    p.push('<div class="sec"><h3>证据校验（四级结论）</h3>' +
      checks.map((c) => '<div class="note" style="margin-bottom:3px">' +
        '<span class="tag ' + (c.ok ? 't-ok' : 't-warn') + '">' + (c.ok ? '✓' : '✗') + '</span> ' +
        '<b>' + esc(c.name) + '</b>' + (c.hard ? ' <span class="muted">[硬]</span>' : '') +
        (c.detail ? '<br><span class="muted" style="margin-left:22px">' + esc(String(c.detail)) + '</span>' : '') +
        '</div>').join('') +
      '<div class="note" style="margin-top:6px">"写了 evidence" 与 "evidence 对得上" 是两件事 —— ' +
      '这一层专门抓后者。弃答单独占一级，因为它和"编了个数"必须分开计数。</div></div>');
  }
  if (run.stages && run.stages.length) {
    p.push('<div class="sec"><h3>阶段</h3><div class="stages">' +
      run.stages.map((s2) => '<span class="stg ' + (s2.ok ? 'ok' : 'bad') + '">' +
        '<span class="sd"></span>' + esc(s2.stage) + '</span>').join('') + '</div></div>');
  }
  if (run.plan && run.plan.ok !== undefined) {
    p.push('<div class="sec"><h3>臂 G 前置计划</h3>' +
      '<div class="note" style="margin-bottom:6px">这是**生成前**的一段思路，只插进提示词，' +
      '<b>不改变控制流</b>（计划本身不产生任何工具调用）。' +
      (run.plan.numbers && run.plan.numbers.length
        ? '<br><b style="color:var(--warn)">计划里出现了数字 ' + esc(jstr(run.plan.numbers, 120)) +
          '</b> —— 已留痕：那个数字会随计划进下一次提示词，而程序合成无从分辨它是猜的还是算的。'
        : '<br>计划里没有出现任何数字。') + '</div>' +
      '<div class="tr-val" style="max-height:220px">' + esc(run.plan.text || jstr(run.plan, 800)) + '</div></div>');
  }
  if (run.program) {
    // ⚠ 只在**明确是 false** 时告警：老产物里没有这个字段，而「没记录」不等于
    //   「不同源」。用 `=== false` 而不是 `!`，就是为了不做这个静默转换。
    if (run.program_matches_trace === false) {
      p.push('<div class="err-box">⚠ 下面这段程序<b>没有被执行过</b>（末轮静态检查没过）。' +
        '下方「工具调用轨迹」来自第 ' + esc(String(run.executed_attempt)) +
        ' 轮的另一段程序 —— 两者不是配套的，不要按它去解释那些调用。</div>');
    }
    p.push('<details class="pf"><summary>生成的程序（' + (run.program_fenced ? '带 ``` 围栏' : '未带围栏') +
      '）—— 这是 LLM 唯一的"控制流"产物</summary><pre>' + esc(run.program) + '</pre></details>');
  }
  if (run.failure) {
    p.push('<div class="err-box">失败信息：' + esc(jstr(run.failure, 500)) + '</div>');
  }
  p.push('<div class="sec"><h3>工具调用轨迹（' + ((run.trace || []).length) + ' 次）</h3></div>');
  p.push(renderTrace(run.trace));

  if (extra && extra.switches) {
    p.push('<div class="sec"><h3>本次开关组合</h3><div class="tr-val">' +
      esc(jstr(extra.switches, 600)) + '</div></div>');
  }
  if (extra && extra.vlm) {
    p.push('<div class="sec"><h3>视觉语义（角色②）</h3><div class="tr-val">' +
      esc(jstr(extra.vlm, 700)) + '</div>' +
      '<div class="note" style="margin-top:6px">角色② 的签名里<b>没有任何空间参数</b>' +
      '（导入期硬断言），它只能回答"是什么样"，不能回答"它在哪"。</div></div>');
  }
  return p.join('');
}

function renderReplay() {
  const host = $('#ask-body');
  const batches = (S.runs && S.runs.batches) || [];
  const mine = batches.filter((b) => b.scene_id === S.sceneId);
  const list = mine.length ? mine : batches;
  if (!list.length) {
    host.innerHTML = '<div class="empty">没有可回放的批次。<br>跑一次 ' +
      '<code>python scripts/run_agent.py --scene ' + esc(S.sceneId || 'living_room') + ' ...</code> ' +
      '再执行 <code>python scripts/export_demo.py</code>。</div>';
    return;
  }
  if (S.replay.batchIdx >= list.length) S.replay.batchIdx = 0;
  const b = list[S.replay.batchIdx];
  const runs = b.runs || [];
  // 默认落在第一题：空态看不出渲染对不对，而"一条完整轨迹"本身就是最好的说明书
  if (S.replay.runIdx < 0 && runs.length) S.replay.runIdx = 0;
  if (S.replay.runIdx >= runs.length) S.replay.runIdx = -1;

  const head = '<div class="ask-form"><div class="ask-row">' +
    '<span class="muted">批次</span>' +
    '<select id="batch-select">' + list.map((x, i) =>
      '<option value="' + i + '"' + (i === S.replay.batchIdx ? ' selected' : '') + '>' +
      esc(x.file || ('批次 ' + (i + 1))) + ' · ' + (x.runs || []).length + ' 题 · ' +
      esc(x.mtime || '') + '</option>').join('') + '</select>' +
    '<span class="muted" style="font-size:11.5px">' +
    esc((b.scene_hint && b.scene_hint.n_objects) || '?') + ' 个物体 · tools ' + esc(b.tools_version || '?') +
    (b.switches ? ' · ' + esc(jstr(b.switches, 90)) : '') + '</span>' +
    '</div></div>';

  const items = runs.map((r, i) => {
    const on = i === S.replay.runIdx ? ' on' : '';
    return '<div class="run-item' + on + '" data-i="' + i + '">' +
      statusTag(r.status) + verdictTag(r.verdict) +
      '<span class="q">' + esc(r.question || '(无问题)') + '</span>' +
      '<span class="muted" style="font-family:var(--mono);font-size:11px">' + esc(fmt(r.elapsed_s, 2)) + 's</span>' +
      '<span class="muted" style="font-family:var(--mono);font-size:11px">→ ' +
      esc(r.answer === null || r.answer === undefined ? '—' : String(r.answer)) + '</span>' +
      '</div>';
  }).join('');

  const detail = S.replay.runIdx >= 0 && runs[S.replay.runIdx]
    ? renderRunDetail(runs[S.replay.runIdx], { switches: b.switches })
    : '<div class="empty">选一道题看它的答案、证据与完整轨迹。</div>';

  host.innerHTML = head + '<div class="run-list">' + items + '</div>' + '<div>' + detail + '</div>';

  $('#batch-select').addEventListener('change', (e) => {
    S.replay.batchIdx = +e.target.value; S.replay.runIdx = -1; renderReplay();
  });
  $$('.run-item', host).forEach((el) => el.addEventListener('click', () => {
    S.replay.runIdx = +el.dataset.i;
    renderReplay();
    // 题目里点名了物体 → 顺手选中，让 3D 和图像联动起来
    const r = runs[S.replay.runIdx];
    const ids = (r.target_ids || []).filter((id) => S.nodes.some((n) => n.id === id));
    S.selected = new Set(ids);
    refreshSelectionVisuals();
  }));
}

function renderLive() {
  const host = $('#ask-body');
  const ready = S.backend.online && S.backend.ready;
  host.innerHTML =
    '<div class="ask-form">' +
    '<textarea id="q-input" placeholder="例：沙发和桌子之间的三维距离是多少米？"></textarea>' +
    '<div class="ask-row">' +
    '<span class="muted">答案类型</span>' +
    '<select id="q-type"><option value="">自动</option><option value="float" selected>float</option>' +
    '<option value="int">int</option><option value="str">str</option><option value="bool">bool</option></select>' +
    '<label class="mini"><input type="checkbox" id="q-planner"> 臂G 前置计划</label>' +
    '<label class="mini"><input type="checkbox" id="q-vlm" checked> 视觉语义</label>' +
    '<button class="btn btn-primary" id="btn-ask"' + (ready ? '' : ' disabled') + '>提问</button>' +
    '<span id="ask-status" class="muted" style="font-size:11.5px">' +
    (ready ? '将真跑一次 AgentLoop（每题一次 LLM 调用）'
           : (S.backend.online ? '后端未就绪：' + esc((S.backend.info && S.backend.info.error) || '缺少 API key')
                               : '后端不在线 —— 只能回放')) + '</span>' +
    '</div>' +
    '<div class="note">提示词里只有"椅子 × 2"这类计数和图像尺寸，<b>没有任何坐标</b>；' +
    '模型拿米数的唯一路径是让工具去算。答案必须能追回某次工具返回值，否则校验器会判它证据不成立。</div>' +
    '</div>' +
    '<div id="live-detail">' + (S.lastLive ? renderRunDetail(S.lastLive.run, S.lastLive) :
      '<div class="empty">问一句试试。现场提问走的就是实验里那套装配 —— ' +
      '同一个工具集、同一个模型、同一套开关（<code>run_agent.build_session</code>）。</div>') + '</div>';

  const btn = $('#btn-ask');
  if (btn && ready) btn.addEventListener('click', doAsk);
  const ta = $('#q-input');
  if (ta) ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doAsk();
  });
}

async function doAsk() {
  if (S.busy) return;
  const q = ($('#q-input').value || '').trim();
  if (!q) { $('#ask-status').textContent = '先写一个问题。'; return; }
  S.busy = true;
  const btn = $('#btn-ask');
  btn.disabled = true;
  const t0 = performance.now();
  const tick = setInterval(() => {
    $('#ask-status').innerHTML = '<span class="spin"></span> 正在跑（' +
      ((performance.now() - t0) / 1000).toFixed(1) + ' s）—— 一次 LLM 调用通常 2~20 s';
  }, 120);

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scene_id: S.sceneId,
        question: q,
        answer_type: $('#q-type').value || null,
        planner: $('#q-planner').checked ? 'on' : 'off',
        vlm: $('#q-vlm').checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || ('HTTP ' + resp.status));
    S.lastLive = data;
    $('#live-detail').innerHTML = renderRunDetail(data.run, data);
    $('#ask-status').textContent = '完成。';
    // 自动选中该题指认的物体，让 3D 立刻联动
    const ids = (data.run.target_ids || []).filter((id) => S.nodes.some((n) => n.id === id));
    if (ids.length) { S.selected = new Set(ids); refreshSelectionVisuals(); }
  } catch (exc) {
    $('#live-detail').innerHTML = '<div class="err-box">提问失败：' + esc(exc.message) +
      '<br><span class="muted">若提示连接被拒，检查 scripts/serve_demo.py 是否还在跑、' +
      '以及本机代理（127.0.0.1:7897）是否占用。</span></div>';
    $('#ask-status').textContent = '失败。';
  } finally {
    clearInterval(tick);
    S.busy = false;
    btn.disabled = false;
  }
}

/* ============================================================== 反事实 */

async function doCounterfactual() {
  if (!S.selected.size || !S.backend.online) return;
  const btn = $('#btn-cf');
  const old = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>';
  try {
    const resp = await fetch('/api/counterfactual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: S.sceneId, remove: [...S.selected] }),
    });
    const data = await resp.json();
    const host = $('#report-body');
    const sec = document.createElement('div');
    sec.className = 'sec';
    sec.innerHTML = '<h3>反事实：移走 ' + esc([...S.selected].join(', ')) + '</h3>' +
      '<div class="tr-val" style="max-height:280px">' +
      esc(jstr((data.result && data.result.value) || data, 2000)) + '</div>' +
      '<div class="note" style="margin-top:6px">纯图操作，不花 API 钱 —— ' +
      '它改的是场景图的边，不改原图，也不改已经落盘的结果。</div>';
    host.prepend(sec);
    host.scrollTop = 0;
  } catch (exc) {
    const host = $('#report-body');
    const sec = document.createElement('div');
    sec.className = 'err-box';
    sec.textContent = '反事实失败：' + exc.message;
    host.prepend(sec);
  } finally {
    btn.textContent = old;
    btn.disabled = false;
  }
}

/* ============================================================== 场景装载 */

async function selectScene(sceneId) {
  S.sceneId = sceneId;
  S.selected.clear();
  S.replay = { batchIdx: 0, runIdx: -1 };
  S.lastLive = null;
  $('#scene-select').value = sceneId;

  const get = async (url) => {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
    return r.json();
  };

  S.scene = await get('data/' + encodeURIComponent(sceneId) + '.scene.json');
  S.nodes = S.scene.nodes || [];
  buildPalette(S.nodes);

  const meta = S.index.scenes.find((s) => s.scene_id === sceneId) || {};
  S.report = null;
  if (meta.has_report) {
    try { S.report = await get('data/' + encodeURIComponent(sceneId) + '.report.json'); }
    catch { S.report = null; }
  }

  build3D();
  await loadImageForScene();

  // —— 默认选中。没有它，首屏就是"什么都没选"，
  //    而三维盒与掩码都是**按选中上色**的，看起来会像功能坏了。
  //    URL 的 ?select=<node_id> 优先（演示时可直接定格到某个物体）。
  const want = (URL_SELECT && S.nodes.some((n) => n.id === URL_SELECT))
    ? URL_SELECT
    : (S.nodes.slice().sort((a, b) => (b.n_points || 0) - (a.n_points || 0))[0] || {}).id;
  if (want) S.selected.add(want);
  refreshSelectionVisuals();

  setBanners();
  renderReport();
  updateChips();
  if (S.mode === 'replay') renderReplay(); else renderLive();
}

/* ============================================================ 上传建图 */

/**
 * ArrayBuffer → base64，**分块**做。
 *
 * `btoa(String.fromCharCode(...new Uint8Array(buf)))` 是这类需求最常见的一行写法，
 * 但它把整张图摊成了函数实参：一张 3 MB 的照片就是 300 万个实参，
 * 浏览器直接 `RangeError: Maximum call stack size exceeded`。
 * 这个错误只在**真选了大图**时才出现 —— 拿小图测永远看不到它。
 */
function toBase64(buf) {
  const bytes = new Uint8Array(buf);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

/**
 * 从 JPEG 头读出 EXIF 的 tag 号集合（够用即可，不做完整解析）。
 *
 * 只为一个问题服务：**这张图有没有 35mm 等效焦距（tag 0xA405）**。
 * 它直接决定横向米制尺度可不可信 —— 有它就能换算内参，没有就只能用模型预测值，
 * 而两者在同一张图上的横向误差相差 118 倍（306.3 px vs 2.6 px）。
 *
 * 之所以要在**点「开始建图」之前**就回答这个问题：等建完图再让用户从横幅里
 * 自己发现，那已经是几十秒之后了，而"要不要选这张图"的判断本该在那之前做。
 */
function readExifTags(buf) {
  const tags = new Set();
  const dv = new DataView(buf);
  if (dv.byteLength < 8 || dv.getUint16(0) !== 0xFFD8) return tags;    // 非 JPEG
  const limit = Math.min(dv.byteLength, 256 * 1024);                   // EXIF 在文件头
  let off = 2;
  while (off + 4 <= limit) {
    if (dv.getUint8(off) !== 0xFF) break;
    const marker = dv.getUint8(off + 1);
    if (marker === 0xD8 || marker === 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { off += 2; continue; }
    if (marker === 0xDA || marker === 0xD9) break;                     // 进入图像数据，不再有 EXIF
    const size = dv.getUint16(off + 2);
    if (size < 2) break;
    if (marker === 0xE1) {
      const base = off + 4;
      if (base + 8 <= limit && dv.getUint32(base) === 0x45786966) {    // "Exif"
        const tiff = base + 6;
        const le = dv.getUint16(tiff) === 0x4949;
        const walk = (ifd, depth) => {
          if (depth > 2 || ifd + 2 > limit) return;
          const n = dv.getUint16(ifd, le);
          for (let i = 0; i < n; i++) {
            const e = ifd + 2 + i * 12;
            if (e + 12 > limit) return;
            const tag = dv.getUint16(e, le);
            tags.add(tag);
            if (tag === 0x8769) walk(tiff + dv.getUint32(e + 8, le), depth + 1);  // ExifIFD
          }
        };
        walk(tiff + dv.getUint32(tiff + 4, le), 0);
      }
    }
    off += 2 + size;
  }
  return tags;
}

const UP = { file: null, data: null, exif35: null, jobId: null, timer: null, busy: false };

/** 重新拉 index.json 并重建场景下拉 —— 新建的场景在这一步出现。 */
async function reloadIndex() {
  const r = await fetch('data/index.json', { cache: 'no-store' });
  S.index = await r.json();
  fillSceneSelect();
}

function fillSceneSelect() {
  const sel = $('#scene-select');
  const cur = S.sceneId;
  sel.innerHTML = (S.index.scenes || []).map((s) =>
    '<option value="' + esc(s.scene_id) + '">' + esc(s.scene_id) +
    ' · ' + s.n_nodes + '物体/' + s.n_edges + '边' +
    (s.up_axis_reliable ? '' : ' ⚠重力') + (s.scale_calibrated ? '' : ' ⚠尺度') +
    '</option>').join('');
  if (cur) sel.value = cur;
}

function setupUpload() {
  const modal = $('#upload-modal');
  const drop = $('#up-drop');
  const input = $('#up-file');
  const preview = $('#up-preview');
  const status = $('#up-status');
  const go = $('#up-go');
  const log = $('#up-log');
  const intrSel = $('#up-intr');
  const manualWrap = $('#up-manual-wrap');
  const manual = $('#up-manual');
  const f35Wrap = $('#up-f35-wrap');
  const f35 = $('#up-f35');

  const setStatus = (text, cls = 'muted') => { status.className = cls; status.textContent = text; };
  const intrinsicsValue = () => {
    if (intrSel.value === 'manual') return manual.value.trim();
    // 用户在框里给的是**等效焦距**，不是四个像素值：换算交给后端
    // （f_px = f35/36 × 长边），前端不自己算 —— 两处各写一份公式迟早分叉，
    // 而分叉之后同一张图经两条路得到不同 K，谁对谁错在产物里看不出来。
    if (intrSel.value === 'f35') return 'f35:' + f35.value.trim();
    return intrSel.value;
  };
  const updateGo = () => {
    const needManual = intrSel.value === 'manual' && !manual.value.trim();
    const needF35 = intrSel.value === 'f35' && !f35.value.trim();
    go.disabled = UP.busy || !UP.data || needManual || needF35;
  };

  intrSel.addEventListener('change', () => {
    manualWrap.classList.toggle('hidden', intrSel.value !== 'manual');
    f35Wrap.classList.toggle('hidden', intrSel.value !== 'f35');
    updateGo();
  });
  manual.addEventListener('input', updateGo);
  f35.addEventListener('input', updateGo);

  const takeFile = async (f) => {
    if (!f) return;
    UP.file = f;
    UP.data = null;
    if (!f.type.startsWith('image/') && !/\.(png|jpe?g|webp|bmp)$/i.test(f.name)) {
      setStatus('这不是图片文件', 'pill-err'); updateGo(); return;
    }
    if (f.size > 16 * 1024 * 1024) {
      setStatus('图片 ' + (f.size / 1048576).toFixed(1) + ' MB，超过 16 MB 上限', 'pill-err');
      updateGo(); return;
    }
    setStatus('读取中…');
    let buf;
    try { buf = await f.arrayBuffer(); }
    catch (e) { setStatus('读文件失败：' + e.message, 'pill-err'); return; }

    UP.data = toBase64(buf);
    let tags = new Set();
    try { tags = readExifTags(buf); }
    catch { /* 解析失败按"无 EXIF"处理：这是提示，不是校验，不该拦人 */ }
    UP.exif35 = tags.has(0xA405);

    preview.innerHTML = '<img src="' + URL.createObjectURL(f) + '" alt="预览"><div class="sm">' +
      esc(f.name) + ' ｜ ' + (f.size / 1024).toFixed(0) + ' KB ｜ ' +
      (UP.exif35 ? '<b style="color:var(--ok)">检测到 EXIF 等效焦距</b>'
                 : '<b style="color:var(--warn)">没有 EXIF 等效焦距</b>') + '</div>';
    if (!$('#up-scene-id').value.trim()) {
      const stem = f.name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 32);
      if (stem) $('#up-scene-id').value = stem;
    }
    setStatus(UP.exif35
      ? '有 EXIF 焦距 → 内参可从 EXIF 换算（选出 auto 即可）'
      : '无 EXIF 焦距（微信/社交软件转发的图必丢）→ 会被标成「尺度未标定」。'
        + '两条出路：改用相册原图，或在下方内参选「等效焦距」手填一次（1× 主摄通常 24 mm）');
    updateGo();
  };

  drop.addEventListener('click', () => { if (!UP.busy) input.click(); });
  input.addEventListener('change', () => takeFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', (e) => {
    if (UP.busy) return;
    const f = e.dataTransfer?.files?.[0];
    if (f) takeFile(f);
  });

  const poll = async () => {
    if (!UP.jobId) return;
    let j;
    try {
      const r = await fetch('/api/build/status?id=' + encodeURIComponent(UP.jobId), { cache: 'no-store' });
      j = await r.json();
      if (!j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    } catch (e) {
      // 单次网络抖动不该把任务判死：任务在后端线程里跑着，
      // 前端丢了它只是丢了"观察窗口"，下一次轮询就能接回来。
      UP.timer = setTimeout(poll, 2500);
      return;
    }
    const stageName = {
      queued: '排队中', decode: '解码', build: '建图中',
      report: '生成报告', export: '导出',
    }[j.stage] || j.stage;
    // 两个时间都要报：只报一个的话，"排队 33 秒"和"建图 33 秒"长得一样，
    // 而它们该让人做的判断完全不同（前者是等，后者是看它有没有卡住）。
    setStatus(j.state === 'queued'
      ? '排队中 · 已等 ' + fmt(j.elapsed_s, 1) + ' s（同一时刻只跑一个建图任务）'
      : stageName + ' · 本阶段 ' + fmt(j.stage_s, 1) + ' s ／ 共计 ' + fmt(j.elapsed_s, 1) + ' s');
    log.textContent = (j.tail || []).join('\n') || '（还没有输出）';
    log.scrollTop = log.scrollHeight;

    if (j.state === 'done') { UP.busy = false; updateGo(); await onBuilt(j); return; }
    if (j.state === 'failed') {
      UP.busy = false; updateGo();
      setStatus('建图失败', 'pill-err');
      log.textContent += '\n\n✗ ' + (j.error || '未知错误');
      return;
    }
    UP.timer = setTimeout(poll, 1200);
  };

  const onBuilt = async (j) => {
    const res = j.result || {};
    setStatus('完成：' + j.scene_id + ' · ' + (res.n_nodes || 0) + ' 个物体 · 用时 ' +
      fmt(j.elapsed_s, 1) + ' s', 't-ok');

    // ★ 结果里第一件事要说的不是"几个物体"，而是**这些数字可不可信**。
    //   上传的图有没有 EXIF 内参，直接决定米数能不能按米读；
    //   只说"9 个物体"会让人以为它和标定过的场景一样可信。
    log.textContent += '\n\n✓ 场景 ' + j.scene_id + ' 已建成，已并入演示台';
    log.textContent += '\n内参来源：' + (res.intrinsics_source || '未知');
    const fov = res.fov || {};
    if (fov.hfov_deg != null || fov.plausible != null) {
      log.textContent += '\n视场：HFoV ' + fmt(fov.hfov_deg, 1) + '°  可信=' + fov.plausible +
        (fov.reason ? '  (' + fov.reason + ')' : '');
    }
    log.textContent += '\n尺度标定：' + (res.scale_calibrated
      ? '已标定 —— 米制数字可按绝对值读'
      : '【未标定 —— 所有米制数字只能按相对值读】');
    log.textContent += '\n重力方向：tilt ' + fmt(res.up_axis_tilt_deg, 2) + '°' +
      '  reliable=' + res.up_axis_reliable +
      (res.up_axis_reason ? '  (' + res.up_axis_reason + ')' : '');
    log.textContent += '\n检测 ' + fmt(res.n_detections_raw, 0) + ' → 保留 ' +
      fmt(res.n_detections_kept, 0) + ' → 建节点 ' + (res.n_nodes || 0);
    log.scrollTop = log.scrollHeight;

    await reloadIndex();
    if (res.scene_id) await selectScene(res.scene_id);
  };

  go.addEventListener('click', async () => {
    if (UP.busy || !UP.data) return;
    UP.busy = true; updateGo();
    log.classList.remove('hidden');
    log.textContent = '提交任务…';
    setStatus('提交中…');
    try {
      const r = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: UP.data,
          filename: UP.file ? UP.file.name : 'upload.png',
          scene_id: $('#up-scene-id').value.trim(),
          prompt: $('#up-prompt').value.trim(),
          intrinsics: intrinsicsValue(),
        }),
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || ('HTTP ' + r.status));
      UP.jobId = j.job_id;
      setStatus('已排队，等待建图…');
      poll();
    } catch (e) {
      UP.busy = false; updateGo();
      setStatus('提交失败：' + e.message, 'pill-err');
      log.textContent += '\n提交失败：' + e.message;
    }
  });

  $('#up-close').addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });
  $('#btn-upload').addEventListener('click', () => modal.classList.remove('hidden'));
  updateGo();
  return { disable: () => { const b = $('#btn-upload'); b.disabled = true; b.title = '需要后端（python scripts/serve_demo.py）'; } };
}

/* ============================================================== 启动 */

async function probeBackend() {
  const el = $('#conn');
  try {
    const r = await fetch('/api/health', { cache: 'no-store' });
    const j = await r.json();
    if (!j || j.ok !== true) throw new Error('health 不是本演示后端');
    S.backend = { online: true, ready: !!j.backend?.ready, info: j.backend, toolsVersion: j.tools_version };
    if (S.backend.ready) {
      el.className = 'pill pill-ok';
      el.innerHTML = '<span class="d"></span>后端就绪 · ' + esc(S.backend.info.model || '');
    } else {
      el.className = 'pill pill-warn';
      el.innerHTML = '<span class="d"></span>后端在线但未就绪';
    }
  } catch {
    S.backend = { online: false, ready: false, info: null, toolsVersion: null };
    el.className = 'pill pill-dim';
    el.innerHTML = '<span class="d"></span>静态回放模式';
    const liveTab = $('.tab[data-mode="live"]');
    if (liveTab) { liveTab.disabled = true; liveTab.style.opacity = '.45'; liveTab.style.cursor = 'not-allowed'; }
  }
}

async function boot() {
  init3D();
  initImage();
  const upload = setupUpload();

  // 顶部交互
  $('#btn-fit').addEventListener('click', fit3D);
  $('#chk-labels').addEventListener('change', updateLabels);
  $('#chk-links').addEventListener('change', rebuildLinks);
  $('#chk-mask').addEventListener('change', drawImage);
  $('#chk-box').addEventListener('change', drawImage);
  $('#btn-clear').addEventListener('click', () => { S.selected.clear(); refreshSelectionVisuals(); });
  $('#btn-cf').addEventListener('click', doCounterfactual);
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const modal = $('#upload-modal');
    if (modal && !modal.classList.contains('hidden')) { modal.classList.add('hidden'); return; }
    S.selected.clear(); refreshSelectionVisuals();
  });

  // —— 布局切换。做成四选一而不是挑一个"最优"，是因为不同内容怕的拥挤
  //    不是同一种：看三维时嫌的是横向视野窄，读报告时嫌的是纵向行数少。
  //    与其替所有人猜一个折中值，不如让用的人一键换。选择记在 localStorage，
  //    演示时不该每次都重选。
  const seg = $('#layout-seg');
  const LAYOUTS = ['grid', 'wide', 'focus', 'stack'];
  const applyLayout = (name, persist = true) => {
    document.body.dataset.layout = name;
    $$('.seg-b', seg).forEach((b) => b.classList.toggle('on', b.dataset.layout === name));
    if (persist) { try { localStorage.setItem('spatialLayout', name); } catch { /* 隐私模式 */ } }
    // 面板尺寸变了 → three.js 必须跟着 resize，否则画面会被拉伸
    requestAnimationFrame(() => { resize3D(); updateLabels(); });
  };
  let saved = '';
  try { saved = localStorage.getItem('spatialLayout') || ''; } catch { /* 隐私模式 */ }
  applyLayout(LAYOUTS.includes(URL_LAYOUT) ? URL_LAYOUT
    : (LAYOUTS.includes(saved) ? saved : 'grid'), false);
  $$('.seg-b', seg).forEach((b) => b.addEventListener('click', () => applyLayout(b.dataset.layout)));

  // 尺寸还会因为别的原因变（窗口缩放、滚动条出现），用 ResizeObserver 统一兜住
  if (window.ResizeObserver) {
    new ResizeObserver(() => { resize3D(); updateLabels(); }).observe($('#viewport').parentElement);
  }

  // 坐标系说明默认收成一行：它有信息量，但连续看三维时它是遮挡物。
  $('#axis-note')?.addEventListener('click', (e) => e.currentTarget.classList.toggle('open'));

  await probeBackend();
  if (!S.backend.online) upload.disable();

  try {
    S.index = await (await fetch('data/index.json', { cache: 'no-store' })).json();
  } catch (exc) {
    $('#ask-body').innerHTML = '<div class="empty">读不到 <code>data/index.json</code>。<br>' +
      '这个页面需要由后端托管（<b>file:// 直接打开会被浏览器的模块/同源策略拦住</b>）：<br><br>' +
      '<code>python scripts/serve_demo.py</code><br>然后打开 http://127.0.0.1:8770/</div>';
    $('#report-body').innerHTML = '<div class="err-box">静态数据不可读：' + esc(exc.message) + '</div>';
    return;
  }

  const sel = $('#scene-select');
  fillSceneSelect();
  sel.addEventListener('change', (e) => selectScene(e.target.value).catch(showSceneError));
  $$('.tab').forEach((t) => t.addEventListener('click', () => {
    if (t.disabled) return;
    S.mode = t.dataset.mode;
    $$('.tab').forEach((x) => x.classList.toggle('on', x === t));
    $('#ask-hint').textContent = S.mode === 'replay' ? '零成本 · 读已落盘的批次' : '真实调用 · 每题一次 LLM';
    if (S.mode === 'replay') renderReplay(); else renderLive();
  }));

  try {
    S.runs = await (await fetch('data/runs.json', { cache: 'no-store' })).json();
  } catch { S.runs = { batches: [] }; }

  // 默认挑一个"最完整"的场景：URL 指定 > 有报告且标定过的 > 有报告的 > 第一个
  const scenes = S.index.scenes;
  const best = (URL_SCENE ? scenes.find((s) => s.scene_id === URL_SCENE) : null)
    || scenes.find((s) => s.has_report && s.up_axis_reliable && s.scale_calibrated)
    || scenes.find((s) => s.has_report) || scenes[0];
  await selectScene(best.scene_id);

  // `?mode=live` 直接进现场提问（演示时少点一次；后端不在线时留在回放）
  const liveTab = $('.tab[data-mode="live"]');
  if (URL_MODE === 'live' && liveTab && !liveTab.disabled) liveTab.click();
  else $('#ask-hint').textContent = '零成本 · 读已落盘的批次';

  // `?upload=1` 直接打开上传面板
  if (URL_UPLOAD && S.backend.online) $('#upload-modal').classList.remove('hidden');
}

function showSceneError(exc) {
  $('#report-body').innerHTML = '<div class="err-box">场景装载失败：' + esc(exc.message) + '</div>';
}

boot().catch((exc) => {
  const host = $('#ask-body');
  if (host) host.innerHTML = '<div class="err-box">启动失败：' + esc(exc.message) + '</div>';
});
