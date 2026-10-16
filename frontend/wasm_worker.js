// ブラウザ内(サーバーなし)で STEP / QIF を読み込むワーカー。
// メインスレッドを塞がないよう、重い処理はすべてここで行い、段階ごとに進捗を送る。
//  - 形状のメッシュ化: occt-import-js (OpenCascade の WebAssembly 版)
//  - PMI の解析     : Pyodide 上で step_pmi.py をそのまま実行
//  - QIF           : Pyodide 上で qif_viewer のパーサ/テッセレータを実行
//  - 面との対応付け : backend/main.py の match_face_triangle_ranges の移植
// 送信メッセージ: {type:"progress", label, p, to}  p=現在の進捗(0-1), to=その段階の終端(あれば時間で漸近表示)
//                {type:"done", result} / {type:"error", message}
const OCCT_BASE = "https://cdn.jsdelivr.net/npm/occt-import-js@0.0.23/dist/";
const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
let occtPromise = null, pyPromise = null, qifPromise = null;

function progress(label, p, to) { postMessage({ type: "progress", label, p, to: to == null ? null : to }); }
const tick = () => new Promise(r => setTimeout(r, 0));

function getOcct() {
  if (!occtPromise) occtPromise = (async () => {
    importScripts(OCCT_BASE + "occt-import-js.js");
    return await occtimportjs({ locateFile: (f) => OCCT_BASE + f });
  })();
  return occtPromise;
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " が見つかりません");
  return await r.text();
}

// step_pmi.py は変更せず、関数を包んで段階ごとの進捗を出す
const PY_SETUP = `
import json, re, step_pmi, pmi_display
_prog = None
def set_prog(f):
    global _prog
    _prog = f
def _stage(name, label, p0, p1):
    fn = getattr(step_pmi, name)
    def w(*a, **k):
        if _prog: _prog(label, p0, p1)
        return fn(*a, **k)
    setattr(step_pmi, name, w)
_stage('find_length_unit_to_mm_factor', 'Reading units...', 0.55, 0.56)
_stage('extract_geometric_tolerances', 'Extracting geometric tolerances...', 0.56, 0.62)
_stage('extract_dimensions', 'Extracting dimensions...', 0.62, 0.66)
_stage('extract_datum_features', 'Extracting datum features...', 0.66, 0.68)
_stage('extract_annotation_targets', 'Extracting annotations...', 0.68, 0.72)
_stage('extract_face_anchor_points', 'Locating target faces...', 0.72, 0.78)
_stage('extract_datums', 'Extracting datums...', 0.78, 0.80)
_stage('extract_tessellated_pmi', 'Extracting graphic PMI...', 0.80, 0.84)
_stage('extract_saved_views', 'Extracting saved views...', 0.84, 0.85)
# エンティティ走査は件数が多いので、位置から実際の進捗を出す(0.28 -> 0.55)
_ENT_PAT = r'#(\\d+)\\s*='
class _Re:
    def __getattr__(self, n): return getattr(re, n)
    def finditer(self, pat, text, *a, **k):
        it = re.finditer(pat, text, *a, **k)
        if pat != _ENT_PAT or not _prog: return it
        total = max(1, len(text)); step = max(1, total // 200)
        def gen():
            nxt = 0
            for m in it:
                if m.start() >= nxt:
                    nxt = m.start() + step
                    _prog('Reading STEP entities... %d%%' % (100 * m.start() // total), 0.28 + 0.27 * m.start() / total, None)
                yield m
        return gen()
step_pmi.re = _Re()
def _extract(text):
    return json.dumps(step_pmi.extract_pmi(text))
def _decorate(js):
    tols = json.loads(js)
    pmi_display.decorate_step_tolerances(tols)
    return json.dumps(tols)
`;

const PY_QIF_SETUP = `
import json, qif_loader
import qif_viewer.tessellate.mesh_builder as _mb
# 面ごとのテッセレーションを数えて、実際の進捗(0.22 -> 0.85)を出す
_st = {'n': 0, 'total': 1, 'next': 0}
_orig_tf = _mb.tessellate_face
def _tf(*a, **k):
    r = _orig_tf(*a, **k)
    _st['n'] += 1
    if _prog and _st['n'] >= _st['next']:
        _st['next'] = _st['n'] + max(1, _st['total'] // 100)
        _prog('Tessellating faces... %d/%d' % (_st['n'], _st['total']), 0.22 + 0.63 * _st['n'] / _st['total'], None)
    return r
_mb.tessellate_face = _tf
_orig_bm = qif_loader.build_mesh
def _bm(doc):
    _st['total'] = max(1, len(doc.faces)); _st['n'] = 0; _st['next'] = 0
    return _orig_bm(doc)
qif_loader.build_mesh = _bm
_orig_ld = qif_loader.load_qif_document
def _ld(path):
    if _prog: _prog('Parsing QIF XML...', 0.14, 0.22)
    return _orig_ld(path)
qif_loader.load_qif_document = _ld
def _load_qif(path):
    mesh, pmi = qif_loader.load_qif(path)
    if _prog: _prog('Packing result...', 0.9, 0.93)
    return json.dumps({'mesh': mesh, 'pmi': pmi})
`;

function getPy() {
  if (!pyPromise) pyPromise = (async () => {
    importScripts(PYODIDE_BASE + "pyodide.js");
    const py = await loadPyodide({ indexURL: PYODIDE_BASE });
    for (const f of ["step_pmi.py", "pmi_display.py"]) py.FS.writeFile(f, await fetchText("py/" + f));
    py.runPython(PY_SETUP);
    py.globals.get("set_prog")(progress);
    return py;
  })();
  return pyPromise;
}

// QIF は Pyodide 上で qif_viewer のパーサ/テッセレータを実行して読む(numpy のみ使用)
function getQif() {
  if (!qifPromise) qifPromise = (async () => {
    const py = await getPy();
    progress("Loading QIF engine...", 0.04, 0.06);
    await py.loadPackage("numpy");
    // 穴あき平面の三角形分割に使う earcut(Python 側の mapbox_earcut スタブ経由)
    importScripts("https://cdn.jsdelivr.net/npm/earcut@2.2.4/dist/earcut.min.js");
    const files = JSON.parse(await fetchText("py/qif_viewer/manifest.json"));
    const dirs = new Set();
    for (const f of files) {
      const parts = f.split("/"); let d = "qif_viewer";
      for (let i = 0; i < parts.length - 1; i++) { d += "/" + parts[i]; dirs.add(d); }
    }
    try { py.FS.mkdir("qif_viewer"); } catch (_) {}
    [...dirs].sort().forEach(d => { try { py.FS.mkdir(d); } catch (_) {} });
    let n = 0;
    await Promise.all(files.map(async (f) => {
      py.FS.writeFile("qif_viewer/" + f, await fetchText("py/qif_viewer/" + f));
      progress("Loading QIF engine... " + (++n) + "/" + files.length, 0.06 + 0.06 * n / files.length, null);
    }));
    for (const f of ["pyvista.py", "mapbox_earcut.py", "qif_loader.py"]) py.FS.writeFile(f, await fetchText("py/" + f));
    py.runPython(PY_QIF_SETUP);
    return py;
  })();
  return qifPromise;
}

async function loadQif(arrayBuffer, filename) {
  progress("Loading QIF engine...", 0.03, 0.12);
  const py = await getQif();
  py.FS.writeFile("/tmp/input.qif", new Uint8Array(arrayBuffer));
  let out;
  try { out = py.globals.get("_load_qif")("/tmp/input.qif"); }
  catch (e) { throw new Error("QIFファイルとして読み込めませんでした: " + String(e.message || e).trim().split("\n").pop()); }
  progress("Transferring result...", 0.93, null);
  return { filename, ...JSON.parse(out) };
}

// メッシュ(occt-import-js の複数メッシュ)を 1 つにまとめ、面ごとの三角形範囲と照合用の格子を作る
function buildMesh(res) {
  const positions = [], indices = [], faceRanges = [];
  const grid = new Map(), faces = [];
  let vOff = 0, tOff = 0;
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const m of res.meshes) {
    const P = m.attributes.position.array, I = m.index.array;
    for (let i = 0; i < P.length; i++) positions.push(P[i]);
    for (let i = 0; i < P.length; i += 3) for (let a = 0; a < 3; a++) {
      if (P[i + a] < mn[a]) mn[a] = P[i + a]; if (P[i + a] > mx[a]) mx[a] = P[i + a];
    }
    for (let i = 0; i < I.length; i++) indices.push(I[i] + vOff);
    for (const bf of (m.brep_faces || [])) {
      const id = faces.length, range = [tOff + bf.first, tOff + bf.last + 1];
      faces.push(range); faceRanges.push(range);
      const seen = new Set();
      for (let t = bf.first; t <= bf.last; t++) for (let k = 0; k < 3; k++) {
        const vi = I[t * 3 + k]; if (seen.has(vi)) continue; seen.add(vi);
        const x = P[vi * 3], y = P[vi * 3 + 1], z = P[vi * 3 + 2];
        const key = Math.round(x * 1000) + "," + Math.round(y * 1000) + "," + Math.round(z * 1000);
        let a = grid.get(key); if (!a) grid.set(key, a = []);
        a.push([x, y, z, id]);
      }
    }
    vOff += P.length / 3; tOff += I.length / 3;
  }
  if (!isFinite(mn[0])) throw new Error("形状が見つかりませんでした");
  return { mesh: { positions, indices, bbox: { min: mn, max: mx }, faceCount: indices.length / 3, faceRanges }, grid, faces };
}

// PMI の対象面の境界頂点座標を、メッシュ頂点と突き合わせて、対応する面の三角形範囲を返す
// (単精度の座標なので 0.3 µm の誤差を許す)
function matchRanges(anchors, grid, faces) {
  if (!anchors || !anchors.length) return [];
  const uniq = new Map();
  for (const p of anchors) uniq.set(p[0].toFixed(3) + "," + p[1].toFixed(3) + "," + p[2].toFixed(3), p);
  const score = new Map();
  for (const p of uniq.values()) {
    const cx = Math.round(p[0] * 1000), cy = Math.round(p[1] * 1000), cz = Math.round(p[2] * 1000);
    const hit = new Set();
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const a = grid.get((cx + dx) + "," + (cy + dy) + "," + (cz + dz)); if (!a) continue;
      for (const v of a) if (Math.hypot(v[0] - p[0], v[1] - p[1], v[2] - p[2]) < 3e-4) hit.add(v[3]);
    }
    hit.forEach(f => score.set(f, (score.get(f) || 0) + 1));
  }
  if (!score.size) return [];
  let best = 0; score.forEach(n => { if (n > best) best = n; });
  const thr = best > 2 ? Math.max(1, best - 1) : 1;
  const out = [];
  score.forEach((n, f) => { if (n >= thr) out.push(faces[f]); });
  out.sort((a, b) => a[0] - b[0]);
  return out;
}

async function load(arrayBuffer, filename) {
  const bytes = new Uint8Array(arrayBuffer);
  progress("Loading CAD engine...", 0.02, 0.12);
  const [occt, py] = await Promise.all([getOcct(), getPy()]);
  progress("Tessellating shape (large files take a while)...", 0.12, 0.27);
  await tick();
  const res = occt.ReadStepFile(bytes, { linearUnit: "millimeter", linearDeflectionType: "bounding_box_ratio", linearDeflection: 0.0015, angularDeflection: 0.5 });
  if (!res.success || !res.meshes.length) throw new Error("STEPファイルとして読み込めませんでした");
  progress("Building mesh...", 0.27, 0.28);
  await tick();
  const { mesh, grid, faces } = buildMesh(res);
  progress("Reading STEP entities...", 0.28, null);
  await tick();
  let text;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch (_) { text = new TextDecoder("latin1").decode(bytes); }
  const pmi = JSON.parse(py.globals.get("_extract")(text));
  const items = [];
  for (const key of ["geometric_tolerances", "annotations", "datums"]) for (const it of (pmi[key] || [])) items.push(it);
  for (let i = 0; i < items.length; i++) {
    if (i % 20 === 0) { progress("Matching faces... " + i + "/" + items.length, 0.85 + 0.07 * i / Math.max(1, items.length), null); await tick(); }
    const a = items[i].anchor_points; delete items[i].anchor_points;
    items[i].face_triangle_ranges = matchRanges(a, grid, faces);
  }
  progress("Finalizing...", 0.92, 0.94);
  await tick();
  pmi.geometric_tolerances = JSON.parse(py.globals.get("_decorate")(JSON.stringify(pmi.geometric_tolerances || [])));
  progress("Transferring result...", 0.94, null);
  return { filename, mesh, pmi };
}

onmessage = async (e) => {
  const { kind, buffer, filename } = e.data;
  try {
    const result = await (kind === "qif" ? loadQif : load)(buffer, filename);
    postMessage({ type: "done", result });
  } catch (err) {
    postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
