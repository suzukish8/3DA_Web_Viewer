// サーバーなしで STEP を読み込む(ブラウザ内完結)。
//  - 形状のメッシュ化: occt-import-js (OpenCascade の WebAssembly 版)
//  - PMI の解析     : Pyodide 上で backend/step_pmi.py をそのまま実行
//  - 面との対応付け : backend/main.py の match_face_triangle_ranges の移植
// 戻り値は /api/upload の応答と同じ形 { filename, mesh, pmi }。
(function () {
  const OCCT_BASE = "https://cdn.jsdelivr.net/npm/occt-import-js@0.0.23/dist/";
  const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  let occtPromise = null, pyPromise = null;

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = () => rej(new Error("読み込めません: " + src));
      document.head.appendChild(s);
    });
  }
  function getOcct() {
    if (!occtPromise) occtPromise = (async () => {
      await loadScript(OCCT_BASE + "occt-import-js.js");
      return await occtimportjs({ locateFile: (f) => OCCT_BASE + f });
    })();
    return occtPromise;
  }
  function getPy() {
    if (!pyPromise) pyPromise = (async () => {
      await loadScript(PYODIDE_BASE + "pyodide.js");
      const py = await loadPyodide({ indexURL: PYODIDE_BASE });
      for (const f of ["step_pmi.py", "pmi_display.py"]) {
        const r = await fetch("py/" + f);
        if (!r.ok) throw new Error("py/" + f + " が見つかりません");
        py.FS.writeFile(f, await r.text());
      }
      py.runPython([
        "import json, step_pmi, pmi_display",
        "def _extract(text):",
        "    return json.dumps(step_pmi.extract_pmi(text))",
        "def _decorate(js):",
        "    tols = json.loads(js)",
        "    pmi_display.decorate_step_tolerances(tols)",
        "    return json.dumps(tols)",
      ].join("\n"));
      return py;
    })();
    return pyPromise;
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

  async function load(arrayBuffer, filename, onStatus) {
    const say = onStatus || (() => {});
    const bytes = new Uint8Array(arrayBuffer);
    say("Loading CAD engine...");
    const [occt, py] = await Promise.all([getOcct(), getPy()]);
    say("Tessellating...");
    await new Promise(r => setTimeout(r, 30));
    const res = occt.ReadStepFile(bytes, { linearUnit: "millimeter", linearDeflectionType: "bounding_box_ratio", linearDeflection: 0.0015, angularDeflection: 0.5 });
    if (!res.success || !res.meshes.length) throw new Error("STEPファイルとして読み込めませんでした");
    const { mesh, grid, faces } = buildMesh(res);
    say("Parsing PMI...");
    await new Promise(r => setTimeout(r, 30));
    let text;
    try { text = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
    catch (_) { text = new TextDecoder("latin1").decode(bytes); }
    const pmi = JSON.parse(py.globals.get("_extract")(text));
    say("Matching faces...");
    await new Promise(r => setTimeout(r, 30));
    for (const key of ["geometric_tolerances", "annotations", "datums"]) {
      for (const it of (pmi[key] || [])) {
        const a = it.anchor_points; delete it.anchor_points;
        it.face_triangle_ranges = matchRanges(a, grid, faces);
      }
    }
    pmi.geometric_tolerances = JSON.parse(py.globals.get("_decorate")(JSON.stringify(pmi.geometric_tolerances || [])));
    return { filename, mesh, pmi };
  }

  window.WasmStepLoader = { load };
})();
