// サーバーなしで STEP / QIF を読み込む(ブラウザ内完結)。実処理は wasm_worker.js(Web Worker)で行い、
// ここでは進捗メッセージを受け取って onStatus(label, p, to) に渡す。
// 戻り値は /api/upload の応答と同じ形 { filename, mesh, pmi }。
(function () {
  function run(kind, arrayBuffer, filename, onStatus) {
    return new Promise((resolve, reject) => {
      let w;
      try { w = new Worker("wasm_worker.js"); } catch (e) {
        const hint = location.protocol === "file:" ? "

ファイル(file://)から開いた画面では、ブラウザ内解析は使えません。サーバー(start_server.bat)が起動していて、応答できる状態か確認してください。" : "";
        reject(new Error("Web Worker を起動できません: " + e.message + hint)); return;
      }
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { if (onStatus) onStatus(m.label, m.p, m.to); }
        else if (m.type === "done") { w.terminate(); resolve(m.result); }
        else if (m.type === "error") { w.terminate(); reject(new Error(m.message)); }
      };
      w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "ワーカーでエラーが発生しました")); };
      w.postMessage({ kind, buffer: arrayBuffer, filename }, [arrayBuffer]);
    });
  }
  window.WasmStepLoader = {
    load: (buf, name, cb) => run("step", buf, name, cb),
    loadQif: (buf, name, cb) => run("qif", buf, name, cb),
  };
})();
