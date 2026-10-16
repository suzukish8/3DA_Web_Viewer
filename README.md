# 3DA Web Viewer

STEP AP242 (PMI付き) をブラウザで表示する 3DA ビューア。FastAPI + OpenCascade (cadquery-ocp) で STEP を読み込み、three.js で表示します。

- 形状・幾何公差・寸法・データムを表示し、特性リストと 3D 表示を相互にハイライト
- Balloon & Mirror スタイル(公差記入枠のバルーン配置、Hidden-Side Mirror)

## インストール不要で使う(GitHub Pages)

https://suzukish8.github.io/3DA_Web_Viewer/ をブラウザで開き、STEP ファイルをドラッグ&ドロップします。
STEP の解析はすべてブラウザ内で行われ、ファイルは外部に送信されません(初回のみ、解析エンジンを CDN から読み込みます)。
QIF ファイル(.qif)も同様にブラウザ内で読み込めます(Pyodide 上でパーサを実行するため、初回は数秒かかります)。

## サーバー版の起動

```
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --port 8765
```

ブラウザで http://localhost:8765/ を開き、STEP ファイルをドラッグ&ドロップします(Windows では `start_server.bat` でも起動できます)。

## 注意

サンプルの STEP/QIF ファイルは含まれていません。各自のファイルを読み込んでください。
