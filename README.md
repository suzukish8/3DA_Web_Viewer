# 3DA Web Viewer

STEP AP242 (PMI付き) をブラウザで表示する 3DA ビューア。FastAPI + OpenCascade (cadquery-ocp) で STEP を読み込み、three.js で表示します。

- 形状・幾何公差・寸法・データムを表示し、特性リストと 3D 表示を相互にハイライト
- Balloon & Mirror スタイル(公差記入枠のバルーン配置、Hidden-Side Mirror)

## 起動

```
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --port 8765
```

ブラウザで http://localhost:8765/ を開き、STEP ファイルをドラッグ&ドロップします(Windows では `start_server.bat` でも起動できます)。

## 注意

サンプルの STEP/QIF ファイルは含まれていません。各自のファイルを読み込んでください。
