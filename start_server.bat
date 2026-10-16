@echo off
rem 3DA Web Viewer backend launcher (http://localhost:8765)
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8765
pause
