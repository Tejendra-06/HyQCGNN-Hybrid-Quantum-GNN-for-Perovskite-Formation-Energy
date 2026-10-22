@echo off
rem Starts the web interface at http://localhost:8765 (API docs at /docs).
cd /d "%~dp0"
start "" http://localhost:8765
".venv\Scripts\python.exe" -m uvicorn webapp.server:app --port 8765
