@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 goto :error
)
if not exist ".venv\surface-lab-ready" (
    echo Premiere installation des dependances. Connexion Internet requise.
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    type nul > ".venv\surface-lab-ready"
)
".venv\Scripts\python.exe" run.py
if errorlevel 1 goto :error
exit /b 0
:error
echo Le demarrage a echoue. Verifiez Python 3.11 a 3.14 et les messages ci-dessus.
pause
exit /b 1
