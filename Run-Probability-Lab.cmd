@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "GS_CACHE_FILE_PATH=%~dp0.cache\genesis"
set "QD_OFFLINE_CACHE_FILE_PATH=%~dp0.cache\quadrants"
set "NUMBA_CACHE_DIR=%~dp0.cache\numba"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo The Genesis virtual environment is missing. See examples\probability\README.md.
    pause
    exit /b 1
)
if not exist "%~dp0out\probability" mkdir "%~dp0out\probability"
echo Opening Genesis Probability Lab. First launch may need a minute to compile.
echo Close the interface window to stop. Data is saved in out\probability.
"%~dp0.venv\Scripts\python.exe" -u -m examples.probability.app -g > "%~dp0out\probability\launcher.log" 2>&1
if errorlevel 1 (
    type "%~dp0out\probability\launcher.log"
    pause
    exit /b 1
)
endlocal
