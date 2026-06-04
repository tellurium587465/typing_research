@echo off
chcp 65001 > nul
cd /d "%~dp0"

REM ─── Python 環境を自動選択 ───────────────────────────────
if exist ".venv310\Scripts\python.exe" (
    set PYTHON=.venv310\Scripts\python.exe
    set STREAMLIT=.venv310\Scripts\streamlit.exe
) else (
    set PYTHON=python
    set STREAMLIT=streamlit
)

REM ─── コマンド振り分け ────────────────────────────────────
if "%1"==""        goto APP
if "%1"=="app"     goto APP
if "%1"=="record"  goto RECORD
if "%1"=="analyze" goto ANALYZE
if "%1"=="report"  goto REPORT
goto HELP

:APP
echo.
echo ============================================
echo  タイピング研究アプリを起動します
echo  ブラウザで http://localhost:8501 が開きます
echo ============================================
echo.
%STREAMLIT% run app.py --server.headless false --browser.gatherUsageStats false
goto END

:RECORD
echo.
echo [録音モード] recorder.py を起動します
%PYTHON% recorder.py %2 %3 %4
goto END

:ANALYZE
echo.
echo [分析モード] run_pipeline.py を起動します
%PYTHON% run_pipeline.py %2 %3 %4 %5 %6
goto END

:REPORT
echo.
echo [レポートモード] report.py を起動します
%PYTHON% report.py --detail %2 %3
goto END

:HELP
echo.
echo  使い方:
echo    start.bat           ^<-- アプリ起動（推奨）
echo    start.bat app       ^<-- アプリ起動
echo    start.bat record    ^<-- 録音のみ（CLIモード）
echo    start.bat analyze   ^<-- 分析のみ（CLIモード）
echo    start.bat report    ^<-- レポートのみ（CLIモード）
echo.

:END
