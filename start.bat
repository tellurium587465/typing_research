@echo off
chcp 65001 > nul

REM ── このファイルがある場所 = プロジェクトルート ─────────────
set PROJECT=%~dp0
if "%PROJECT:~-1%"=="\" set PROJECT=%PROJECT:~0,-1%

set PYTHON=%PROJECT%\.venv310\Scripts\python.exe
set STREAMLIT=%PROJECT%\.venv310\Scripts\streamlit.exe

REM ── プロジェクトフォルダに移動 ───────────────────────────────
cd /d "%PROJECT%"

REM ── コマンド振り分け ────────────────────────────────────────
if "%1"==""        goto APP
if "%1"=="app"     goto APP
if "%1"=="record"  goto RECORD
if "%1"=="analyze" goto ANALYZE
if "%1"=="report"  goto REPORT
if "%1"=="setup"   goto SETUP
goto HELP

:APP
echo.
echo  ============================================
echo   タイピング研究アプリを起動します
echo   ブラウザで http://localhost:8501 が開きます
echo  ============================================
echo.
"%STREAMLIT%" run app.py --server.headless false --browser.gatherUsageStats false --server.port 8501
goto END

:RECORD
echo.
echo [録音モード] recorder.py を起動します
"%PYTHON%" recorder.py %2 %3 %4
goto END

:ANALYZE
echo.
echo [分析モード] run_pipeline.py を起動します
"%PYTHON%" run_pipeline.py %2 %3 %4 %5 %6
goto END

:REPORT
echo.
echo [レポートモード] report.py を起動します
"%PYTHON%" report.py --detail %2 %3
goto END

:SETUP
echo.
echo [セットアップ] setup.bat を起動します
call "%PROJECT%\setup.bat"
goto END

:HELP
echo.
echo  使い方:
echo    start.bat           ^<-- アプリ起動（推奨）
echo    start.bat app       ^<-- アプリ起動
echo    start.bat record    ^<-- 録音のみ（CLI）
echo    start.bat analyze   ^<-- 分析のみ（CLI）
echo    start.bat report    ^<-- レポートのみ（CLI）
echo    start.bat setup     ^<-- セットアップ再実行
echo.

:END
