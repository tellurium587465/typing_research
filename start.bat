@echo off
chcp 65001 > nul
cd /d "%~dp0"

REM ─── Python 環境を自動選択 ───────────────────────────────
if exist ".venv310\Scripts\python.exe" (
    set PYTHON=.venv310\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM ─── コマンド振り分け ────────────────────────────────────
if "%1"==""        goto HELP
if "%1"=="record"  goto RECORD
if "%1"=="analyze" goto ANALYZE
if "%1"=="report"  goto REPORT
goto HELP

:RECORD
echo.
echo ============================================
echo  [録音] recorder.py を起動します
echo  タイピングを開始 → ESC で終了
echo ============================================
echo.
%PYTHON% recorder.py
goto END

:ANALYZE
echo.
echo ============================================
echo  [分析] run_pipeline.py を起動します
echo ============================================
echo.
%PYTHON% run_pipeline.py %2 %3 %4 %5 %6
goto END

:REPORT
echo.
echo ============================================
echo  [レポート] report.py を起動します
echo ============================================
echo.
%PYTHON% report.py --detail %2 %3
goto END

:HELP
echo.
echo  使い方:
echo    start.bat record               ^<-- タイピングを録音（ESCで終了）
echo    start.bat analyze              ^<-- 最新セッションを分析（全工程）
echo    start.bat analyze --from onset ^<-- onsetから再実行
echo    start.bat analyze --skip-pose  ^<-- pose をスキップして高速化
echo    start.bat report               ^<-- 全セッション結果を表示
echo    start.bat report --detail      ^<-- キー別詳細も表示
echo.
echo  通常の流れ:
echo    1. start.bat record   ... タイピングを録音
echo    2. start.bat analyze  ... 自動で全分析（数分かかる）
echo    3. start.bat report   ... 結果を確認
echo.

:END
