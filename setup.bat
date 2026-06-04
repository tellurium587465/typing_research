@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

REM ════════════════════════════════════════════════════════════
REM  タイピング研究 セットアップスクリプト
REM
REM  使い方:
REM    1. このフォルダで setup.bat をダブルクリック
REM    2. 完了するとデスクトップにランチャーが作成される
REM    3. 以降は「タイピング研究.bat」をダブルクリックするだけ
REM ════════════════════════════════════════════════════════════

REM ── このファイルがある場所 = プロジェクトルート ─────────────
set PROJECT=%~dp0
REM 末尾の \ を除去
if "%PROJECT:~-1%"=="\" set PROJECT=%PROJECT:~0,-1%

echo.
echo  ============================================
echo   タイピング研究 セットアップ
echo   プロジェクト: %PROJECT%
echo  ============================================
echo.

REM ── Python の確認 ───────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python が見つかりません。
    echo  https://www.python.org からインストールしてください。
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python: %PYVER%

REM ── 仮想環境の作成 ──────────────────────────────────────────
if exist "%PROJECT%\.venv310\Scripts\python.exe" (
    echo  仮想環境: 既存のものを使用します
) else (
    echo  仮想環境を作成中...
    python -m venv "%PROJECT%\.venv310"
    if errorlevel 1 (
        echo [ERROR] 仮想環境の作成に失敗しました
        pause
        exit /b 1
    )
    echo  仮想環境: 作成完了
)

REM ── pip アップグレード ───────────────────────────────────────
echo.
echo  pip をアップグレード中...
"%PROJECT%\.venv310\Scripts\python.exe" -m pip install --upgrade pip --quiet

REM ── 依存パッケージのインストール ─────────────────────────────
echo  パッケージをインストール中（数分かかります）...
"%PROJECT%\.venv310\Scripts\pip.exe" install -r "%PROJECT%\requirements.txt" --quiet
if errorlevel 1 (
    echo [ERROR] パッケージのインストールに失敗しました
    pause
    exit /b 1
)
echo  パッケージ: インストール完了

REM ── MediaPipe モデルのダウンロード ───────────────────────────
if exist "%PROJECT%\hand_landmarker.task" (
    echo  MediaPipe モデル: 既存のものを使用します
) else (
    echo  MediaPipe モデルをダウンロード中...
    "%PROJECT%\.venv310\Scripts\python.exe" -c ^
        "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', r'%PROJECT%\hand_landmarker.task'); print('  完了')"
    if errorlevel 1 (
        echo [WARNING] モデルの自動ダウンロードに失敗しました。
        echo  手動で以下からダウンロードして %PROJECT%\ に置いてください:
        echo  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    ) else (
        echo  MediaPipe モデル: ダウンロード完了
    )
)

REM ── sessions / output フォルダの作成 ─────────────────────────
if not exist "%PROJECT%\sessions" mkdir "%PROJECT%\sessions"
if not exist "%PROJECT%\output"   mkdir "%PROJECT%\output"
echo  データフォルダ: 確認済み

REM ── デスクトップにランチャーを生成 ───────────────────────────
echo.
echo  デスクトップにランチャーを作成中...

set DESKTOP=%USERPROFILE%\Desktop
set LAUNCHER=%DESKTOP%\タイピング研究.bat

REM ランチャーの内容を動的生成（このマシンのパスを埋め込む）
(
    echo @echo off
    echo chcp 65001 ^> nul
    echo.
    echo REM ─ このファイルは setup.bat によって自動生成されました ─
    echo set PROJECT=%PROJECT%
    echo set STREAMLIT=%PROJECT%\.venv310\Scripts\streamlit.exe
    echo.
    echo cd /d "%%PROJECT%%"
    echo.
    echo echo.
    echo echo  ============================================
    echo echo   タイピング研究アプリを起動しています...
    echo echo   ブラウザで http://localhost:8501 が開きます
    echo echo   終了: このウィンドウを閉じてください
    echo echo  ============================================
    echo echo.
    echo.
    echo "%%STREAMLIT%%" run app.py --server.headless false --browser.gatherUsageStats false --server.port 8501
) > "%LAUNCHER%"

echo  ランチャー: %LAUNCHER%

REM ── 完了メッセージ ────────────────────────────────────────────
echo.
echo  ════════════════════════════════════════════════
echo   セットアップ完了！
echo.
echo   デスクトップの「タイピング研究.bat」を
echo   ダブルクリックするとアプリが起動します。
echo  ════════════════════════════════════════════════
echo.
pause
