"""
session_utils.py  ─ セッション管理ユーティリティ

ディレクトリ構造:
  sessions/
    {session_id}/          ← 各セッションのデータ
      keys.json
      audio.wav
      video.avi
      pose.json
      onset_matched.json
      integrated.json
      keyboard_grid.json
      meta.json
      frame.png
      keyboard_detect.png
      onset_result.png
      typing_analysis.xlsx
      jasp_trigram.csv
  output/                  ← 横断分析の出力
    trend.png
    weakness_heatmap.png
    arpeggio_map.json
    ...
  arpeggio_map.json        ← ルートにも置く（後方互換）

旧構造(session_*_keys.json)との後方互換あり。
"""
import glob
import json
import os
import re
import shutil

from constants import SESSIONS_DIR, OUTPUT_DIR, PROJECT_ROOT

# ── カレントディレクトリをプロジェクトルートに固定 ──────────────
# デスクトップや別フォルダから実行しても正しいパスを参照できるようにする
os.chdir(PROJECT_ROOT)

# ── ディレクトリ確認・作成 ──────────────────────────────────────
def ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,   exist_ok=True)

ensure_dirs()

# ── セッションディレクトリ ──────────────────────────────────────
def get_session_dir(session_id: str) -> str:
    """sessions/{session_id}/ のパスを返す（作成もする）"""
    d = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(d, exist_ok=True)
    return d

# ── セッション検索（新旧両構造対応） ──────────────────────────────
def list_all_session_ids() -> list[str]:
    """
    全セッションIDをリストで返す（新旧両構造に対応）。
    新構造: sessions/{id}/keys.json
    旧構造: session_{id}_keys.json（ルートに散在）
    """
    ids = set()

    # 新構造
    for p in glob.glob(os.path.join(SESSIONS_DIR, "*", "keys.json")):
        sid = os.path.basename(os.path.dirname(p))
        ids.add(sid)

    # 旧構造（後方互換）─ PROJECT_ROOT 直下を検索
    for p in glob.glob(os.path.join(PROJECT_ROOT, "session_*_keys.json")):
        m = re.search(r"session_(\d+)_keys", p)
        if m:
            ids.add(m.group(1))

    return sorted(ids)

def get_latest_session() -> str:
    """最新のセッションIDを返す"""
    ids = list_all_session_ids()
    if not ids:
        raise FileNotFoundError("セッションファイルが見つかりません")
    return max(ids)  # タイムスタンプ = 数値なので max で最新を取得

def session_id_from_path(path: str) -> str:
    """新旧どちらの構造のパスからでもセッションIDを抽出する"""
    # 新構造: …/sessions/{id}/keys.json
    # SESSIONS_DIR は絶対パスなのでパス内の sessions フォルダ名で判定
    norm = os.path.normpath(os.path.abspath(path))
    sessions_norm = os.path.normpath(SESSIONS_DIR)
    if norm.startswith(sessions_norm + os.sep):
        # sessions/ の直下のフォルダ名 = session_id
        rel = os.path.relpath(norm, sessions_norm)
        sid = rel.split(os.sep)[0]
        if sid.isdigit():
            return sid
    # 旧構造: session_{id}_keys.json
    m = re.search(r"session_(\d+)", path)
    if m:
        return m.group(1)
    raise ValueError(f"セッションIDを抽出できません: {path}")

def find_session_key_files() -> list[str]:
    """全セッションの keys.json パスを返す（新旧両構造対応）"""
    result = []
    for sid in list_all_session_ids():
        sf = get_session_files(sid)
        if os.path.exists(sf["keys"]):
            result.append(sf["keys"])
    return sorted(result)

def find_session_integrated_files() -> list[str]:
    """全セッションの integrated.json パスを返す（新旧両構造対応）"""
    result = []
    for sid in list_all_session_ids():
        sf = get_session_files(sid)
        if os.path.exists(sf["integrated"]):
            result.append(sf["integrated"])
    return sorted(result)

# ── セッションファイルパス ──────────────────────────────────────
def get_session_files(session_id: str = None) -> dict:
    """
    セッションIDに対応するすべてのファイルパスを返す。
    新構造のパスを返すが、旧構造のファイルが存在すればそちらも示す。
    """
    if session_id is None:
        session_id = get_latest_session()

    session_dir = get_session_dir(session_id)

    def p(filename):
        """セッションディレクトリ内のパスを返す"""
        return os.path.join(session_dir, filename)

    def p_fallback(new_path, old_path):
        """新パスを返す。旧パスのファイルが存在し新パスがなければ旧パスを返す。"""
        if not os.path.exists(new_path) and os.path.exists(old_path):
            return old_path
        return new_path

    files = {
        "session_id":    session_id,
        "session_dir":   session_dir,
        # データファイル
        "keys":          p_fallback(p("keys.json"),           f"session_{session_id}_keys.json"),
        "audio":         p_fallback(p("audio.wav"),           f"session_{session_id}_audio.wav"),
        "video":         p_fallback(p("video.avi"),           f"session_{session_id}_video.avi"),
        "pose":          p_fallback(p("pose.json"),           f"session_{session_id}_pose.json"),
        "onset":         p_fallback(p("onset_matched.json"),  f"session_{session_id}_onset_matched.json"),
        "integrated":    p_fallback(p("integrated.json"),     f"session_{session_id}_integrated.json"),
        "keyboard_grid": p_fallback(p("keyboard_grid.json"),  f"session_{session_id}_keyboard_grid.json"),
        "meta":          p_fallback(p("meta.json"),           f"session_{session_id}_meta.json"),
        # セッション内出力
        "frame":             p("frame.png"),
        "keyboard_detect_img": p("keyboard_detect.png"),
        "onset_result_img":    p("onset_result.png"),
        "typing_analysis_xlsx": p("typing_analysis.xlsx"),
        "jasp_trigram_csv":    p("jasp_trigram.csv"),
        # 横断分析出力（output/ ディレクトリ）
        "arpeggio_map":  os.path.join(OUTPUT_DIR, "arpeggio_map.json"),
    }
    return files

# ── メタデータのロード ──────────────────────────────────────────
def load_meta(session_id: str) -> dict:
    sf = get_session_files(session_id)
    if os.path.exists(sf["meta"]):
        with open(sf["meta"], encoding="utf-8") as f:
            return json.load(f)
    return {"session_id": session_id, "label": "", "category": ""}

# ── 出力パス ──────────────────────────────────────────────────
def output_path(filename: str) -> str:
    """output/ ディレクトリのパスを返す"""
    return os.path.join(OUTPUT_DIR, filename)

# ── CLI 確認用 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"全セッション ({len(list_all_session_ids())}件):")
    for sid in list_all_session_ids():
        sf = get_session_files(sid)
        meta = load_meta(sid)
        label = meta.get("label", "")
        has = {k: "OK" if os.path.exists(v) else "--"
               for k, v in sf.items()
               if k not in ("session_id", "session_dir")}
        label_str = f"  [{label}]" if label else ""
        print(f"\n  {sid}{label_str}")
        for k, mark in has.items():
            print(f"    {mark} {k}: {sf[k]}")
