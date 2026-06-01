"""
plot_utils.py  ─ matplotlib の日本語フォント設定ユーティリティ

使い方:
  from plot_utils import setup_jp_font
  setup_jp_font()   # どのスクリプトでも冒頭に呼ぶだけ
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_jp_font():
    """Windows/Mac/Linux で日本語が表示できるフォントを自動設定"""
    candidates = [
        "Yu Gothic",        # Windows
        "Meiryo",           # Windows
        "BIZ UDGothic",     # Windows
        "Hiragino Sans",    # macOS
        "Hiragino Kaku Gothic Pro",
        "IPAexGothic",      # Linux
        "Noto Sans CJK JP", # Linux
        "DejaVu Sans",      # フォールバック（日本語非対応だが警告を抑制）
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return font

    # フォント名で直接指定できなければパスで試みる
    path_candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/BIZ-UDGothicR.ttc",
    ]
    import os
    for path in path_candidates:
        if os.path.exists(path):
            fe = fm.FontEntry(fname=path, name="_jp_font")
            fm.fontManager.ttflist.append(fe)
            plt.rcParams["font.family"] = "_jp_font"
            return path

    return None  # 設定できなかった
