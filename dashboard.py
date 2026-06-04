"""
dashboard.py  ─ Streamlit Web ダッシュボード（⑦）

ブラウザ上でセッションを選択しながらグラフを確認できる。

起動方法:
  .venv310\Scripts\streamlit run dashboard.py

依存:
  pip install streamlit plotly
"""
import json
import glob
import os
import re
from collections import defaultdict

import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Streamlit / Plotly が必要です:")
    print("  .venv310\\Scripts\\pip install streamlit plotly")
    raise

# ── ページ設定 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="タイピング研究ダッシュボード",
    page_icon="⌨️",
    layout="wide",
)

PHRASE_MS = 1000
FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
}

# ── データ読み込み ─────────────────────────────────────────────
@st.cache_data
def load_all_sessions():
    sessions = {}
    for kf in sorted(glob.glob("session_*_keys.json")):
        sid = re.search(r"session_(\d+)_keys", kf).group(1)
        with open(kf, encoding="utf-8") as f:
            keys = json.load(f)

        # meta
        meta = {}
        mf = f"session_{sid}_meta.json"
        if os.path.exists(mf):
            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)

        # integrated
        integ = None
        inf = f"session_{sid}_integrated.json"
        if os.path.exists(inf):
            with open(inf, encoding="utf-8") as f:
                integ = json.load(f)

        sessions[sid] = {"keys": keys, "meta": meta, "integrated": integ}
    return sessions

sessions = load_all_sessions()

def session_label(sid, meta):
    lbl = meta.get("label", "")
    cat = meta.get("category", "")
    parts = [sid[-6:]]
    if lbl: parts.append(lbl)
    if cat:  parts.append(f"[{cat}]")
    return " | ".join(parts)

# ── サイドバー ─────────────────────────────────────────────────
st.sidebar.title("⌨️ タイピング研究")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "表示モード",
    ["📈 全体トレンド", "🔍 セッション詳細", "🖐️ 運指分析", "⚠️ エラー分析"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📈 全体トレンド
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if mode == "📈 全体トレンド":
    st.title("📈 全セッション成長トレンド")

    # データ集計
    rows = []
    for sid, data in sessions.items():
        keys   = data["keys"]
        meta   = data["meta"]
        integ  = data["integrated"]
        normal = [k for k in keys if not k.get("is_backspace")
                  and not k["key"].startswith("Key.")]
        errors = [k for k in normal if k.get("is_error")]
        ivs    = [k["interval_ms"] for k in normal
                  if 0 < k["interval_ms"] <= PHRASE_MS]
        pure_ms = (keys[-1]["timestamp_ms"] - keys[0]["timestamp_ms"]) - \
                  sum(k["interval_ms"] for k in normal if k["interval_ms"] > PHRASE_MS) \
                  if keys else 1
        wpm = (len(normal) / 5) / (pure_ms / 60000) if pure_ms > 0 else 0

        cam_rate  = fin_rate = key_rate = None
        if integ:
            cam = [r for r in integ if r["source"] == "camera"]
            key_ok = sum(1 for r in cam if r.get("key_match"))
            def is_fin(r):
                af = r.get("actual_finger")
                return af is not None and af == r.get("std_finger")
            fin_ok = sum(1 for r in cam if is_fin(r))
            cam_rate = len(cam) / max(len(integ), 1) * 100
            key_rate = key_ok  / max(len(cam),   1) * 100
            fin_rate = fin_ok  / max(len(cam),   1) * 100

        rows.append({
            "sid":       sid,
            "label":     session_label(sid, meta),
            "n_keys":    len(normal),
            "wpm":       round(wpm, 1),
            "avg_iv":    round(np.mean(ivs), 1) if ivs else 0,
            "error_rate": round(len(errors) / max(len(normal), 1) * 100, 2),
            "cam_rate":  cam_rate,
            "key_rate":  key_rate,
            "fin_rate":  fin_rate,
            "date":      meta.get("recorded_at", sid)[:10],
        })

    # メトリクスカード
    if rows:
        latest = rows[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新WPM",       f"{latest['wpm']:.0f}",
                  delta=f"{latest['wpm'] - rows[0]['wpm']:+.0f}" if len(rows) > 1 else None)
        c2.metric("平均間隔",      f"{latest['avg_iv']:.0f}ms",
                  delta=f"{latest['avg_iv'] - rows[0]['avg_iv']:+.0f}ms" if len(rows) > 1 else None,
                  delta_color="inverse")
        c3.metric("ミス率",        f"{latest['error_rate']:.1f}%")
        c4.metric("運指一致率",    f"{latest['fin_rate']:.0f}%" if latest['fin_rate'] is not None else "N/A")

    st.markdown("---")

    # WPM トレンドグラフ
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["WPM", "打鍵間隔 (ms)", "キー一致率 / 運指一致率", "ミス率"])

    labels  = [r["label"] for r in rows]
    wpms    = [r["wpm"]    for r in rows]
    avg_ivs = [r["avg_iv"] for r in rows]

    fig.add_trace(go.Scatter(x=labels, y=wpms, mode="lines+markers+text",
                             text=[f"{v:.0f}" for v in wpms],
                             textposition="top center",
                             line=dict(color="#2196F3", width=2),
                             marker=dict(size=8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=labels, y=avg_ivs, mode="lines+markers+text",
                             text=[f"{v:.0f}" for v in avg_ivs],
                             textposition="top center",
                             line=dict(color="#FF9800", width=2),
                             marker=dict(size=8)), row=1, col=2)

    key_rates = [r["key_rate"] for r in rows if r["key_rate"] is not None]
    fin_rates = [r["fin_rate"] for r in rows if r["fin_rate"] is not None]
    lbls_k    = [r["label"]   for r in rows if r["key_rate"] is not None]
    if key_rates:
        fig.add_trace(go.Bar(x=lbls_k, y=key_rates, name="キー一致率",
                             marker_color="#4CAF50"), row=2, col=1)
        fig.add_trace(go.Bar(x=lbls_k, y=fin_rates, name="運指一致率",
                             marker_color="#9C27B0"), row=2, col=1)

    errs  = [r["error_rate"] for r in rows]
    fig.add_trace(go.Bar(x=labels, y=errs,
                         marker_color=["#F44336" if e > 0 else "#4CAF50" for e in errs]),
                  row=2, col=2)

    fig.update_layout(height=600, showlegend=True,
                      title_text="セッション間の成長トレンド")
    st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 セッション詳細
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "🔍 セッション詳細":
    st.title("🔍 セッション詳細")

    sid = st.selectbox(
        "セッションを選択",
        list(sessions.keys()),
        format_func=lambda s: session_label(s, sessions[s]["meta"])
    )

    data   = sessions[sid]
    keys   = data["keys"]
    meta   = data["meta"]
    normal = [k for k in keys if not k.get("is_backspace")
              and not k["key"].startswith("Key.")]
    ivs    = [k["interval_ms"] for k in normal if 0 < k["interval_ms"] <= PHRASE_MS]

    # 基本情報
    col1, col2, col3 = st.columns(3)
    col1.metric("打鍵数",    len(normal))
    col2.metric("録画時間",  f"{meta.get('video_duration_s', 0):.0f}秒")
    col3.metric("ラベル",    meta.get("label", "なし"))

    # interval の時系列
    fig_ts = go.Figure()
    ts = [k["timestamp_ms"] / 1000 for k in normal if 0 < k["interval_ms"] <= PHRASE_MS]
    fig_ts.add_trace(go.Scatter(
        x=ts, y=ivs, mode="lines", opacity=0.7,
        line=dict(color="#2196F3", width=1), name="interval"
    ))
    fig_ts.add_hline(y=np.mean(ivs), line_dash="dash",
                     line_color="red", annotation_text="平均")
    for k in normal:
        if k.get("is_error"):
            fig_ts.add_vline(x=k["timestamp_ms"]/1000,
                             line_color="red", line_width=2)
    fig_ts.update_layout(title="打鍵間隔の時系列（赤縦線=エラー）",
                          xaxis_title="経過時間 (s)", yaxis_title="interval (ms)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # キー頻度ヒートマップ
    key_counts = defaultdict(int)
    for k in normal:
        if len(k["key"]) == 1 and k["key"].isalpha():
            key_counts[k["key"].lower()] += 1

    rows_layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    z_data, x_data, y_data = [], [], []
    for row_s in rows_layout:
        y_data.append(row_s[0].upper() + "行")
        z_data.append([key_counts.get(c, 0) for c in row_s])
        x_data = list(range(len(rows_layout[0])))

    fig_hm = go.Figure(data=go.Heatmap(
        z=z_data, colorscale="Blues",
        text=[[f"{rows_layout[ri][ci].upper()}: {key_counts.get(rows_layout[ri][ci], 0)}"
               for ci in range(len(rows_layout[ri]))]
              for ri in range(len(rows_layout))],
        texttemplate="%{text}",
    ))
    fig_hm.update_layout(title="キー使用頻度ヒートマップ",
                          yaxis=dict(tickvals=[0,1,2], ticktext=y_data))
    st.plotly_chart(fig_hm, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🖐️ 運指分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "🖐️ 運指分析":
    st.title("🖐️ 運指分析")

    finger_total   = defaultdict(int)
    finger_correct = defaultdict(int)
    key_finger_ivs = defaultdict(lambda: defaultdict(list))

    for sid, data in sessions.items():
        integ = data.get("integrated")
        if not integ:
            continue
        for r in integ:
            if r["source"] != "camera" or not r.get("actual_finger"):
                continue
            af = r["actual_finger"]
            finger_total[af] += 1
            def is_fin(rec):
                a = rec.get("actual_finger")
                return a is not None and a == rec.get("std_finger")
            if is_fin(r):
                finger_correct[af] += 1
            if len(r["key"]) == 1 and r["key"].isalpha() and 0 < r["interval_ms"] <= PHRASE_MS:
                key_finger_ivs[r["key"]][af].append(r["interval_ms"])

    # 運指一致率バー
    finger_order = ["L1","L2","L3","L4","R4","R3","R2","R1"]
    fnames  = [FINGER_NAMES.get(f, f) for f in finger_order]
    totals  = [finger_total.get(f, 0)  for f in finger_order]
    corrects= [finger_correct.get(f, 0) for f in finger_order]
    rates   = [c/max(t,1)*100 for c,t in zip(corrects, totals)]

    fig_f = make_subplots(rows=1, cols=2, subplot_titles=["運指一致率", "使用回数"])
    fig_f.add_trace(go.Bar(x=fnames, y=rates, marker_color="#9C27B0",
                            text=[f"{r:.0f}%" for r in rates],
                            textposition="outside"), row=1, col=1)
    fig_f.add_trace(go.Bar(x=fnames, y=totals, marker_color="#2196F3",
                            text=totals, textposition="outside"), row=1, col=2)
    fig_f.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_f, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠️ エラー分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "⚠️ エラー分析":
    st.title("⚠️ エラー誘発パターン")
    st.info("詳細な分析は `python error_analysis.py` を実行してください。")

    all_errors = []
    for sid, data in sessions.items():
        keys   = data["keys"]
        normal = [k for k in keys if not k["key"].startswith("Key.")]
        baselines = [k["interval_ms"] for k in normal
                     if 0 < k["interval_ms"] <= PHRASE_MS and not k.get("is_error")]
        bm = np.mean(baselines) if baselines else 1
        for k in normal:
            if k.get("is_error"):
                all_errors.append({
                    "session": sid[-6:],
                    "key":     k["key"],
                    "interval_ms":  k["interval_ms"],
                    "ratio_to_base": k["interval_ms"] / bm,
                    "baseline_mean": bm,
                })

    if not all_errors:
        st.warning("エラーデータがありません。ミス後にバックスペースを押したセッションが必要です。")
    else:
        for e in all_errors:
            pct = e["ratio_to_base"] * 100
            rushing = "⚡急加速" if pct < 60 else ""
            st.markdown(
                f"**session {e['session']}** | key=[**{e['key'].upper()}**] | "
                f"interval={e['interval_ms']}ms | baseline比={pct:.0f}% {rushing}"
            )

        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(
            x=[e["ratio_to_base"] for e in all_errors],
            nbinsx=20, marker_color="#F44336", name="エラー時のratio"
        ))
        fig_e.add_vline(x=1.0, line_dash="dash", line_color="blue",
                         annotation_text="baseline")
        fig_e.update_layout(
            title="エラー時の interval / baseline 分布",
            xaxis_title="interval / baseline mean",
            yaxis_title="件数"
        )
        st.plotly_chart(fig_e, use_container_width=True)
        st.markdown(
            "**傾向**: ratio < 0.6（baseline の60%以下）で急加速しているときにエラーが多い"
        )
