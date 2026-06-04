"""
app.py  ─ タイピング研究 ローカルアプリ

起動方法:
  .venv310\\Scripts\\streamlit run app.py

機能:
  📋 ホーム    ─ 最新セッションの概要と次のアクション
  🎙️ 録音      ─ ラベル選択 → recorder.py を別プロセス起動
  ⚙️  分析      ─ パイプライン実行（リアルタイム進捗表示）
  📊 結果      ─ セッションブラウザ・グラフ・比較
  ✋ 運指      ─ 運指最適化の推奨とアルペジオマップ
"""
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from constants import (FINGER_NAMES, FINGER_ORDER, PHRASE_BOUNDARY_MS,
                        SESSIONS_DIR, OUTPUT_DIR, STANDARD_FINGER)
from session_utils import (get_session_files, list_all_session_ids,
                            load_meta, output_path, session_id_from_path)

# ── ページ設定 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="タイピング研究",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PYTHON = str(Path(".venv310/Scripts/python.exe").resolve())

# ── ラベル設定ファイル ──────────────────────────────────────────
LABELS_FILE = "labels_config.json"

def load_labels_config():
    default = {
        "presets": ["寿司打", "e-typing", "英単語", "コード入力", "自由練習"],
        "recent": [],
    }
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, encoding="utf-8") as f:
            return {**default, **json.load(f)}
    return default

def save_labels_config(cfg):
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def add_recent_label(label: str):
    cfg = load_labels_config()
    if label and label not in cfg["recent"]:
        cfg["recent"] = [label] + cfg["recent"][:9]  # 最大10件
        save_labels_config(cfg)

# ── データ読み込み（キャッシュ付き） ───────────────────────────────
@st.cache_data(ttl=10)  # 10秒でキャッシュ更新
def load_session_data(session_id):
    sf   = get_session_files(session_id)
    meta = load_meta(session_id)
    keys, integ, onset = [], None, None

    if os.path.exists(sf["keys"]):
        with open(sf["keys"], encoding="utf-8") as f:
            keys = json.load(f)
    if os.path.exists(sf["integrated"]):
        with open(sf["integrated"], encoding="utf-8") as f:
            integ = json.load(f)
    if os.path.exists(sf["onset"]):
        with open(sf["onset"], encoding="utf-8") as f:
            onset = json.load(f)
    return {"keys": keys, "integrated": integ, "onset": onset, "meta": meta, "sf": sf}

def compute_session_stats(data):
    keys  = data["keys"]
    integ = data["integrated"]
    meta  = data["meta"]

    normal = [k for k in keys
              if not k.get("is_backspace") and not k["key"].startswith("Key.")]
    errors = [k for k in normal if k.get("is_error")]
    ivs    = [k["interval_ms"] for k in normal if 0 < k["interval_ms"] <= PHRASE_BOUNDARY_MS]
    wait   = sum(k["interval_ms"] for k in normal if k["interval_ms"] > PHRASE_BOUNDARY_MS)
    pure_ms = (keys[-1]["timestamp_ms"] - keys[0]["timestamp_ms"] - wait) if keys else 1

    stats = {
        "n_keys":     len(normal),
        "error_rate": len(errors) / max(len(normal), 1) * 100,
        "wpm":        (len(normal) / 5) / (pure_ms / 60000) if pure_ms > 0 else 0,
        "avg_iv":     np.mean(ivs) if ivs else 0,
        "median_iv":  np.median(ivs) if ivs else 0,
        "label":      meta.get("label", ""),
        "category":   meta.get("category", ""),
        "recorded_at": meta.get("recorded_at", ""),
    }
    if integ:
        cam    = [r for r in integ if r["source"] == "camera"]
        key_ok = sum(1 for r in cam if r.get("key_match"))
        def is_fin(r):
            af = r.get("actual_finger")
            return af is not None and af == r.get("std_finger")
        fin_ok = sum(1 for r in cam if is_fin(r))
        stats["cam_rate"]  = len(cam) / max(len(integ), 1) * 100
        stats["key_match"] = key_ok / max(len(cam), 1) * 100
        stats["fin_match"] = fin_ok / max(len(cam), 1) * 100
    return stats

# ── サイドバー ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⌨️ タイピング研究")
    st.markdown("---")
    page = st.radio(
        "ページ",
        ["📋 ホーム", "🎙️ 録音", "⚙️ 分析", "📊 結果", "✋ 運指"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    session_ids = list_all_session_ids()
    st.caption(f"セッション数: {len(session_ids)}")
    if session_ids:
        latest_meta = load_meta(session_ids[-1])
        lbl = latest_meta.get("label", "")
        st.caption(f"最新: {session_ids[-1][-6:]}" + (f" [{lbl}]" if lbl else ""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 ホーム
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "📋 ホーム":
    st.title("⌨️ タイピング研究 ダッシュボード")

    if not session_ids:
        st.info("セッションがまだありません。「🎙️ 録音」から始めましょう。")
    else:
        # 最新3セッションのカード
        st.subheader("最近のセッション")
        recent = session_ids[-3:][::-1]
        cols = st.columns(len(recent))
        for col, sid in zip(cols, recent):
            data  = load_session_data(sid)
            stats = compute_session_stats(data)
            meta  = data["meta"]
            with col:
                label_str = meta.get("label", "") or "ラベルなし"
                st.metric(label_str, f"{stats['wpm']:.0f} WPM")
                st.caption(f"打鍵: {stats['n_keys']}  ミス: {stats['error_rate']:.1f}%")
                st.caption(f"運指一致: {stats.get('fin_match', 0):.0f}%")

        st.markdown("---")

        # WPM トレンド（全セッション）
        st.subheader("WPM の推移")
        all_stats = []
        for sid in session_ids:
            try:
                d = load_session_data(sid)
                s = compute_session_stats(d)
                s["sid"] = sid
                all_stats.append(s)
            except Exception:
                pass

        if len(all_stats) >= 2:
            labels_x = [f"S{i+1}" + (f"\n{s['label'][:6]}" if s["label"] else "")
                        for i, s in enumerate(all_stats)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels_x, y=[s["wpm"] for s in all_stats],
                mode="lines+markers+text",
                text=[f"{s['wpm']:.0f}" for s in all_stats],
                textposition="top center",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=8),
            ))
            fig.update_layout(height=250, margin=dict(t=20, b=20),
                               yaxis_title="WPM")
            st.plotly_chart(fig, use_container_width=True)

        # クイックアクション
        st.markdown("---")
        st.subheader("クイックアクション")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎙️ 録音を始める", use_container_width=True):
                st.session_state["goto"] = "🎙️ 録音"
                st.rerun()
        with c2:
            if st.button("⚙️ 最新を分析", use_container_width=True):
                st.session_state["goto"] = "⚙️ 分析"
                st.rerun()
        with c3:
            if st.button("📊 結果を見る", use_container_width=True):
                st.session_state["goto"] = "📊 結果"
                st.rerun()

    if "goto" in st.session_state:
        page = st.session_state.pop("goto")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎙️ 録音
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🎙️ 録音":
    st.title("🎙️ 録音")
    st.info("カメラ・マイク・キーロガーを同時録音します。終了は **ESC キー** です。")

    cfg = load_labels_config()

    # ── ラベル選択 ──────────────────────────────────────────────
    st.subheader("① ラベルを選択")

    # プリセットボタン
    st.caption("プリセット")
    preset_cols = st.columns(len(cfg["presets"]))
    selected_preset = st.session_state.get("selected_preset", "")
    for col, preset in zip(preset_cols, cfg["presets"]):
        is_selected = (selected_preset == preset)
        btn_style = "primary" if is_selected else "secondary"
        if col.button(preset, key=f"preset_{preset}", type=btn_style,
                      use_container_width=True):
            st.session_state["selected_preset"] = preset
            st.session_state["custom_label"] = ""
            st.rerun()

    # 最近使ったラベル
    if cfg["recent"]:
        st.caption("最近使ったラベル")
        recent_cols = st.columns(min(5, len(cfg["recent"])))
        for col, lbl in zip(recent_cols, cfg["recent"][:5]):
            if col.button(lbl, key=f"recent_{lbl}", use_container_width=True):
                st.session_state["selected_preset"] = lbl
                st.session_state["custom_label"] = ""
                st.rerun()

    # カスタム入力
    st.caption("カスタムラベル（自由入力）")
    custom = st.text_input("", value=st.session_state.get("custom_label", ""),
                            placeholder="例: 寿司打5000点挑戦",
                            label_visibility="collapsed",
                            key="custom_label_input")
    if custom:
        st.session_state["selected_preset"] = ""
        st.session_state["custom_label"] = custom

    # 決定されたラベル
    final_label = custom if custom else st.session_state.get("selected_preset", "")
    if final_label:
        st.success(f"ラベル: **{final_label}**")
    else:
        st.warning("ラベルなしで録音します（後から変更できません）")

    # ── カテゴリ選択 ──────────────────────────────────────────
    st.subheader("② カテゴリ（任意）")
    category = st.selectbox(
        "",
        ["", "sushida", "etyping", "english", "code", "free"],
        format_func=lambda x: {
            "": "選択なし", "sushida": "寿司打",
            "etyping": "e-typing", "english": "英単語",
            "code": "コード入力", "free": "自由練習",
        }.get(x, x),
        label_visibility="collapsed",
    )

    # ── 録音開始 ──────────────────────────────────────────────
    st.subheader("③ 録音開始")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_btn = st.button("🔴 録音開始", type="primary",
                               use_container_width=True, key="start_record")
    with col_info:
        st.caption("クリックするとカメラプレビューが開きます")
        st.caption("ESC キーで録音を終了します")

    if start_btn:
        if final_label:
            add_recent_label(final_label)
        cmd = [PYTHON, "recorder.py"]
        if final_label:
            cmd += ["--label", final_label]
        if category:
            cmd += ["--category", category]

        # Windows: 新しいウィンドウで起動
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        st.success(f"recorder.py を起動しました（ラベル: {final_label or 'なし'}）")
        st.info("カメラウィンドウが開いたらタイピングを開始してください。\n"
                "終了後、「⚙️ 分析」ページで分析を実行してください。")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ 分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚙️ 分析":
    st.title("⚙️ 分析パイプライン")

    session_ids = list_all_session_ids()
    if not session_ids:
        st.warning("セッションがありません")
        st.stop()

    # セッション選択
    def sid_label(sid):
        meta = load_meta(sid)
        lbl  = meta.get("label", "")
        date = meta.get("recorded_at", "")[:10]
        return f"{sid[-8:]}  {date}  {('['+lbl+']') if lbl else ''}"

    selected_sid = st.selectbox(
        "分析するセッション",
        session_ids[::-1],
        format_func=sid_label,
    )

    # オプション
    with st.expander("オプション", expanded=False):
        skip_pose   = st.checkbox("pose をスキップ（既に完了済みなら高速化）", value=False)
        from_step   = st.selectbox("このステップから再実行",
                                    ["extract","keyboard","pose","onset","integrate","export","report","arpeggio","error"],
                                    index=0)
        no_excel    = st.checkbox("Excel 出力をスキップ", value=False)

    # 実行ボタン
    if st.button("▶️ 分析を実行", type="primary", use_container_width=True):
        cmd = [PYTHON, "run_pipeline.py", "--session-id", selected_sid,
               "--from", from_step]
        if skip_pose:
            cmd.append("--skip-pose")
        if no_excel:
            cmd.append("--no-excel")

        st.markdown("---")
        st.subheader("実行ログ")
        log_box = st.empty()
        prog    = st.progress(0)

        STEP_NAMES = {
            "Step 1": "フレーム抽出", "Step 2": "キーボード検出",
            "Step 3": "骨格推定",     "Step 4": "onset検出",
            "Step 5": "データ統合",   "Step 6": "Excel出力",
            "Step 7": "レポート",     "Step 8": "アルペジオ更新",
            "Step 9": "エラー分析",
        }

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.getcwd(),
            )
            lines = []
            current_step = 0
            for line in proc.stdout:
                line = line.rstrip()
                lines.append(line)
                # ステップ番号の検出
                m = re.search(r"Step (\d+)/(\d+)", line)
                if m:
                    current_step = int(m.group(1))
                    total_steps  = int(m.group(2))
                    prog.progress(current_step / total_steps,
                                  text=STEP_NAMES.get(f"Step {current_step}", line))
                log_box.code("\n".join(lines[-30:]), language="")
            proc.wait()
            prog.progress(1.0, text="完了")
            if proc.returncode == 0:
                st.success("分析が完了しました！「📊 結果」ページで確認できます。")
                st.cache_data.clear()
            else:
                st.error(f"エラーで終了しました（コード: {proc.returncode}）")
        except Exception as e:
            st.error(f"起動エラー: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 結果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 結果":
    st.title("📊 結果")

    session_ids = list_all_session_ids()
    if not session_ids:
        st.warning("セッションがありません")
        st.stop()

    tab_all, tab_session = st.tabs(["📈 全体トレンド", "🔍 セッション詳細"])

    # ── 全体トレンド ─────────────────────────────────────────────
    with tab_all:
        all_stats = []
        for sid in session_ids:
            try:
                d = load_session_data(sid)
                s = compute_session_stats(d)
                s["sid"] = sid
                all_stats.append(s)
            except Exception:
                pass

        if len(all_stats) < 2:
            st.info("トレンドを表示するには2セッション以上必要です")
        else:
            # サマリーテーブル
            import pandas as pd
            rows = []
            for i, s in enumerate(all_stats):
                rows.append({
                    "#": i + 1,
                    "ラベル":   s["label"] or "-",
                    "打鍵数":   s["n_keys"],
                    "WPM":      round(s["wpm"]),
                    "ミス率%":  round(s["error_rate"], 1),
                    "平均iv(ms)": round(s["avg_iv"]),
                    "キー一致%": round(s.get("key_match", 0)),
                    "運指一致%": round(s.get("fin_match", 0)),
                    "録音日":   (s["recorded_at"] or s["sid"])[:10],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # グラフ
            x_labels = [f"S{s['#']}" + (f"\n{s['ラベル'][:5]}" if s["ラベル"] != "-" else "")
                        for s in rows]
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=["WPM", "平均打鍵間隔(ms)",
                                                "キー/運指一致率", "ミス率"])

            fig.add_trace(go.Scatter(
                x=x_labels, y=[s["wpm"] for s in all_stats],
                mode="lines+markers", name="WPM",
                line=dict(color="#2196F3")), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=x_labels, y=[s["avg_iv"] for s in all_stats],
                mode="lines+markers", name="avg_iv",
                line=dict(color="#FF9800")), row=1, col=2)

            fig.add_trace(go.Bar(
                x=x_labels, y=[s.get("key_match", 0) for s in all_stats],
                name="キー一致%", marker_color="#4CAF50"), row=2, col=1)
            fig.add_trace(go.Bar(
                x=x_labels, y=[s.get("fin_match", 0) for s in all_stats],
                name="運指一致%", marker_color="#9C27B0"), row=2, col=1)

            fig.add_trace(go.Bar(
                x=x_labels, y=[s["error_rate"] for s in all_stats],
                name="ミス率",
                marker_color=["#F44336" if s["error_rate"] > 0.5 else "#4CAF50"
                              for s in all_stats]), row=2, col=2)

            fig.update_layout(height=500, showlegend=True,
                               margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── セッション詳細 ────────────────────────────────────────────
    with tab_session:
        def fmt_sid(sid):
            meta = load_meta(sid)
            lbl  = meta.get("label", "")
            date = meta.get("recorded_at", "")[:10]
            return f"{sid[-8:]}  {date}  {('['+lbl+']') if lbl else ''}"

        sel_sid = st.selectbox("セッション", session_ids[::-1],
                                format_func=fmt_sid, key="detail_sid")

        data  = load_session_data(sel_sid)
        stats = compute_session_stats(data)
        keys  = data["keys"]
        meta  = data["meta"]

        # メトリクス
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WPM",    f"{stats['wpm']:.0f}")
        c2.metric("打鍵数", stats["n_keys"])
        c3.metric("ミス率", f"{stats['error_rate']:.1f}%")
        c4.metric("運指一致", f"{stats.get('fin_match', 0):.0f}%")
        if meta.get("label"):
            st.caption(f"ラベル: {meta['label']}  |  カテゴリ: {meta.get('category','')}  |  録音: {meta.get('recorded_at','')}")

        # 打鍵間隔の時系列
        normal = [k for k in keys if not k.get("is_backspace")
                  and not k["key"].startswith("Key.")]
        valid  = [k for k in normal if 0 < k["interval_ms"] <= PHRASE_BOUNDARY_MS]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=[k["timestamp_ms"] / 1000 for k in valid],
            y=[k["interval_ms"] for k in valid],
            mode="lines", opacity=0.6,
            line=dict(color="#2196F3", width=1),
        ))
        fig_ts.add_hline(y=np.mean([k["interval_ms"] for k in valid]) if valid else 0,
                          line_dash="dash", line_color="red",
                          annotation_text="平均")
        for k in normal:
            if k.get("is_error"):
                fig_ts.add_vline(x=k["timestamp_ms"] / 1000,
                                  line_color="red", line_width=2, opacity=0.7)
        fig_ts.update_layout(
            title="打鍵間隔の時系列（赤縦線=ミス）",
            xaxis_title="経過時間 (s)", yaxis_title="interval (ms)",
            height=300, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # キー使用頻度
        key_counts = defaultdict(int)
        for k in normal:
            if len(k["key"]) == 1 and k["key"].isalpha():
                key_counts[k["key"].lower()] += 1

        rows_layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        z_data, y_labels = [], []
        max_len = max(len(r) for r in rows_layout)
        for row_s in rows_layout:
            padded = [key_counts.get(c, 0) for c in row_s]
            padded += [None] * (max_len - len(padded))
            z_data.append(padded)
            y_labels.append(row_s[0].upper() + "行")

        fig_hm = go.Figure(go.Heatmap(
            z=z_data, colorscale="Blues",
            xgap=2, ygap=2,
            text=[[f"{rows_layout[ri][ci].upper()}: {key_counts.get(rows_layout[ri][ci], 0)}"
                    if ci < len(rows_layout[ri]) else ""
                    for ci in range(max_len)]
                   for ri in range(len(rows_layout))],
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig_hm.update_layout(
            title="キー使用頻度",
            yaxis=dict(tickvals=[0, 1, 2], ticktext=y_labels),
            height=200, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # 運指一致率
        if data["integrated"]:
            integ = data["integrated"]
            cam   = [r for r in integ if r["source"] == "camera"]
            finger_total   = defaultdict(int)
            finger_correct = defaultdict(int)
            for r in cam:
                af = r.get("actual_finger")
                if af:
                    finger_total[af] += 1
                    if af == r.get("std_finger"):
                        finger_correct[af] += 1
            if finger_total:
                fnames  = [FINGER_NAMES.get(f, f) for f in FINGER_ORDER]
                totals  = [finger_total.get(f, 0) for f in FINGER_ORDER]
                rates   = [finger_correct.get(f, 0) / max(finger_total.get(f, 1), 1) * 100
                           for f in FINGER_ORDER]
                fig_fn = make_subplots(rows=1, cols=2,
                                        subplot_titles=["運指一致率 (%)", "使用回数"])
                fig_fn.add_trace(
                    go.Bar(x=fnames, y=rates,
                           marker_color=["#4CAF50" if r >= 80 else "#FF9800" if r >= 60 else "#F44336"
                                         for r in rates],
                           text=[f"{r:.0f}%" for r in rates], textposition="outside"),
                    row=1, col=1)
                fig_fn.add_trace(
                    go.Bar(x=fnames, y=totals,
                           marker_color="#2196F3",
                           text=totals, textposition="outside"),
                    row=1, col=2)
                fig_fn.update_layout(height=350, showlegend=False,
                                      margin=dict(t=30, b=10))
                st.plotly_chart(fig_fn, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✋ 運指
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "✋ 運指":
    st.title("✋ 運指最適化")

    tab_arp, tab_opt = st.tabs(["🎵 アルペジオマップ", "🔄 切り替え推奨"])

    with tab_arp:
        arp_path = output_path("arpeggio_map.json")
        if not os.path.exists(arp_path):
            st.info("アルペジオマップがありません。分析を実行してください。")
            if st.button("アルペジオ分析を実行"):
                with st.spinner("実行中..."):
                    result = subprocess.run([PYTHON, "arpeggio_analysis.py"],
                                            capture_output=True, text=True,
                                            encoding="utf-8", errors="replace")
                    st.code(result.stdout)
                    st.cache_data.clear()
                    st.rerun()
        else:
            with open(arp_path, encoding="utf-8") as f:
                arpeggio_map = json.load(f)

            by_type = defaultdict(list)
            for v in arpeggio_map.values():
                by_type[v["type"]].append(v)

            # サマリーカード
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("アルペジオペア", len(by_type["arpeggio"]))
            c2.metric("Fast ペア", len(by_type["fast"]))
            c3.metric("変動大ペア", len(by_type["variable"]), delta="急加速リスク",
                       delta_color="inverse")
            c4.metric("通常ペア", len(by_type["normal"]))

            # アルペジオペア一覧
            st.subheader("アルペジオペア（速い + 一貫）")
            import pandas as pd
            arp_rows = [
                {"ペア": f"{e['prev_key']}→{e['curr_key']}",
                 "平均(ms)": e["mean"], "SD(ms)": e["std"],
                 "CV": e["cv"], "N": e["n"]}
                for e in sorted(by_type["arpeggio"], key=lambda x: x["mean"])
            ]
            st.dataframe(pd.DataFrame(arp_rows), use_container_width=True,
                          hide_index=True)

            # 散布図
            st.subheader("ペア分類マップ")
            type_colors = {"arpeggio": "red", "fast": "orange",
                           "variable": "purple", "normal": "lightblue"}
            df_arp = pd.DataFrame([
                {"pair": f"{v['prev_key']}→{v['curr_key']}",
                 "mean": v["mean"], "cv": v["cv"],
                 "type": v["type"], "n": v["n"]}
                for v in arpeggio_map.values()
            ])
            if not df_arp.empty:
                fig_sc = px.scatter(
                    df_arp, x="mean", y="cv", color="type",
                    color_discrete_map=type_colors,
                    hover_data=["pair", "n"],
                    labels={"mean": "平均間隔(ms)", "cv": "変動係数 CV"},
                    title="ペア分類マップ（赤=アルペジオ、紫=変動大）",
                    height=400,
                )
                fig_sc.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5)
                fig_sc.add_hline(y=0.35, line_dash="dot", line_color="red", opacity=0.5)
                fig_sc.add_hline(y=0.60, line_dash="dot", line_color="purple", opacity=0.5)
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab_opt:
        st.subheader("切り替え推奨（距離・信頼性フィルタ済み）")
        st.caption("物理的に届く指の中で、切り替えることで最も速くなる推奨")

        if st.button("最適化分析を実行"):
            with st.spinner("実行中..."):
                result = subprocess.run(
                    [PYTHON, "finger_optimize.py"],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    cwd=os.getcwd(),
                )
                st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

        # 画像があれば表示
        opt_img = output_path("finger_optimize.png")
        if os.path.exists(opt_img):
            st.image(opt_img, use_container_width=True)
