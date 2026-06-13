"""
finger_select.py ─ 打鍵→使用指の推定（多フレーム投票＋信頼度＋品質分類）

従来 integrate.py は「打鍵キーに最も近い指先を 1 フレームだけ見て採用」していた。
その弱点（単一フレームのジッタに弱い・確からしさが残らない・"非標準運指" と
"誤検出" を区別できない）を解消するため、ここに純関数として切り出した。

設計:
  ・onset 周辺の窓内 "全フレーム" で各指に投票する
        票 = 時間の近さ(w_time) × キーへの近さ(w_prox)
    合計票が最大の指を採用 → 単一フレームのノイズに強い。
  ・confidence = 勝者票 / 総票、  margin = (勝者票 − 2位票) / 総票
    打鍵ごとの確からしさを数値として残す。
  ・classify_quality() で
        standard    : 高信頼かつ標準運指と一致
        nonstandard : 高信頼だが標準と不一致（＝本人の運指の癖。計測したい本物）
        uncertain   : 低信頼（誤検出の疑い。集計から除外できる）
    に分け、"癖" と "ノイズ" を分離する。

依存なし（標準ライブラリのみ）。integrate.py から import して使う。
"""
from collections import defaultdict

# 既定パラメータ
REACH_FACTOR_DEFAULT = 2.0   # 候補にする最大距離（キー幅単位）
CONF_HI_DEFAULT      = 0.60  # これ以上の得票割合なら検出を信頼する
MARGIN_HI_DEFAULT    = 0.15  # 2位との得票差がこれ以上なら確定的とみなす


# 同じ手の中で指先が x 方向に並ぶべき順序（俯瞰・手のひら下向き）
# MediaPipe が中指/薬指を取り違えると、この単調性が崩れるので検出できる。
_HAND_ORDER = {
    "Left":  ["thumb", "index", "middle", "ring", "pinky"],   # 左手: 親指が右→小指が左
    "Right": ["pinky", "ring", "middle", "index", "thumb"],   # 右手: 小指が左→親指が右
}


def reorder_fingers_by_x(hand, tol_px=6.0):
    """
    1つの手の fingertips を x 座標の並びで検査し、隣接指が
    期待順序と逆転していれば名前を入れ替えて返す（R3↔R2 取り違え対策・実験的）。

    俯瞰カメラで手のひらを下に向けると、各指先の x は手ごとに単調に並ぶはず。
    例（右手）: pinky < ring < middle < index < thumb（px 昇順）。
    隣り合う2指が tol_px を超えて逆転していたら名前を交換する。
    返り値: 補正後の fingertips dict（元は変更しない）。
    """
    label = hand.get("label")
    order = _HAND_ORDER.get(label)
    tips = hand.get("fingertips", {})
    if not order:
        return dict(tips)
    present = [name for name in order if name in tips]
    fixed = dict(tips)
    # 隣接ペアの逆転を1パスで補正（バブル1段相当）
    changed = True
    guard = 0
    while changed and guard < 5:
        changed = False
        guard += 1
        for i in range(len(present) - 1):
            a, b = present[i], present[i + 1]
            xa, xb = fixed[a]["px"], fixed[b]["px"]
            # 期待は xa < xb。tol を超えて xa > xb なら名前を入れ替える
            if xa - xb > tol_px:
                fixed[a], fixed[b] = fixed[b], fixed[a]
                changed = True
    return fixed


def select_finger(nearby, onset_ms, std_kx, std_ky, key, window_ms,
                  key_w_px, finger_to_code, excluded_fingers, pixel_to_key,
                  reach_factor=REACH_FACTOR_DEFAULT, use_z=False, reorder=False):
    """
    onset 周辺フレーム群から使用指を投票で推定する。

    引数:
      nearby          : [{timestamp_ms, hands:[{label, fingertips:{name:{px,py,z}}}]}, ...]
      onset_ms        : 打鍵タイミング(ms)
      std_kx, std_ky  : 打鍵キーの中心座標(px)
      key             : 打鍵文字（thumb 除外判定に使用）
      window_ms       : 探索窓の半幅(ms)。時間重みの正規化にも使う
      key_w_px        : キー幅(px)。reach 距離の基準
      finger_to_code  : {(label, fname): "R3" ...} 変換表
      excluded_fingers: {(label, fname), ...} 検出対象外
      pixel_to_key    : (px,py)->キー名 を返す callable
      reach_factor    : 候補距離 = key_w_px * reach_factor
      use_z           : True なら z（深度）の押下キューを加味（実験的）

    返り値: dict（推定結果）または None（候補なし）
    """
    reach_px = key_w_px * reach_factor
    votes    = defaultdict(float)
    cnt      = defaultdict(int)
    best_pos = {}   # code -> (px, py, dist)  その指が最もキーに近づいた瞬間
    z_at_min = {}   # code -> z（最近接時）

    for frame in nearby:
        ts_diff = abs(frame["timestamp_ms"] - onset_ms)
        w_time  = max(0.1, 1.0 - ts_diff / window_ms) if window_ms else 1.0
        for hand in frame.get("hands", []):
            label = hand["label"]
            tips = reorder_fingers_by_x(hand) if reorder else hand["fingertips"]
            for fname, coords in tips.items():
                if (label, fname) in excluded_fingers:
                    continue
                if fname == "thumb" and key != " ":
                    continue
                px, py = coords["px"], coords["py"]
                dist = ((px - std_kx) ** 2 + (py - std_ky) ** 2) ** 0.5
                if dist > reach_px:
                    continue
                code   = finger_to_code.get((label, fname), f"{label[0]}_{fname}")
                w_prox = max(0.0, 1.0 - dist / reach_px)
                votes[code] += w_time * w_prox
                cnt[code]   += 1
                if code not in best_pos or dist < best_pos[code][2]:
                    best_pos[code] = (px, py, dist)
                    z_at_min[code] = coords.get("z")

    if not votes:
        return None

    # z 押下キュー（実験的）：最近接時に最も "下がっている" 指に小ボーナス
    if use_z:
        zs = {c: z for c, z in z_at_min.items() if z is not None}
        if len(zs) >= 2:
            lo, hi = min(zs.values()), max(zs.values())
            if hi > lo:
                for c, z in zs.items():
                    votes[c] += 0.2 * (z - lo) / (hi - lo)  # z大=下=押下寄り

    ranked = sorted(votes.items(), key=lambda x: -x[1])
    winner, win_v = ranked[0]
    total = sum(votes.values())
    confidence = win_v / total if total else 0.0
    margin = (win_v - ranked[1][1]) / total if len(ranked) > 1 and total else 1.0

    px, py, dist = best_pos[winner]
    return {
        "finger_code":  winner,
        "px": px, "py": py,
        "dist": round(float(dist), 1),
        "detected_key": pixel_to_key(px, py),
        "confidence":   round(confidence, 3),
        "margin":       round(margin, 3),
        "n_frames":     cnt[winner],
        "n_candidates": len(votes),
    }


def classify_quality(confidence, margin, finger_match,
                     conf_hi=CONF_HI_DEFAULT, margin_hi=MARGIN_HI_DEFAULT):
    """
    検出の品質を 3 段階に分類して "本人の癖" と "誤検出" を分離する。
      standard    : 高信頼 & 標準運指と一致
      nonstandard : 高信頼 & 標準運指と不一致（＝計測したい本物の運指の癖）
      uncertain   : 低信頼（誤検出の疑い。集計から除外できる）
    """
    if confidence >= conf_hi and margin >= margin_hi:
        return "standard" if finger_match else "nonstandard"
    return "uncertain"
