"""
test_finger_select.py ─ finger_select の単体テスト（合成 pose データ）

カメラ実データ無しでも投票ロジックの正しさを検証できる。
  python test_finger_select.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from finger_select import select_finger, classify_quality, reorder_fingers_by_x
from validate_finger import compute_validation

KEY_W = 30.0
FINGER_TO_CODE = {
    ("Right", "index"): "R4", ("Right", "middle"): "R3",
    ("Right", "ring"): "R2",  ("Right", "pinky"): "R1",
    ("Left",  "index"): "L4", ("Left",  "middle"): "L3",
}
EXCLUDED = {("Right", "thumb")}

# 打鍵キー I の中心を (300, 100) とする
KX, KY = 300, 100
KEYMAP = {"R3": (300, 100), "R4": (270, 100), "R2": (330, 100)}
def fake_pixel_to_key(px, py):  # 距離が最も近いキー名を返すダミー
    return min(KEYMAP, key=lambda k: (KEYMAP[k][0]-px)**2 + (KEYMAP[k][1]-py)**2)


def frame(ts, fingers):
    """fingers: {(label,fname):(px,py,z)} → 1フレーム分の pose 構造を作る"""
    hands = {}
    for (label, fname), (px, py, z) in fingers.items():
        hands.setdefault(label, {"label": label, "fingertips": {}})
        hands[label]["fingertips"][fname] = {"px": px, "py": py, "z": z}
    return {"timestamp_ms": ts, "hands": list(hands.values())}


def run(name, cond):
    print(f"  [{'OK' if cond else 'NG'}] {name}")
    assert cond, name


print("=== finger_select 単体テスト ===")

# ── テスト1: 明確に R3 がキー上にある → 高信頼 standard ──────────────
nearby = [
    frame(990, {("Right", "middle"): (300, 100, 0.05),   # キー直上
                ("Right", "index"):  (240, 100, 0.0)}),   # 遠い
    frame(1000, {("Right", "middle"): (300, 101, 0.06),
                 ("Right", "index"):  (242, 100, 0.0)}),
    frame(1010, {("Right", "middle"): (301, 100, 0.05),
                 ("Right", "index"):  (240, 101, 0.0)}),
]
sel = select_finger(nearby, 1000, KX, KY, "i", 150, KEY_W,
                    FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key)
print(f"  T1 結果: {sel}")
run("T1 勝者が R3", sel["finger_code"] == "R3")
run("T1 高信頼", sel["confidence"] >= 0.6)
q = classify_quality(sel["confidence"], sel["margin"], sel["finger_code"] == "R3")
run("T1 standard 判定", q == "standard")

# ── テスト2: 非標準運指（R4 で I を打つ）が高信頼 → nonstandard ──────
nearby2 = [
    frame(1000, {("Right", "index"):  (300, 100, 0.05),   # R4 がキー上
                 ("Right", "middle"): (350, 100, 0.0)}),   # R3 は遠い
    frame(1005, {("Right", "index"):  (300, 99, 0.05),
                 ("Right", "middle"): (352, 100, 0.0)}),
]
sel2 = select_finger(nearby2, 1000, KX, KY, "i", 150, KEY_W,
                     FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key)
std_finger = "R3"  # I の標準運指
q2 = classify_quality(sel2["confidence"], sel2["margin"],
                      sel2["finger_code"] == std_finger)
print(f"  T2 結果: {sel2}  quality={q2}")
run("T2 勝者が R4", sel2["finger_code"] == "R4")
run("T2 nonstandard 判定（誤検出ではなく癖として分離）", q2 == "nonstandard")

# ── テスト3: 2指が拮抗 → 低信頼 uncertain ───────────────────────────
nearby3 = [
    frame(1000, {("Right", "middle"): (300, 100, 0.05),
                 ("Right", "index"):  (302, 100, 0.05)}),  # ほぼ同距離
]
sel3 = select_finger(nearby3, 1000, KX, KY, "i", 150, KEY_W,
                     FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key)
q3 = classify_quality(sel3["confidence"], sel3["margin"],
                      sel3["finger_code"] == "R3")
print(f"  T3 結果: {sel3}  quality={q3}")
run("T3 margin が小さい", sel3["margin"] < 0.15)
run("T3 uncertain 判定（誤検出疑いとして除外可）", q3 == "uncertain")

# ── テスト4: reach 外しかない → None ────────────────────────────────
nearby4 = [frame(1000, {("Right", "middle"): (500, 500, 0.0)})]  # 遠すぎ
sel4 = select_finger(nearby4, 1000, KX, KY, "i", 150, KEY_W,
                     FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key)
run("T4 候補なしで None", sel4 is None)

# ── テスト5: 除外指(右親指)は無視される ─────────────────────────────
nearby5 = [frame(1000, {("Right", "thumb"):  (300, 100, 0.0),   # 除外
                        ("Right", "middle"): (305, 100, 0.05)})]
sel5 = select_finger(nearby5, 1000, KX, KY, "i", 150, KEY_W,
                     FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key)
run("T5 除外指を選ばず R3", sel5["finger_code"] == "R3")

# ── テスト6: 順序補正で R3↔R2 取り違えを直す ───────────────────────
# 右手で middle と ring の px が逆転している（取り違え）フレームを与える
bad_hand = {"label": "Right", "fingertips": {
    "pinky":  {"px": 100, "py": 100, "z": 0.0},
    "ring":   {"px": 300, "py": 100, "z": 0.0},   # ring が middle より右（逆転）
    "middle": {"px": 200, "py": 100, "z": 0.0},   # middle が ring より左
    "index":  {"px": 400, "py": 100, "z": 0.0},
}}
fixed = reorder_fingers_by_x(bad_hand)
# 補正後は ring(px=200) < middle(px=300) になっているはず（名前が入れ替わる）
run("T6 順序補正で ring が左(px=200)に", fixed["ring"]["px"] == 200)
run("T6 順序補正で middle が右(px=300)に", fixed["middle"]["px"] == 300)

# 正しい順序なら何も変えない
good_hand = {"label": "Right", "fingertips": {
    "pinky":  {"px": 100, "py": 100, "z": 0.0},
    "ring":   {"px": 200, "py": 100, "z": 0.0},
    "middle": {"px": 300, "py": 100, "z": 0.0},
    "index":  {"px": 400, "py": 100, "z": 0.0},
}}
fixed_g = reorder_fingers_by_x(good_hand)
run("T6 正しい順序は不変", fixed_g["middle"]["px"] == 300 and fixed_g["ring"]["px"] == 200)

# ── テスト7: z 押下キュー（use_z）─ 拮抗時に "下がった" 指を選ぶ ────
# 2指がほぼ同距離だが、R3 の z が大きい（下がっている＝押下）
nearby7 = [
    frame(1000, {("Right", "middle"): (300, 100, 0.20),   # z 大 = 押下寄り
                 ("Right", "index"):  (303, 100, 0.00)}),
]
sel7_z = select_finger(nearby7, 1000, KX, KY, "i", 150, KEY_W,
                       FINGER_TO_CODE, EXCLUDED, fake_pixel_to_key, use_z=True)
run("T7 z キューで押下している R3 を選ぶ", sel7_z["finger_code"] == "R3")

# ── テスト8: 妥当性検証ロジック（compute_validation）──────────────
recs = [
    {"predicted": "R3", "true": "R3", "quality": "standard",    "confidence": 0.95},
    {"predicted": "R3", "true": "R3", "quality": "standard",    "confidence": 0.90},
    {"predicted": "R4", "true": "R4", "quality": "nonstandard", "confidence": 0.85},
    {"predicted": "R3", "true": "R2", "quality": "uncertain",   "confidence": 0.52},  # 外れ R2→R3
    {"predicted": "R2", "true": "R3", "quality": "uncertain",   "confidence": 0.55},  # 外れ R3→R2
    {"predicted": "L4", "true": "",   "quality": "standard",    "confidence": 0.80},  # 未ラベル
]
m = compute_validation(recs)
print(f"\n  validate 結果: n={m['n']} acc={m['overall_acc']:.2f} "
      f"by_q={ {k:round(v['acc'],2) for k,v in m['by_quality'].items()} }")
run("T8 未ラベルを除外して n=5", m["n"] == 5)
run("T8 全体正解率 3/5=0.6", abs(m["overall_acc"] - 0.6) < 1e-9)
run("T8 standard は高精度", m["by_quality"]["standard"]["acc"] == 1.0)
run("T8 uncertain は低精度", m["by_quality"]["uncertain"]["acc"] == 0.0)
run("T8 R3↔R2 取り違えを検出", (("R2", "R3") in m["confusion"]) and (("R3", "R2") in m["confusion"]))
run("T8 当たりの方が高信頼", m["mean_conf_correct"] > m["mean_conf_wrong"])

print("\n全テスト合格 ✅")
