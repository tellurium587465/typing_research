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
from finger_select import select_finger, classify_quality

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

print("\n全テスト合格 ✅")
