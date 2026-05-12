import json

KEYLOG_FILE  = "keylog_1778545201.json"
DTLD_FILE    = r"C:\Users\25B1029\Desktop\DtldKHJY.log"

# -----------------------------------------------
# タイプウェルの単語リストを読み込む
# -----------------------------------------------
with open(DTLD_FILE, encoding="cp932", errors="ignore") as f:
    raw = f.read().strip()

# \x01\x00 などのバイナリゴミを除去してASCIIのみ残す
cleaned = "".join(c for c in raw if c.isprintable() and ord(c) < 128)

# _ 区切りで単語に分割
words = [w for w in cleaned.split("_") if w]
print(f"単語数: {len(words)}語")
print(f"単語例: {words[:5]}")

# キーログの打鍵数に近い長さに単語列を調整
# 打鍵列の長さに合わせて単語を使う
expected_str = "".join(words)
print(f"\n正解列（先頭100文字）: {expected_str[:100]}")

# -----------------------------------------------
# キーログから打鍵文字列を再構成
# -----------------------------------------------
with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)

# 通常キーのみ（特殊キー・バックスペース除外）
typed_keys = [
    e for e in keylog
    if not e.get("is_backspace", False)
    and not e["key"].startswith("Key.")
    and len(e["key"]) == 1
]

typed_str = "".join(e["key"] for e in typed_keys)
print(f"\n打鍵列（先頭100文字）: {typed_str[:100]}")
print(f"\n正解列の長さ: {len(expected_str)}")
print(f"打鍵列の長さ: {len(typed_str)}")

# -----------------------------------------------
# Levenshtein距離でミスを検出
# -----------------------------------------------
def levenshtein_align(s1, s2):
    """
    s1=正解列, s2=打鍵列
    DPテーブルを使って各打鍵がミスか正解かを判定する
    戻り値: 打鍵列の各文字に対する操作リスト
    """
    m, n = len(s1), len(s2)
    # DPテーブル
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 削除
                    dp[i][j-1],    # 挿入（余分なキー）
                    dp[i-1][j-1]   # 置換
                )

    # バックトラックでアライメント取得
    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append(("match", s2[j-1]))
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] < dp[i-1][j] and dp[i][j-1] <= dp[i-1][j-1]):
            ops.append(("insert", s2[j-1]))  # 余分なキー（ミス）
            j -= 1
        elif i > 0 and j > 0 and dp[i-1][j-1] <= dp[i-1][j]:
            ops.append(("replace", s2[j-1]))  # 置換（ミス）
            i -= 1; j -= 1
        else:
            ops.append(("delete", s1[i-1]))  # 打ち忘れ
            i -= 1
    ops.reverse()
    return ops

# 長い場合は最初の300文字で比較
def normalize_roman(s):
    """ローマ字表記ゆれを正規化する"""
    s = s.replace("nn", "n")       # ん（nn→n）
    s = s.replace("xtsu", "ltu")   # っ
    s = s.replace("xtu", "ltu")    # っ
    s = s.replace("ltsu", "ltu")   # っ
    s = s.replace("shi", "si")     # し
    s = s.replace("chi", "ti")     # ち
    s = s.replace("tsu", "tu")     # つ
    s = s.replace("fu", "hu")      # ふ
    s = s.replace("ji", "zi")      # じ
    s = s.replace("t-", "-t")      # っ（長音前の位置ゆれ）
    return s

# 正規化後に比較
expected_norm = normalize_roman(expected_str)
typed_norm    = normalize_roman(typed_str)

print(f"\n正規化後 正解列（先頭100文字）: {expected_norm[:100]}")
print(f"正規化後 打鍵列（先頭100文字）: {typed_norm[:100]}")

limit = min(len(expected_norm), len(typed_norm), 400)
ops = levenshtein_align(expected_norm[:limit], typed_norm[:limit])

# -----------------------------------------------
# 結果集計
# -----------------------------------------------
match_count   = sum(1 for op, _ in ops if op == "match")
insert_count  = sum(1 for op, _ in ops if op == "insert")
replace_count = sum(1 for op, _ in ops if op == "replace")
delete_count  = sum(1 for op, _ in ops if op == "delete")

total_expected = match_count + replace_count + delete_count
mistake_count  = insert_count + replace_count

print(f"\n{'='*50}")
print(f"【ミス検出結果】")
print(f"{'='*50}")
print(f"正解打鍵:   {match_count}回")
print(f"余分な打鍵: {insert_count}回（タイプウェルのミス）")
print(f"置換ミス:   {replace_count}回")
print(f"打ち忘れ:   {delete_count}回")
print(f"ミス率:     {mistake_count/max(len(typed_str[:limit]),1)*100:.1f}%")

# -----------------------------------------------
# キーログにミスフラグを付与して保存
# -----------------------------------------------
# アライメント結果を使って打鍵列の各キーにフラグ付与
typed_idx = 0
flagged = []
for op, char in ops:
    if op in ("insert", "replace"):
        # この打鍵はミス
        while typed_idx < len(typed_keys) and typed_keys[typed_idx]["key"] != char:
            typed_idx += 1
        if typed_idx < len(typed_keys):
            typed_keys[typed_idx]["is_error"] = True
            typed_idx += 1
    elif op == "match":
        if typed_idx < len(typed_keys):
            typed_keys[typed_idx]["is_error"] = False
            typed_idx += 1

output_file = KEYLOG_FILE.replace(".json", "_with_errors.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(keylog, f, ensure_ascii=False, indent=2)

print(f"\n保存: {output_file}")
print(f"（ミスフラグを付与した打鍵ログ）")