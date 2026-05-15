import json
import glob
import jaconv
from collections import defaultdict
import numpy as np

# -----------------------------------------------
# ローマ字→ひらがな変換
# -----------------------------------------------
import pykakasi
kks = pykakasi.kakasi()

# ローマ字→ひらがな変換テーブル（Microsoft IME準拠・網羅版）
ROMAN_TO_HIRA = {
    # 3文字以上（先に定義して優先させる）
    "sha":"しゃ","shi":"し","shu":"しゅ","she":"しぇ","sho":"しょ",
    "chi":"ち","cha":"ちゃ","chu":"ちゅ","che":"ちぇ","cho":"ちょ",
    "tsu":"つ","tsa":"つぁ",
    "thi":"てぃ","thu":"てゅ","tha":"てゃ","tho":"てょ",
    "dhi":"でぃ","dhu":"でゅ","dha":"でゃ","dho":"でょ",
    "kya":"きゃ","kyi":"きぃ","kyu":"きゅ","kye":"きぇ","kyo":"きょ",
    "gya":"ぎゃ","gyi":"ぎぃ","gyu":"ぎゅ","gye":"ぎぇ","gyo":"ぎょ",
    "sya":"しゃ","syi":"しぃ","syu":"しゅ","sye":"しぇ","syo":"しょ",
    "zya":"じゃ","zyi":"じぃ","zyu":"じゅ","zye":"じぇ","zyo":"じょ",
    "tya":"ちゃ","tyi":"ちぃ","tyu":"ちゅ","tye":"ちぇ","tyo":"ちょ",
    "nya":"にゃ","nyi":"にぃ","nyu":"にゅ","nye":"にぇ","nyo":"にょ",
    "hya":"ひゃ","hyi":"ひぃ","hyu":"ひゅ","hye":"ひぇ","hyo":"ひょ",
    "mya":"みゃ","myi":"みぃ","myu":"みゅ","mye":"みぇ","myo":"みょ",
    "rya":"りゃ","ryi":"りぃ","ryu":"りゅ","rye":"りぇ","ryo":"りょ",
    "bya":"びゃ","byi":"びぃ","byu":"びゅ","bye":"びぇ","byo":"びょ",
    "pya":"ぴゃ","pyi":"ぴぃ","pyu":"ぴゅ","pye":"ぴぇ","pyo":"ぴょ",
    "dya":"ぢゃ","dyi":"ぢぃ","dyu":"ぢゅ","dye":"ぢぇ","dyo":"ぢょ",
    "fya":"ふゃ","fyu":"ふゅ","fyo":"ふょ",
    "fwa":"ふぁ","fwi":"ふぃ","fwu":"ふぅ","fwe":"ふぇ","fwo":"ふぉ",
    "twa":"とぁ","twi":"とぃ","twu":"とぅ","twe":"とぇ","two":"とぉ",
    "dwa":"どぁ","dwi":"どぃ","dwu":"どぅ","dwe":"どぇ","dwo":"どぉ",
    "kwa":"くぁ","kwi":"くぃ","kwu":"くぅ","kwe":"くぇ","kwo":"くぉ",
    "gwa":"ぐぁ","gwi":"ぐぃ","gwu":"ぐぅ","gwe":"ぐぇ","gwo":"ぐぉ",
    "mwa":"むぁ",
    "vya":"ゔゃ","vyu":"ゔゅ","vyo":"ゔょ",
    "va":"ゔぁ","vi":"ゔぃ","vu":"ゔ","ve":"ゔぇ","vo":"ゔぉ",
    "xya":"ゃ","xyu":"ゅ","xyo":"ょ",
    "xtu":"っ","xtsu":"っ","ltu":"っ","ltsu":"っ",
    "xka":"ヵ","xke":"ヶ",
    "xa":"ぁ","xi":"ぃ","xu":"ぅ","xe":"ぇ","xo":"ぉ",
    "la":"ぁ","li":"ぃ","lu":"ぅ","le":"ぇ","lo":"ぉ",
    "lya":"ゃ","lyi":"ぃ","lyu":"ゅ","lye":"ぇ","lyo":"ょ",
    "lwa":"ゎ","xwa":"ゎ",
    "wyi":"ゐ","wye":"ゑ",
    "nn":"ん","n'":"ん",
    # 2文字
    "ba":"ば","bi":"び","bu":"ぶ","be":"べ","bo":"ぼ",
    "ca":"か","ci":"し","cu":"く","ce":"せ","co":"こ",
    "da":"だ","di":"ぢ","du":"づ","de":"で","do":"ど",
    "fa":"ふぁ","fi":"ふぃ","fu":"ふ","fe":"ふぇ","fo":"ふぉ",
    "ga":"が","gi":"ぎ","gu":"ぐ","ge":"げ","go":"ご",
    "ha":"は","hi":"ひ","hu":"ふ","he":"へ","ho":"ほ",
    "ja":"じゃ","ji":"じ","ju":"じゅ","je":"じぇ","jo":"じょ",
    "ka":"か","ki":"き","ku":"く","ke":"け","ko":"こ",
    "ma":"ま","mi":"み","mu":"む","me":"め","mo":"も",
    "na":"な","ni":"に","nu":"ぬ","ne":"ね","no":"の",
    "pa":"ぱ","pi":"ぴ","pu":"ぷ","pe":"ぺ","po":"ぽ",
    "ra":"ら","ri":"り","ru":"る","re":"れ","ro":"ろ",
    "sa":"さ","si":"し","su":"す","se":"せ","so":"そ",
    "ta":"た","ti":"ち","tu":"つ","te":"て","to":"と",
    "wa":"わ","wi":"ゐ","we":"ゑ","wo":"を",
    "ya":"や","yi":"い","yu":"ゆ","ye":"いぇ","yo":"よ",
    "za":"ざ","zi":"じ","zu":"ず","ze":"ぜ","zo":"ぞ",
    "nb":"ん","nm":"ん","np":"ん",
    # 1文字
    "a":"あ","i":"い","u":"う","e":"え","o":"お",
}

def roman_to_hira(roman):
    """ローマ字文字列をひらがなに変換する（多表記対応）"""
    roman = roman.lower()
    result = ""
    i = 0
    while i < len(roman):
        matched = False
        # 長い順にマッチを試みる（4文字→3文字→2文字→1文字）
        for length in [4, 3, 2, 1]:
            chunk = roman[i:i+length]
            if chunk in ROMAN_TO_HIRA:
                result += ROMAN_TO_HIRA[chunk]
                i += length
                matched = True
                break
        if not matched:
            # 促音（同じ子音が連続）
            if i + 1 < len(roman) and roman[i] == roman[i+1] and roman[i] not in "aiueonn":
                result += "っ"
                i += 1
            # n単体（次が子音または文末）
            elif roman[i] == "n" and (i+1 >= len(roman) or roman[i+1] not in "aiueo"):
                result += "ん"
                i += 1
            else:
                result += roman[i]
                i += 1
    return result

# -----------------------------------------------
# キーログから打鍵文字列を再構成
# -----------------------------------------------
def reconstruct_typed(keylog):
    """キーログから実際に打った文字列を再構成する"""
    chars = []
    for e in keylog:
        key = e["key"]
        if e.get("is_backspace"):
            if chars:
                chars.pop()
        elif not key.startswith("Key.") and len(key) == 1:
            chars.append(key)
    return "".join(chars)

# -----------------------------------------------
# ミス検出（Levenshtein + アライメント）
# -----------------------------------------------
def levenshtein_align(s1, s2):
    """s1=正解列, s2=打鍵列でアライメントを取る"""
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append(("match", s1[i-1], s2[j-1]))
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] <= dp[i-1][j] and dp[i][j-1] <= dp[i-1][j-1]):
            ops.append(("insert", "", s2[j-1]))
            j -= 1
        elif i > 0 and j > 0 and dp[i-1][j-1] <= dp[i-1][j]:
            ops.append(("replace", s1[i-1], s2[j-1]))
            i -= 1; j -= 1
        else:
            ops.append(("delete", s1[i-1], ""))
            i -= 1
    ops.reverse()
    return ops

# -----------------------------------------------
# 文字3-gram分析
# -----------------------------------------------
def analyze_char_trigrams(keylog, min_count=3):
    """文字レベルの3-gram分析"""
    normal = [
        e for e in keylog
        if not e.get("is_backspace", False)
        and not e["key"].startswith("Key.")
        and len(e["key"]) == 1
    ]

    trigrams = defaultdict(list)
    for i in range(len(normal) - 2):
        a, b, c = normal[i], normal[i+1], normal[i+2]
        key = (a["key"], b["key"], c["key"])
        avg_interval = (b["interval_ms"] + c["interval_ms"]) / 2
        trigrams[key].append(avg_interval)

    results = []
    for (k1, k2, k3), intervals in trigrams.items():
        if len(intervals) < min_count:
            continue
        roman = k1 + k2 + k3
        hira  = roman_to_hira(roman)
        avg   = np.mean(intervals)
        score = len(intervals) * avg
        results.append({
            "roman":    roman,
            "hira":     hira,
            "key_1": k1, "key_2": k2, "key_3": k3,
            "count":    len(intervals),
            "avg_ms":   round(avg, 1),
            "score":    round(score, 1),
        })

    return sorted(results, key=lambda x: -x["score"])

# -----------------------------------------------
# メイン処理
# -----------------------------------------------
keylog_files = glob.glob("keylog_*_clean.json")
if not keylog_files:
    keylog_files = glob.glob("keylog_*_with_errors.json") + [
        f for f in glob.glob("keylog_*.json")
        if "with_errors" not in f and "clean" not in f
    ]

print(f"対象ファイル: {len(keylog_files)}件")

all_keylog = []
for fname in keylog_files:
    with open(fname, encoding="utf-8") as f:
        data = json.load(f)
    all_keylog.extend(data)

# -----------------------------------------------
# 打鍵文字列の可視化
# -----------------------------------------------
print("\n" + "="*60)
print("【打鍵文字列の可視化】")
print("="*60)

typed_roman = reconstruct_typed(all_keylog)
typed_hira  = roman_to_hira(typed_roman)

print(f"打鍵ローマ字（先頭200文字）: {typed_roman[:200]}")
print(f"ひらがな変換（先頭100文字）: {typed_hira[:100]}")

# -----------------------------------------------
# 文字3-gram分析
# -----------------------------------------------
print("\n" + "="*60)
print("【文字3-gram ボトルネック上位20件】")
print("スコア = 出現頻度 × 平均打鍵間隔")
print("="*60)

trigrams = analyze_char_trigrams(all_keylog, min_count=3)

print(f"\n{'ローマ字':8s} {'ひらがな':10s} {'出現':6s} {'平均ms':8s} {'スコア':8s}")
print("-" * 50)
for r in trigrams[:20]:
    print(f"  {r['roman']:8s} {r['hira']:10s} {r['count']:4d}回  "
          f"{r['avg_ms']:7.1f}ms  {r['score']:8.1f}")

# -----------------------------------------------
# 最速パターン上位10件
# -----------------------------------------------
fastest = sorted(trigrams, key=lambda x: x["avg_ms"])
print("\n" + "="*60)
print("【文字3-gram 最速パターン上位10件】")
print("="*60)
print(f"\n{'ローマ字':8s} {'ひらがな':10s} {'出現':6s} {'平均ms':8s}")
print("-" * 40)
for r in fastest[:10]:
    print(f"  {r['roman']:8s} {r['hira']:10s} {r['count']:4d}回  {r['avg_ms']:7.1f}ms")

# -----------------------------------------------
# ミス検出（正解文字列が必要な場合はここで比較）
# -----------------------------------------------
print("\n" + "="*60)
print("【バックスペースなしミス推定】")
print("（前後の文脈から浮いているキーを検出）")
print("="*60)

# 連続する同じキーで間隔が極端に長いものをミス候補として検出
normal_all = [
    e for e in all_keylog
    if not e.get("is_backspace", False)
    and not e["key"].startswith("Key.")
    and len(e["key"]) == 1
]

intervals = np.array([e["interval_ms"] for e in normal_all])
mean_int  = np.mean(intervals)
std_int   = np.std(intervals)
threshold = mean_int + 2 * std_int  # 平均+2SD以上を異常な間隔とする

slow_keys = [
    e for e in normal_all
    if e["interval_ms"] > threshold
]

print(f"\n全打鍵数: {len(normal_all)}件")
print(f"平均打鍵間隔: {mean_int:.1f}ms  SD: {std_int:.1f}ms")
print(f"異常間隔閾値（平均+2SD）: {threshold:.1f}ms")
print(f"異常間隔の打鍵数: {len(slow_keys)}件（{len(slow_keys)/len(normal_all)*100:.1f}%）")

print(f"\n異常間隔の打鍵サンプル（先頭10件）：")
print(f"{'キー':6s} {'ローマ字→ひらがな':15s} {'間隔ms':10s}")
print("-" * 35)
for e in slow_keys[:10]:
    hira = roman_to_hira(e["key"])
    print(f"  {e['key']:4s}  {e['key']}→{hira:8s}  {e['interval_ms']:6d}ms")

# JSON保存
output = {
    "trigram_bottleneck": trigrams[:30],
    "trigram_fastest":    fastest[:10],
    "slow_keystrokes":    len(slow_keys),
    "total_keystrokes":   len(normal_all),
    "threshold_ms":       round(threshold, 1)
}
with open("nlp_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("\n保存: nlp_analysis.json")