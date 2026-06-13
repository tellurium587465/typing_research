# タイピング研究システム

カメラ・マイク・キーロガーを同時録画し、**「どの指でどのキーをどのタイミングで打ったか」** をフレーム単位で計測・分析するシステムです。

---

## 基本の流れ

```
start.bat record  →  start.bat analyze  →  start.bat report
  （録音）               （全自動分析）          （結果確認）
```

---

## 新しいPC へのセットアップ（GitHub 経由）

```bat
REM 1. リポジトリをクローン
git clone https://github.com/tellurium587465/typing_research.git
cd typing_research

REM 2. セットアップを実行（1回だけ）
setup.bat
```

`setup.bat` が自動で行うこと：
- Python 仮想環境 (`.venv310`) の作成
- 依存パッケージのインストール
- MediaPipe モデル (`hand_landmarker.task`) のダウンロード
- デスクトップに `タイピング研究.bat` ランチャーを生成

以降はデスクトップの **`タイピング研究.bat`** をダブルクリックするだけ。

---

## セットアップ（開発環境を直接構築する場合）

### 必要なもの

- Windows 10/11
- Python 3.10
- バックライト付きキーボード（紫・ピンク系）
- 真上から撮影できるカメラ（USB / 内蔵）
- マイク（打鍵音が録れるもの）

### 環境構築

```bat
cd typing_research
python -m venv .venv310
.venv310\Scripts\pip install -r requirements.txt
```

---

## 使い方

### 1. 録音

```bat
start.bat record
```

- カメラプレビューが開く
- マイクは RMS 音量で自動選択
- タイピング開始 → **ESC で終了**
- `session_<timestamp>_video.avi / _audio.wav / _keys.json` を自動保存

### 2. 分析（1コマンドで全工程）

```bat
start.bat analyze
```

| Step | 内容 | 所要時間 |
|---|---|---|
| 1 | キーボード検出用フレーム自動抽出 | 数秒 |
| 2 | 紫バックライトでキー位置を検出 | 数秒 |
| 3 | MediaPipe で全フレームの指先を追跡 | **動画1分あたり約1〜2分** |
| 4 | 打鍵音 onset 検出・キーログと照合 | 数十秒 |
| 5 | キーログ × 音 × カメラを統合 | 数秒 |
| 6 | Excel / JASP 用 CSV 出力 | 数秒 |
| 7 | ターミナルにサマリー表示 | 即時 |

**オプション：**

```bat
start.bat analyze --from onset      # onset ステップから再実行
start.bat analyze --skip-pose       # pose をスキップして高速化（再実行時）
start.bat analyze --session-id 1778934176  # セッション指定
```

### 3. 結果確認

```bat
start.bat report          # 全セッションのサマリー
start.bat report --detail # キー別の詳細も表示
```

---

## 分析スクリプト

録音・分析以外に、個別で呼べる分析ツールがあります。

```bat
# 全セッション統合分析（依存なし・標準ライブラリだけで動く）
#   セッション別トレンド/疲労・3gram統計検定・キー別苦手スコア・
#   同指ビッグラム・母音パターンを1コマンドで横断集計
#   → consolidated_report.txt と jasp_trigram_ALL_clean.csv を出力
python consolidated_analysis.py
# 各分析の「なぜその手法か・何が分かるか」は
#   分析手法の選定理由と得られる知見.txt を参照

# 成長グラフ（WPM・打鍵間隔・運指一致率の推移）
.venv310\Scripts\python trend.py

# 苦手キーヒートマップ（練習優先度を色で表示）
.venv310\Scripts\python weakness.py

# 3gram 統計検定（左右交互 vs 同手 vs 同指の速度差）
.venv310\Scripts\python stats_test.py

# 運指最適化（どの指に切り替えると速くなるか）
.venv310\Scripts\python finger_optimize.py
```

---

## 計測できること

### キーログから直接計算

| 指標 | 内容 |
|---|---|
| WPM | フレーズ待機時間を除いた純粋な打鍵速度 |
| ミス率 | バックスペースで消した打鍵の割合 |
| 打鍵間隔 | キー間の時間（ms）。フレーズ境界は自動除外 |

### カメラ + 音声から推定

| 指標 | 精度 |
|---|---|
| カメラによる指先検出率 | **99〜100%**（タイムスタンプ補正後） |
| キー位置一致率 | **58〜84%** （ホームポジション付近は高精度） |
| 運指一致率 | **81〜91%** |

### 統計分析

- **3gram 分析**：連続3打鍵の速度・手パターン・母音パターン別集計
- **統計検定**：左右交互 vs 同手3連 vs 同指連打の速度差（Mann-Whitney U, Cohen's d）
- **運指最適化**：距離フィルタ + 信頼性フィルタで実践的な切り替え推奨を生成

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `session_*_integrated.json` | 打鍵1件ごとのフル分析（キー・指・タイミング） |
| `typing_analysis_*.xlsx` | Excel：打鍵ログ / 3gram / パターン集計 / 母音パターン |
| `jasp_trigram_ALL.csv` | JASP 用 CSV（フレーズ境界除外済み） |
| `trend.png` | セッション間の成長グラフ |
| `weakness_heatmap.png` | 苦手キーヒートマップ（赤=練習が必要） |
| `stats_test_result.png` | 統計検定の分布図 |
| `finger_optimize.png` | 運指最適化グラフ（遷移コスト行列・推奨） |

---

## 実測結果（例）

### セッション間の成長

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| 打鍵数 | 151 | 329 | 1963 | 1974 | 1337 |
| WPM | 89 | 76 | 111 | 110 | 104 |
| 打鍵間隔 | 145ms | 158ms | 107ms | 108ms | 114ms |
| ミス率 | 0.7% | 0.3% | 0% | 0% | 0% |

### 指ごとの運指一致率（全セッション合計）

| 指 | 使用回数 | 一致率 |
|---|---|---|
| 左小指（L1） | 641回 | 99.8% |
| 右人差し（R4） | 1351回 | 93.5% |
| 左人差し（L4） | 802回 | 89.5% |
| 右中指（R3） | 1179回 | 68.5% |
| 右薬指（R2） | 576回 | 72.6% |

### 3gram 統計検定結果

| パターン | 平均間隔 | vs 左右交互 |
|---|---|---|
| 左右交互 | 110.6ms | 基準 |
| 同手3連 | 116.8ms | +6ms |
| 同指連打 | 144.0ms | **+33ms（d=0.51, 効果量:中）** |

---

## パイプラインの技術的詳細

### タイムスタンプ同期

3つのセンサーが異なるタイムスタンプ軸を持つため、以下の補正を実施：

- **onset_ms**：`trim_start_ms` を加算して録画開始基準に統一
- **pose_ms**：VideoWriter の設定FPS(30) vs 実FPS(≈21.5) のズレを音声ファイル長から自動補正
  - 補正係数 ≈ 0.718（1.4倍圧縮を解消）
- **keylog_ms**：そのまま録画開始基準

### キーボード検出

1. HSV色空間で紫バックライトを抽出
2. `minAreaRect` で傾き角を計測
3. 5°以上の場合のみホモグラフィ透視補正を適用（不要な補正で精度が下がるため）
4. セッションごとに `session_*_keyboard_grid.json` として保存

### 指先照合フィルタ

**距離フィルタ**：各指のホームポジションからキーまでの距離（キー幅単位）で実用性を判定  
**信頼性フィルタ**：あるキーの全検出数の15%未満の指はカメラ誤検出として除外

```
例: I キーの検出
  R3（右中指）: 93% → 主指として採用
  R4（右人差し）: 5%  → 誤検出として除外
  R2（右薬指）:  2%  → 誤検出として除外
```

### 使用指の推定（多フレーム投票＋信頼度＋品質分類）

`integrate.py` は `finger_select.py` を使い、打鍵ごとに onset±150ms の**全フレームで投票**して使用指を決める（単一フレーム最近傍からの改良）。

- 票 = 時間の近さ × キーへの近さ。合計票が最大の指を採用 → 1フレームのジッタに強い
- 出力に `detection_confidence`（勝者票/総票）と `vote_margin`（2位との差）を追加
- `finger_quality` で **「本人の運指の癖」と「カメラ誤検出」を分離**：
  - `standard` … 高信頼かつ標準運指と一致
  - `nonstandard` … 高信頼だが標準と不一致（＝本人の運指の癖。計測したい本物）
  - `uncertain` … 低信頼（誤検出の疑い。集計から除外できる）

これにより「運指一致率が低い」を“癖”と“ノイズ”に切り分けられ、`uncertain` を
除いた**妥当な運指一致率**を報告できる。ロジックは `test_finger_select.py` で
合成データ単体テスト済み（カメラ実データ不要で検証可）。

#### 妥当性検証（誤検出率の定量化）

推定運指が本当に当たっているかを人手ラベルで検証する：

```bat
REM 情報量の多い打鍵を30件抽出（uncertain/nonstandard優先）
python validate_finger.py --sample 30 --session-id <id>
REM → sessions/<id>/validate_labels.csv の true_finger 列を録画を見て埋める
python validate_finger.py --score --session-id <id>
```

全体正解率・品質クラス別正解率・取り違え行列（R3↔R2 を自動フラグ）・信頼度較正を出す。
`uncertain` の正解率が低く `standard` が高ければ品質分類が妥当と確認できる。

#### 実験的オプション（検証してから有効化）

```bat
python integrate.py --use-z      # z深度の押下キューを加味
python integrate.py --reorder    # 指先x順序で R3↔R2 取り違えを補正
```

いずれも既定 OFF。`validate_finger.py` で ON/OFF の正解率を比べてから常用する想定。

### フレーズ境界の自動検出

寿司打等で次のお題が出るまでの待機時間が `interval_ms` に混入する問題を自動除外：

- `interval_ms > 1000ms` のキーを `is_phrase_start` フラグで管理
- トリグラムに `spans_boundary` 列を追加
- WPM・統計集計・JASP 用 CSV からは自動除外

---

## ファイル構成

```
typing_research/
├── recorder.py           # 録音（カメラ + マイク + キーロガー同時起動）
├── run_pipeline.py       # エンドツーエンド分析（Step 1〜7）
├── start.bat             # ランチャー
│
├── keyboard_detect.py    # キーボード位置・グリッド検出
├── pose_analysis.py      # MediaPipe による指先追跡
├── onset_detection.py    # 打鍵音タイミング検出
├── integrate.py          # 3ソース統合・指/キー判定
├── finger_select.py      # 使用指の多フレーム投票＋信頼度＋品質分類（純関数）
├── validate_finger.py    # 運指推定の妥当性検証（人手ラベルと突き合わせ）
├── test_finger_select.py # finger_select / validate の単体テスト（合成データ）
├── export_excel.py       # Excel / JASP CSV 出力
│
├── report.py             # 全セッションサマリー表示
├── trend.py              # セッション間成長グラフ
├── weakness.py           # 苦手キーヒートマップ
├── stats_test.py         # 3gram 統計検定
├── finger_optimize.py    # 運指最適化分析
├── plot_utils.py         # 日本語フォント設定ユーティリティ
│
├── session_utils.py      # セッションID・ファイルパス管理
├── fix_pose_timestamps.py # 既存データの FPS 補正（移行用）
│
├── requirements.txt      # 依存パッケージ
└── hand_landmarker.task  # MediaPipe モデルファイル
```

---

## 既知の限界

| 問題 | 状況 |
|---|---|
| マイクが拾えない環境 | onset が 0% になるが、キーログで代用して継続 |
| 上段・下段の端キー（y, b, w） | 指を伸ばすと指先がキー位置からずれやすい |
| 右中指・右薬指の運指特定 | 指の形状が似ており誤検出しやすい |
| カメラ実FPS ≈ 21.5fps | 自動補正済みだが、新録画時は meta.json で記録 |
| セッション数が少ない | finger_optimize は 50件以上の検出があると信頼性が上がる |

---

## 注意事項

- `recorder.py` 実行中はキーログが記録されます（パスワード等の入力は避ける）
- 録画データはローカルにのみ保存されます
- `hand_landmarker.task` は同フォルダに配置が必要です
