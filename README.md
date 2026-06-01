# タイピング研究システム

カメラ・マイク・キーロガーを同時に使い、**「どの指でどのキーをどのタイミングで打っているか」** をフレーム単位で計測・分析するシステムです。

---

## このシステムで何がわかるか

### 計測できること

| 項目 | 内容 | 精度 |
|---|---|---|
| **タイピング速度** | WPM（1分あたりの語数） | キーログから直接計算 |
| **ミス率** | バックスペースで消した打鍵の割合 | キーログから直接計算 |
| **打鍵タイミング** | キー間の間隔（ms） | ±1ms 精度 |
| **使用した指** | 実際にどの指でキーを打ったか | 約80〜90% の正解率 |
| **指先の着地位置** | キー中心から何px ずれたか | フレーム精度（±10px程度） |
| **標準運指との差** | 教科書通りの運指と自分の運指のズレ | 打鍵ごとに判定 |
| **3連打パターン** | 連続3打鍵ごとの速度・指パターン | JASP 用 CSV として出力 |

### 具体的に読み取れる知見（実データより）

```
右中指（R3）の運指一致率が 68%
→ i・k など右中指担当キーを別の指で打っている

左小指（L1）の運指一致率が 99%
→ a・q など左小指担当キーは正確に担当指で打っている

a・s キーのキー位置一致率 98〜99%
→ ホームポジションは安定している

y・b キーのキー位置一致率 14〜36%
→ 上段・下段の端のキーで指先位置がズレやすい
```

### 出力されるファイル

| ファイル | 内容 |
|---|---|
| `session_*_integrated.json` | 打鍵1件ごとのフル分析データ（キー・指・タイミング・一致率） |
| `typing_analysis_*.xlsx` | Excel：打鍵ログ / 3gram全データ / パターン集計 / 母音パターン |
| `jasp_trigram_*.csv` | JASP 統計ソフト用 CSV（3連打データ） |
| `keyboard_detect.png` | キーボードグリッド検出の可視化 |
| `onset_result.png` | 打鍵音と波形の対応図 |
| `session_*_keyboard_grid.json` | このセッションのキー座標マップ |

---

## システム構成

```
カメラ（真上から撮影）
    ↓ 動画 (.avi)
マイク
    ↓ 音声 (.wav)      → recorder.py で同時録画
キーボード
    ↓ キーログ (.json)
    
    ↓ run_pipeline.py で自動分析
    
[Step 1] extract_frame   → 手が映っていないフレームを自動選択
[Step 2] keyboard_detect → 紫バックライトでキー位置を検出
[Step 3] pose_analysis   → MediaPipe で全フレームの指先を追跡
[Step 4] onset_detection → 打鍵音からタイミング検出・キーログ照合
[Step 5] integrate       → 3ソースを1打鍵ずつ照合・統合
[Step 6] export_excel    → Excel / CSV 出力
[Step 7] report          → ターミナルにサマリー表示
```

---

## セットアップ

### 必要なもの

- Python 3.10
- バックライト付きキーボード（紫・ピンク系）
- 真上から撮影できるカメラ（USB / 内蔵）
- マイク（キーボードの打鍵音が録れるもの）

### 環境構築

```bat
cd typing_research
python -m venv .venv310
.venv310\Scripts\pip install -r requirements.txt
```

---

## 使い方（エンドツーエンド）

### 1. 録音

```bat
start.bat record
```

または

```bat
.venv310\Scripts\python recorder.py
```

- カメラプレビューが開く
- マイクデバイスを自動選択（RMS音量が最大のもの）
- タイピングを開始する
- **ESC キーで終了**
- `session_<タイムスタンプ>_video.avi / _audio.wav / _keys.json` が保存される

### 2. 分析（1コマンド）

```bat
start.bat analyze
```

または

```bat
.venv310\Scripts\python run_pipeline.py
```

最新のセッションを自動検出して全工程（Step 1〜7）を実行します。  
動画が長い場合、Step 3（骨格推定）に数分かかります。

**途中から再実行したい場合：**

```bat
# onsetステップから再実行
.venv310\Scripts\python run_pipeline.py --from onset

# セッションを指定
.venv310\Scripts\python run_pipeline.py --session-id 1778934176

# pose をスキップして高速化（既に完了済みのとき）
.venv310\Scripts\python run_pipeline.py --skip-pose
```

### 3. レポート確認

```bat
start.bat report
```

または

```bat
# 全セッションのサマリー
.venv310\Scripts\python report.py

# キー別の詳細も表示
.venv310\Scripts\python report.py --detail
```

---

## パイプライン各ステップの詳細

### Step 1: フレーム抽出（`run_pipeline.py` 内）

動画の最初の2秒から、キーボードのバックライト（紫）が最もよく見えるフレームを自動選択します。手が映っていないフレームが理想です。

### Step 2: キーボード検出（`keyboard_detect.py`）

HSV色空間で紫バックライトを抽出し、キーボード全体の輪郭と各キーの位置を検出します。

- 傾き角を minAreaRect で計測し、5°以上の場合はホモグラフィ透視補正を適用
- 各キーの中心座標と矩形領域（key_rects）を JSON に保存
- セッションごとに `session_*_keyboard_grid.json` として保存（カメラ位置がずれても対応可）

### Step 3: 骨格推定（`pose_analysis.py`）

MediaPipe の HandLandmarker を使い、全フレームの両手の指先座標（5本 × 2手）を取得します。

- 実際の録画 FPS（約21.5fps）を音声ファイル長から算出してタイムスタンプを補正
- `session_*_pose.json` に保存（フレームごとの指先ピクセル座標）

### Step 4: onset 検出（`onset_detection.py`）

librosa で打鍵音の立ち上がり（onset）を検出し、キーログのタイムスタンプと照合します。

- 音声を正規化後、onset を検出（wait=5フレーム≒115ms、delta=0.07）
- キーログとの1対1マッチング（許容誤差±80ms）
- **音声が小さい・マイクエラーの場合は onset なしで進む**（Step 5 でキーログ直接使用）

### Step 5: 統合（`integrate.py`）

打鍵1件ごとに「実際に使った指」「指先がどのキーの上にあったか」を判定します。

**判定ロジック：**
1. onset_ms（打鍵時刻）前後 ±150ms の pose フレームを取得
2. 標準運指の担当キー座標から 2キー幅以内の指先を候補とする
3. `距離 + 時間差 × 0.5` のスコアで最良の指を選択
4. 指先が key_rect 内なら `key_match=True`（外でも最近傍1.5キー以内なら代用）

**出力フィールド（per 打鍵）：**

| フィールド | 内容 |
|---|---|
| `key` | 実際に押されたキー |
| `std_finger` | 標準運指の担当指（L1〜R1） |
| `actual_finger` | カメラで検出した実際の指 |
| `finger_match` | 標準運指と一致したか |
| `detected_key` | 指先が最も近かったキー |
| `key_match` | 指先が正しいキーの上にあったか |
| `dist_to_key` | 指先とキー中心の距離（px） |
| `interval_ms` | 前の打鍵からの間隔（ms） |
| `is_error` | この打鍵の後にバックスペースが来たか |
| `source` | `camera`（カメラ検出成功）/ `std_only`（キーログのみ） |

### Step 6: Excel 出力（`export_excel.py`）

| シート | 内容 |
|---|---|
| 打鍵ログ | 全打鍵のキー・指・タイミング |
| 3gram_全データ | 連続3打鍵ごとの詳細（手パターン・母音パターン等） |
| 3gram_集計 | 指パターン別の平均間隔・エラー率 |
| パターン別サマリ | 左右交互 / 同手3連 / 同指連打 の比較 |
| 母音パターン別サマリ | CCC/CCV/CVC/VCC 等の速度比較 |

### Step 7: レポート（`report.py`）

全セッションの集計をターミナルに表示します。

---

## 精度・限界

### 現時点の精度（実測値）

| 指標 | 精度 |
|---|---|
| カメラによる指先検出率 | **99〜100%**（タイムスタンプ補正後） |
| 打鍵音マッチング率（onset） | 0〜100%（マイク品質に依存） |
| キー位置一致率 | **58〜84%** |
| 運指一致率 | **81〜91%** |

### 限界・既知の問題

| 問題 | 状況 |
|---|---|
| マイクが拾えない環境では onset が動かない | キーログ代用で継続可能 |
| 上段・下段の端キー（y, b, w）はキー位置誤差が大きい | 指が伸びた状態を真上から見ると誤差が増える |
| 右中指（R3）・右薬指（R2）の運指特定精度が低い | 指の形状が似ているため |
| カメラのフレームレートは約21.5fps（設定30fpsより低い） | タイムスタンプ自動補正で対応済み |
| 解析には動画が必要（後処理は可能だが長時間かかる） | 1分の動画で pose 解析約1〜2分 |

---

## ファイル構成

```
typing_research/
├── recorder.py          # 録音（カメラ + マイク + キーロガー同時起動）
├── run_pipeline.py      # エンドツーエンド分析（Step 1〜7）
├── report.py            # 全セッションのサマリー表示
├── start.bat            # ランチャー（record / analyze / report）
│
├── keyboard_detect.py   # キーボード位置・キーグリッド検出
├── pose_analysis.py     # MediaPipe による指先追跡
├── onset_detection.py   # 打鍵音タイミング検出
├── integrate.py         # 3ソース統合・指 / キー判定
├── export_excel.py      # Excel / CSV 出力
│
├── session_utils.py     # セッションID・ファイルパス管理
├── fix_pose_timestamps.py  # 既存データのFPS補正（移行用）
│
├── heatmap.py           # キーボードヒートマップ可視化
├── char_heatmap.py      # キー別ヒートマップ
├── finger_velocity.py   # 指ごとの打鍵速度分析
├── show_trigram.py      # 3gram散布図
│
├── requirements.txt     # 依存パッケージ
└── hand_landmarker.task # MediaPipe モデルファイル
```

---

## 研究上の活用例

- **自分の運指の客観的把握**：感覚ではなくカメラデータで「どの指でどのキーを打っているか」を記録
- **練習効果の追跡**：セッションをまたいで WPM・ミス率・運指一致率の変化を追う
- **3gram 分析**：左右交互・同手3連・同指連打でタイピング速度がどう変わるかを JASP で統計検定
- **個人差の記録**：標準運指との差がどのパターンで生じているかを視覚化

---

## 注意事項

- `recorder.py` 実行中はキーログが記録されます（パスワード等の入力は避けること）
- 録画データ（動画・音声）はローカルにのみ保存されます
- MediaPipe モデル（`hand_landmarker.task`）は同フォルダに配置が必要です
