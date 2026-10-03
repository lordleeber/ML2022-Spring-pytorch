# HW01 教材事實清單（維護筆記，不進教材）

教材對應 commit：`717528b`（HW01 程式碼最後變動於 `81d1817`）。
原始碼根目錄：`HW01/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW01/HW01.ipynb`。

## 全書約定
- 檔名：index, outline, ch00–ch07, appendix。
- 圖例「色彩 = 角色」：
  - 藍 `#58a6ff` = CPU 端資料（pandas / numpy / Dataset）
  - 綠 `#3fb950` = GPU 上的運算（模型、tensor 計算）
  - 金 `#e3b341` = 磁碟檔案（csv、model.ckpt、pred.csv、runs/）
  - 紫 `#a371f7` = 控制與超參數（config、迴圈控制）
- Python 程式碼用 `<pre class="py">`。

## 環境（實測 2026-10-02）
- Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.3、pandas 3.0.6。
- GPU：NVIDIA RTX PRO 4000 Blackwell（sm_120），WSL2。原 requirements 的 torch 1.13.1+cu116 不支援。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW01/` 內用 `../.venv/bin/python train.py`。

## 資料
- covid.train.csv：2699 列 × 118 欄；covid.test.csv：1078 列 × 117 欄（少最後一欄 tested_positive）。
- 欄位：0 = id；1–37 = 37 州 one-hot（AL…WA，每列恰好一個 1）；
  38–53 = 第 1 天 16 欄；54–69 第 2 天；70–85 第 3 天；86–101 第 4 天；102–117 第 5 天。
- 每天 16 欄 = cli, ili, hh_cmnty_cli, nohh_cmnty_cli（4 類症狀）+ 8 行為 + 3 心理 + tested_positive。
- pandas 讀進來重複欄名會改成 `cli.1`、`cli.2`…
- train id：最小 0、最大 2699（2699 列 → 中間缺一號）；test id 0–1077。
- tested_positive（第 1 天欄）平均 9.60、範圍 0.345–30.30。

## 程式事實
- `select_all=True` → feat_idx = range(117)，**含 id 欄**。
- `train_valid_split`：valid = int(0.2 × 2699) = 539；train = 2160。註解寫 `train_size * valid_ratio` 是錯的。
- `random_split` 用 `torch.Generator().manual_seed(seed)`，與全域 seed 無關。
- 回傳 `np.array(Subset)` → 逐筆取出組成 (2160, 118) 陣列。
- batch_size 256 → train 每 epoch 9 個 batch（8 滿 + 1 個 112）；valid 3 個 batch（256,256,27）；test 5 個 batch。
- 模型參數量：117·16+16 = 1888；16·8+8 = 136；8·1+1 = 9；共 2033。model.ckpt 11005 bytes。
- 優化器 SGD lr 1e-5 momentum 0.9；loss MSELoss(mean)。
- early_stop 400：連續 400 個 epoch valid loss 沒有創新低就停。
- `config.py` device 寫死 `"cuda"`，import 時印 `True` 與 `0`（原版為 cuda/cpu fallback）。
- `valid_loader` shuffle=True（不影響 loss 平均值？每 batch 平均再平均，batch 大小不同時結果會隨分組微變）。
- `train.py` 重複 import tqdm；`random_split` 在 train.py import 但沒用。
- predict.py 為了拿 input_dim 重讀並切分 train。

## 實測執行（2026-10-02 21:34，seed 5201314）
- train.py：real 34.2 s；Epoch 1 train 134.2442 / valid 107.2155；Epoch 2 69.8929 / 50.8182。
- 最佳 valid loss 1.6611 於 epoch 1483；epoch 1883 early stop；共存檔 49 次。
- 兩次執行結果相同（17:39 與 21:34 都是 1.661 / 1883）。
- predict.py：5 個 batch，pred.csv 1079 行（header + 1078）。
- runs/ 下產生 `Oct02_21-34-17_ValtecBlackwell` 這類目錄（SummaryWriter 預設命名：月日_時-分-秒_主機名）。
- 驗證 loss 軌跡：ep20 12.4151、ep50 10.0611、ep100 4.2813、ep300 2.6302、ep1000 3.8595。
- 沒 GPU（CUDA_VISIBLE_DEVICES=""）時 `import config`：先印 False，再 `RuntimeError: No CUDA GPUs are available`。
- covid.train.csv / covid.test.csv 雖被 HW01/.gitignore 的 `*.csv` 涵蓋，但早已 commit，git 仍追蹤。

## ch01 查證（2026-10-02）
- 各州列數 53–91（平均 ~73）；第 5 天陽性率州平均最低 CT 3.23、最高 MS 22.15。
- 無缺值；.values → numpy float64 (2699,118)。pandas 後綴：無=第1天、.1=第2天…、.4=第5天。
- **滑動視窗**：同州按 id 排序，相鄰列錯開一天（29/37 州完全連續；其餘 8 州各 1–2 處斷）。FL：id 0,32,63,67…
- 缺號 id 2239。test id 0–1077 另一套編號。test 第 1 天問卷出現在 train 的只有 36/1078 列。
- 與答案（tested_positive.4）相關：tp.3 0.985、tp.2 0.970、tp.1 0.953、tp 0.935、hh_cmnty_cli.4 0.898；id 0.264；wearing_mask.4 −0.037。
- 圖例另加：紅 `#f85149` = 預測目標（答案欄）。

## 已在前面章節定義過的名詞（後續章節不必重講，可簡短回指）
- ch00：迴歸、MSE、Kaggle 與四條基準線、PyTorch、CUDA、compute capability/sm、kernel、venv、uv、
  超參數、seed（亂數種子）、驗證集/訓練集/測試集、epoch、batch、early stopping（概念）、
  weights/參數、checkpoint、stdout/stderr、`2>`、`VAR=x cmd`、`from X import *`、Jupyter 筆記本。
- ch01：pandas 基本語法（read_csv、d[col]、iloc、布林篩選、groupby、corr）、python -c 多行、np.isclose、one-hot、滑動視窗、相關係數（Pearson）、標準化（概念）、pandas DataFrame、`.values`、重複欄名後綴、`if __name__ == '__main__'`、0 起算 vs 1 起算欄號。

## Baseline 實測（2026-10-03，與範例同一驗證集：random_split seed 5201314）
- 抄第 4 天 tested_positive（第 101 欄）：MSE 1.313
- 線性迴歸 4 個 tp 欄（53,69,85,101）：1.303
- 線性迴歸 116 欄（拿掉 id）：1.166；117 欄含 id：1.172
- 範例 DNN：1.661（見上）
- 全書開頭（index.html#now）已寫「過時三層次」；各章遇到過時寫法要加「現在的做法」框；ch07 要延伸 baseline 比較。
