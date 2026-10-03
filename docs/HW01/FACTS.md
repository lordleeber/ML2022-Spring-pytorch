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
- `config.py` device = "cuda" 是**刻意設計**（fail fast and loud，不要 CPU fallback）；教材以中性描述，不列為問題或練習。import 時印 `True` 與 `0`。
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
- ch02：偽亂數（pseudo-random）與「每個函式庫各有獨立的亂數產生器」、cuDNN（deterministic／benchmark 旗標）、cuBLAS（一句帶過）、確定性演算法、docstring 與 `__doc__`、`torch.Generator`、`random_split`／`Subset`（indices）、duck typing、`int()` 截斷 vs 四捨五入、x（特徵）／y（目標）慣例、NumPy 整數陣列索引（fancy indexing）、`feat_idx`、全域產生器（沒指定 generator 時共用的那一個）、Dataset（只先粗略定義為「能 len() 與 [i] 的容器」，正式在 ch03）。
  ch02 圖號：圖 2.1 本章資料流、圖 2.2 random_split 打亂再切、圖 2.3 select_feat 欄位選擇。

## Baseline 實測（2026-10-03，與範例同一驗證集：random_split seed 5201314）
- 抄第 4 天 tested_positive（第 101 欄）：MSE 1.313
- 線性迴歸 4 個 tp 欄（53,69,85,101）：1.303
- 線性迴歸 116 欄（拿掉 id）：1.166；117 欄含 id：1.172
- 範例 DNN：1.661（見上）
- 全書開頭（index.html#now）已寫「過時三層次」；各章遇到過時寫法要加「現在的做法」框；ch07 要延伸 baseline 比較。

## 時間切分實測（每州依 id 排序，前 80% 訓練／後 20% 驗證）
- 2143／556 筆；線性迴歸 116 欄 1.21；抄第 4 天 1.241。
- 標準化（只用訓練集統計量）：訓練集 mean 0.0 / std 1.0；驗證集 0.011 / 1.002。
- uv 專案流程實測：uv init --bare + uv add torch==2.11.0 --index pytorch-cu128=… → 解析 30 個套件，lock 內 torch 2.11.0+cu128。

## 全書結構約定（使用者要求，2026-10-03）
- **每個 HW 的 ch00 都要有「模型總覽」一節**（架構圖、各層 shape、參數量、所在檔案）；模型的逐行細講仍在後面的模型章。HW01：ch00 §0.2，圖 0.1；原檔案地圖改為圖 0.2、§0.8。

## ch02 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
執行方式：在 `HW01/` 內 `PYTHONPATH=. ../.venv/bin/python <腳本>`，import `utils.py` 的函式。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

`train.py` 開頭的 stdout，逐字照抄，含行尾空白與縮排（後面接 tqdm 進度條和 Epoch 1 的 loss，見上方「實測執行」）：
```
True
0
train_data size: (2160, 118) 
    valid_data size: (539, 118) 
    test_data size: (1078, 117)
number of features: 117
```
- 第 1、2 行來自 `config.py` import 時的 print。`train_data size` 那行行尾有一個空格，下一行開頭有 4 格縮排，因為 f-string 用三引號跨行。

same_seed：
- docstring 寫成 `""""`（4 個引號），所以 `__doc__` 實際是 `'" Fixes random number generator seeds for reproducibility. '`，開頭多一個 `"`。
- cudnn 旗標：呼叫前 deterministic=False、benchmark=False；呼叫後 deterministic=True、benchmark=False。
- 呼叫 `same_seed(5201314)` 後，`torch.rand(3)` = [0.5436, 0.9728, 0.8315]，`np.random.rand(3)` = [0.6076, 0.9134, 0.281]。重設 seed 再取一次，結果完全相同。
- **不會**設定 Python 內建的 `random` 模組，因為 same_seed 沒有呼叫 random.seed。
- 有 `torch.cuda.is_available()` 判斷，在沒 GPU 的機器上也不會報錯。但 config.py 的 device="cuda" 在別處會失敗，見上方。

train_valid_split（seed 5201314）：
- `0.2*2699` = 539.8000000000001，經 `int()` 截斷成 539，不是四捨五入。train 2160。
- `random_split` 的回傳型別是 `Subset`，長度 2160 和 539。
- train indices 前 10 個：[696, 2073, 909, 1475, 2165, 2487, 479, 1795, 2150, 2588]（未排序）。
- valid indices 前 10 個：[1995, 1042, 86, 1667, 464, 481, 1040, 85, 423, 788]。因為 id 欄 = 列號，valid 的 id 前 10 個也是這組數字。
- 同一個 seed 重切一次，結果相同。改用 seed 1 時，train 前 10 個是 [1659, 792, 971, 1755, 777, 408, 450, 248, 1528, 1631]。
- 先改全域 `torch.manual_seed(0)` 再切，結果仍與原本相同，因為切分用的是獨立的 Generator。
- 不傳 generator 時，切分依賴全域 seed：全域 seed 相同，切分就相同；全域 seed 不同，切分就不同。
- `random_split(data, [0.8, 0.2], ...)`（比例寫法，torch ≥1.13 支援）切出來也是 [2160, 539]。
- `np.array(Subset)` 得到 (2160,118) float64，耗時約 0.0015 s。`data[subset.indices]` 結果完全相同（array_equal True），耗時約 0.0012 s。這組資料量太小，兩種寫法的速度看不出差別。
- valid 裡 37 州都有出現，每州 7–25 列。
- y_valid 平均 10.1355、範圍 0.3448–29.8157；y_train 平均 9.7404。

select_feat：
- select_all=True：x_train (2160,117)、x_valid (539,117)、x_test (1078,117)、y_train (2160,)、y_valid (539,)。
- select_all=False：feat_idx [0,1,2,3,4] = 欄名 ['id','AL','AK','AZ','AR']，也就是 id 加 4 個州的 one-hot，全部不是有用的特徵。shape 分別是 (2160,5) (539,5) (1078,5)。
- test 沒有答案欄，所以 `raw_x_test = test_data` 直接使用全部 117 欄。train[:, :-1] 也是 117 欄，兩邊對齊；第 116 欄是 worried_finances.4。

## ch02 審稿補測（2026-10-03 本機）
- y_train：平均 9.7404、範圍 0.4545–30.3046。
- 動手做第 1 題逐字輸出：`(2160, 118) (539, 118)`，接著 `[1995. 1042.   86. 1667.  464.  481. 1040.   85.  423.  788.]`。
- 先執行 `torch.manual_seed(0)` 再切，valid 前 10 個 id 不變。seed 1 時 valid 前 10 個 id：`[1265. 2071.  158. 1588. 1822. 1955. 1547.  969.  476.  856.]`。
- **拿掉 id**（utils.py:32 改成 `feat_idx = list(range(1, raw_x_train.shape[1]))`）：116 個特徵；最佳 valid loss 0.982，出現在 `Epoch [1369/3000]: Train loss: 1.0857, Valid loss: 0.9820`；第 1769 個 epoch early stop；共存檔 55 次。
- **保留 id 但除以 2699**（117 欄）：最佳 valid loss 1.003，在第 1367 個 epoch。結論：id 拖累的主因是數值尺度太大（id 最大 2699，次大的欄位是 wearing_mask 的 89.8），程式又沒有做標準化。
- 這兩組實驗都在暫存複本裡跑，repo 的 models/model.ckpt 沒有被動到。
