# HW02 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW02/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **寫作分工**：本機（有 GPU）把全書需要的數字一次量完，寫在這裡（使用者 2026-10-04 選「一次量完」）；雲端 session 沒有 GPU、沒有資料、不跑程式，只引用這裡的數字寫章。這裡找不到的，標 `<!-- TODO(本機實測): 要量什麼 -->`，PR 回來後由本機補。
>
> **重現方法**（需要 GPU 與 `HW02/libriphone/`）：在 `HW02/` 裡跑
> - `PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw02_facts.py [env data split concat memory shapes model ckpt]`：資料、形狀、模型、checkpoint 評估（唯讀）。
> - `PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw02_exp.py --name X [選項]`：訓練實驗，預設參數就是 train.py，不寫檔（除非給 `--save`）。

教材對應 commit：`7f11c7e`（HW02 程式最後一次改動是 `a517359`，之後沒改過）。
原始碼根目錄：`HW02/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW02/HW02.ipynb`（Colab notebook，不在 repo 裡，下面「原版 vs 本 repo」列了差異）。作業投影片：`HW02/hw2_slides 2022.pdf`（25 頁，已在 repo）。

## 全書約定
- 樣式：用 mySkills **新版共用資產**（`completed-repo-to-html-textbook/assets/style.css`、`enhance.js`；與 docs/HW09 相同），使用者 2026-10-04 選的。**不要**複製 docs/HW01 的舊版。
- 指令塊：讀者要貼上執行的用 `<pre class="shell cmd">`，輸出用 `<pre class="shell">`。Python 用 `<pre class="py">`。
- listing 的 `data-hot` 寫**原始碼行號**（與 figcaption `檔名:起–迄` 同一套）。
- 檢查：`python3 docs/tools/verify_book.py . docs/HW02/chNN.html`（依 HTML 所在資料夾找原始碼，`docs/HW02/x.html` 引用 `HW02/<file>`）。
- 每本書的規則（使用者指定）：ch00 要有「模型總覽」（架構 SVG、每層 tensor 形狀、參數量：手算 + PyTorch 印出、哪個檔案定義模型），放在任務說明之後；目錄頁有「2022 vs 現在」導論；2022 寫法過時的地方加「現在的做法」框。參考 docs/HW01 的 ch00 §0.2、圖 0.1。
- 大綱：`docs/HW02/outline.html`（使用者 2026-10-04 核可：ch00–ch07 + appendix）。
- ch07 實驗規格（使用者 2026-10-04 選）：所有對照組用**同一套縮小規格**並排比，另附一次原規格（20 epoch）的完整紀錄。縮小規格見「ch07 實測」。

## 環境（實測 2026-10-04）
- Python 3.12.3、torch 2.11.0+cu128、numpy 2.5.3、tqdm 4.70.1、cuDNN 91900。
- GPU：NVIDIA RTX PRO 4000 Blackwell（23.9 GiB），WSL2；RAM 47 GB。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW02/` 裡用 `../.venv/bin/python train.py`。`train.py:12`、`predict.py:15` 寫死 `device = "cuda"`，**這是刻意的設計**，教材以中性描述，不列為問題或練習。

## 資料與檔案
- 來源：本機 Kaggle zip `ml2022spring-hw2.zip`（480,098,274 bytes，Windows Documents/poyi/ml_2022_data；zip 內是巢狀的 `libriphone/libriphone/`）。解到 `HW02/libriphone/`，約 511 MB。官方 notebook 用 `wget` 從 GitHub release 下載 `libriphone.zip`。
- `HW02/.gitignore`：`libriphone/*`、`*.csv`、`*.ckpt`。但 `HW02/prediction.csv`（6,066,399 bytes，646,269 行）**在 .gitignore 之前就被加進 git，所以仍被追蹤**；重跑 predict.py 會改到它。`HW02/model.ckpt`（80,264,975 bytes）沒被追蹤。
- `libriphone/` 的內容：
  - `train_split.txt` 69,380 bytes、4286 行（訓練＋驗證的句子 id）；`train_labels.txt` 6,903,008 bytes、4286 行；`test_split.txt` 17,440 bytes、1078 行。（`wc -l` 印 4285／1077，因為最後一行沒有換行字元。）
  - `feat/train/` 4286 個 `.pt`、`feat/test/` 1078 個 `.pt`。
  - 句子 id 格式 `說話者-章節-句號`，例如 `2007-149877-0023`。前 3 個訓練 id：`2007-149877-0023`、`60-121082-0044`、`5688-41232-0018`；前 3 個測試 id：`1963-142776-0022`、`1841-150351-0006`、`481-123720-0082`。
- `train_labels.txt` 一行一句：`句子id 標籤 標籤 …`，每個音框一個 0–40 的整數。第一行開頭：`2007-149877-0023 0 0 0 …（45 個 0）… 29 29 29 29 29 29 39 39 …`。
- 每個 `.pt` 是 `torch.load` 讀出來的 float32 tensor，形狀 (T, 39)。例：`2007-149877-0023.pt` 是 (500, 39)，檔案 78,699 bytes；第 0 格前 8 維 `[-2.0803, -0.7538, -0.4708, -0.1630, 0.5476, -0.4360, 1.2979, -0.3249]`。
- 每一句的 39 維**各自**平均 ≈ 0（全部句子、全部維度 |平均| 最大 4.6e-07）、標準差 ≈ 1（0.99999994–1.00000012）：投影片 p.11 說的「39-dim MFCC w/ CMVN」，CMVN 是逐句做的。
- 每一句的音框數 T 與標籤數完全相同（4286 句全部核過）。
- 音框數：訓練＋驗證 **2,644,158** 格；T 最小 139、p10 293、中位數 620、平均 616.9、p90 933、最大 998。測試 **646,268** 格；T 最小 176、中位數 598、最大 998。
- **投影片與資料不一致**：投影片 p.10 寫「Training: 4268 preprocessed audio features with labels (total 2644158 frames)」，實際是 **4286** 句；音框數 2,644,158 一致（4268 應是 4286 的筆誤）。測試「1078 … total 646268 frames」一致。
- 41 類（id 0–40）。投影片與 repo 都**沒有**給 id 對應的音素名稱，教材不要自己編。各類音框數（全部 4286 句）：

  | id | 音框數 | 比例 | | id | 音框數 | 比例 | | id | 音框數 | 比例 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 460,513 | 0.1742 | | 37 | 83,036 | 0.0314 | | 9 | 41,980 | 0.0159 |
  | 31 | 159,347 | 0.0603 | | 36 | 75,033 | 0.0284 | | 29 | 40,291 | 0.0152 |
  | 2 | 139,635 | 0.0528 | | 25 | 72,454 | 0.0274 | | 26 | 39,782 | 0.0150 |
  | 4 | 131,318 | 0.0497 | | 8 | 69,750 | 0.0264 | | 24 | 35,515 | 0.0134 |
  | 5 | 126,718 | 0.0479 | | 10 | 67,376 | 0.0255 | | 3 | 34,806 | 0.0132 |
  | 27 | 104,399 | 0.0395 | | 28 | 64,611 | 0.0244 | | 15 | 30,466 | 0.0115 |
  | 39 | 97,567 | 0.0369 | | 6 | 64,547 | 0.0244 | | 13 | 26,633 | 0.0101 |
  | 30 | 88,165 | 0.0333 | | 40 | 61,729 | 0.0233 | | 11 | 26,191 | 0.0099 |
  | 19 | 85,522 | 0.0323 | | 14 | 52,479 | 0.0198 | | 34 | 23,968 | 0.0091 |
  | | | | | 23 | 47,634 | 0.0180 | | 21 | 18,381 | 0.0070 |
  | | | | | 35 | 47,302 | 0.0179 | | 16 | 17,678 | 0.0067 |
  | | | | | 12 | 46,158 | 0.0175 | | 38 | 16,303 | 0.0062 |
  | | | | | 32 | 44,775 | 0.0169 | | 1 | 15,485 | 0.0059 |
  | | | | | 33 | 44,615 | 0.0169 | | 7 | 14,095 | 0.0053 |
  | | | | | | | | | 18 | 13,496 | 0.0051 |
  | | | | | | | | | 22 | 7,696 | 0.0029 |
  | | | | | | | | | 17 | 5,365 | 0.0020 |
  | | | | | | | | | 20 | 1,344 | 0.0005 |

  （由多到少，分三欄排；最多的第 0 類 460,513 格，最少的第 20 類 1,344 格，差 343 倍。）
- 第 0 類：4286 句裡有 4281 句**第一格**是 0、4244 句**最後一格**是 0。據此推測它是靜音／停頓，但**資料沒有標明**，教材要寫成推測。
- 連續相同標籤的一段（segment）：共 276,010 段；每段平均 9.58 格、中位數 7、p90 16、最大 305；**73.12% 的段短於 11 格**（也就是 11 格的視窗常常橫跨兩個以上的音素）。
- 投影片 p.8：每格 25 ms（投影片原文 "each frame only contains 25 ms of speech"）。投影片沒寫 frame shift，教材不要自己補。

## 切分（utils.py:54-60，config.py:5）
- `train_ratio = 0.9`（config.py:5）會傳進 `preprocess_data`，蓋掉函式預設的 0.8（utils.py:42）。`train_val_seed=1337` 用預設值。
- 以**句**為單位切：`random.seed(1337)` 後 `random.shuffle` 4286 行，前 `int(4286*0.9)=3857` 句當訓練，後 429 句當驗證。
  - 訓練 3,857 句、**2,379,588** 格；驗證 429 句、**264,570** 格（與 train.py 印出的形狀一致）。
  - 前 3 個驗證 id：`5049-25947-0112`、`1898-145724-0020`、`6019-3185-0096`。
- 說話者：訓練 250 人、驗證 184 人、測試 231 人；**驗證的 184 人全部也出現在訓練集；測試的 231 人也全部出現在訓練＋驗證**。所以驗證集量的是「看過的說話者、沒看過的句子」，與測試集同一種情況。
- 驗證集多數類別：第 0 類 46,898 格，佔 **0.177261**。也就是「全部猜 0」在驗證集上的準確率是 0.177261（多數類別基準）。
- 官方 sample 的比例 0.8：訓練 3,428 句、2,116,368 格；驗證 858 句、527,790 格（與本 repo 的驗證集不同）。

## 拼接（utils.py:13-39；ch02）
- `hw02_facts.py concat` 的逐字輸出（T=4、每格 2 維的小例子）：
  ```
  x (T=4, dim=2):
  tensor([[1., 2.],
          [3., 4.],
          [5., 6.],
          [7., 8.]])
  shift(x, 1):
  tensor([[3., 4.],
          [5., 6.],
          [7., 8.],
          [7., 8.]])
  shift(x, -1):
  tensor([[1., 2.],
          [1., 2.],
          [3., 4.],
          [5., 6.]])
  concat_feat(x, 3):
  tensor([[1., 2., 1., 2., 3., 4.],
          [1., 2., 3., 4., 5., 6.],
          [3., 4., 5., 6., 7., 8.],
          [5., 6., 7., 8., 7., 8.]])
  concat_feat(x, 5):
  tensor([[1., 2., 1., 2., 1., 2., 3., 4., 5., 6.],
          [1., 2., 1., 2., 3., 4., 5., 6., 7., 8.],
          [1., 2., 3., 4., 5., 6., 7., 8., 7., 8.],
          [3., 4., 5., 6., 7., 8., 7., 8., 7., 8.]])
  ```
  注意 `shift(x, n)` 的 n>0 是「往後看 n 格」（第 t 列變成第 t+n 格，尾端用最後一格補），n<0 是往前看。concat 後每一列由左到右是第 t−k … t … t+k 格。
- 標籤也拼（本 repo 的改動，utils.py:83-84）：`2007-149877-0023` 拼 11 格後第 44–50 列：
  ```
  tensor([[ 0,  0,  0,  0,  0,  0, 29, 29, 29, 29, 29],
          [ 0,  0,  0,  0,  0, 29, 29, 29, 29, 29, 29],
          [ 0,  0,  0,  0, 29, 29, 29, 29, 29, 29, 39],
          [ 0,  0,  0, 29, 29, 29, 29, 29, 29, 39, 39],
          [ 0,  0, 29, 29, 29, 29, 29, 29, 39, 39, 39],
          [ 0, 29, 29, 29, 29, 29, 29, 39, 39, 39, 39],
          [29, 29, 29, 29, 29, 29, 39, 39, 39, 39, 39]])
  ```
  第 j 欄 = 第 t−5+j 格的標籤，中間第 5 欄（0 起算）就是原本那一格的標籤。
- 實際特徵：(500, 39) → (500, 429)；第 0 列的前 6 個 39 維區塊（第 −5..0 格）都等於第 0 格（句首用第 0 格補）；第 10 列的最後一塊（第 10 塊）等於第 15 格。

## 記憶體（utils.py:69-73、86-95；ch02）
- `torch.empty(3000000, 429)` float32：5,148,000,000 bytes（4.794 GiB）。標籤緩衝 `torch.empty(3000000, 11, dtype=torch.long)`：264,000,000 bytes（0.246 GiB）。
- `X = X[:idx, :]` 是 view，storage 仍是整塊 5,148,000,000 bytes（`untyped_storage().nbytes()` 實測）；驗證集實際只用到 454,002,120 bytes（264,570 × 429 × 4）。`del train_X, …` 刪掉的只是名字，Dataset 還握著同一塊 storage。
- **但沒寫到的部分不佔實體記憶體**：`torch.empty` 只向作業系統要虛擬位址，頁面第一次被寫入才真的配置。train.py 實測峰值 RSS（`/usr/bin/time -v` 的 Maximum resident set size）是 **6,029,716 KB ≈ 6.0 GB**，接近實際寫入的量：訓練特徵 4,083,373,008 bytes（2,379,588 × 429 × 4）＋驗證特徵 454,002,120 ＋兩份標籤（(2,379,588 + 264,570) × 11 × 8 = 232,685,904）≈ 4.77 GB，再加上 PyTorch/CUDA 本身。**不要寫成「佔了 10 GB 記憶體」**；正確說法是「預約了兩塊各 5.15 GB 的位址空間，實際用到約 4.8 GB」。在不允許 overcommit 的系統（或 Windows）上，這種寫法才會真的要求 10 GB。
- predict.py 峰值 RSS 2,117,040 KB ≈ 2.1 GB（測試特徵 646,268 × 429 × 4 = 1,108,995,888 bytes）。
- `LibriDataset.__init__` 的 `torch.LongTensor(y)`：y 已經是 int64 tensor 時**不複製**（與 y 共用記憶體，實測 data_ptr 相同）。
- 官方 sample 的 `max_len = 3000000` 同樣寫法（concat 1 時只有 39 欄，約 0.47 GB）。

## 形狀（ch03）
- 每個 epoch 的 batch 數（batch_size 64）：train 37,182（最後一個 batch 4 筆）、val 4,134（最後一個 58 筆）、test 10,098（最後一個 60 筆）。
- 一個 batch：features (64, 429) float32、labels (64, 11) int64；`features.view(-1, 11, 39)` → (64, 11, 39)。
- `view` 可以直接用，因為 429 欄的排列是「第 −5 格的 39 維、第 −4 格的 39 維、…」（frame-major），正好是 (11, 39) 的 row-major 排列。

## 模型（model.py；ch00 模型總覽、ch04）
- `print(model)`（train.py:39）逐字：
  ```
  Classifier(
    (lstm): LSTM(39, 512, num_layers=10, batch_first=True, dropout=0.5)
    (out): Linear(in_features=512, out_features=41, bias=True)
  )
  ```
- 參數：
  - 第 0 層：`weight_ih_l0` (2048, 39) 79,872；`weight_hh_l0` (2048, 512) 1,048,576；`bias_ih_l0`、`bias_hh_l0` 各 (2048,) 2,048。小計 **1,132,544**。2048 = 4 個 gate × 512。
  - 第 1–9 層每層：`weight_ih` (2048, 512) 1,048,576 + `weight_hh` (2048, 512) 1,048,576 + 兩個 bias 4,096 = **2,101,248**；9 層 18,911,232。
  - `out`：(41, 512) 20,992 + 41 = **21,033**。
  - 總計 **20,064,809**（手算 1,132,544 + 9 × 2,101,248 + 21,033 = 20,064,809）。checkpoint 80,264,975 bytes ≈ 20,064,809 × 4 bytes + key 名稱等額外資訊。
  - 公式：每層 4·h·(輸入維度 + h) + 2·4·h（PyTorch 的 LSTM 有 `bias_ih`、`bias_hh` 兩組 bias）。
- 形狀：輸入 (64, 11, 39) → `lstm_out` (64, 11, 512)、`h_n` (10, 64, 512)、`c_n` (10, 64, 512) → `out` (64, 11, 41)。`lstm_out[:, -1]` 等於 `h_n[-1]`（最後一層的最後時間點；model.py:25 的註解只在這個意義下對）。
- `Classifier(input_dim=429, hidden_layers=1, hidden_dim=512)`：`input_dim`、`hidden_layers` 傳進去**沒被用到**（model.py:9 寫死 `input_size=39`、model.py:11 寫死 `num_layers=10`）；只有 `hidden_dim` 有作用。
- **單向（因果）**：把 eval 模式下輸入的第 6–10 格換成別的亂數，中間第 5 格的輸出**完全不變**（`torch.equal` 為 True），最後一格的輸出會變。所以被評分的中間格只看得到第 0–5 格（前 5 格加自己）。
- `model_dnn.py`：官方 sample 的模型，沒有任何程式 import 它（train.py:2、predict.py:6 都是 `from model import *`）。官方 sample 設定（輸入 39、hidden 256、1 個 hidden layer）參數 **86,569**；若用本 repo 的 config（429、1、512）是 503,849。
- model.py 的註解是簡體中文、從 MNIST 的 RNN 範例抄來的：model.py:9「图片每行的数据像素点」、model.py:12 解釋 batch_first、model.py:20「h_n 是分线, h_c 是主线」、model.py:24-25「选取最后一个时间点的 r_out 输出」；model.py:27 留著原本的 `out = self.out(lstm_out[:, -1, :])  - original`，實際用的是 model.py:28 的全部位置。

## 原版 notebook vs 本 repo（index「2022 vs 現在」與 ch07 的素材）
- 一個 notebook 拆成 7 個檔：config.py、utils.py、data_loader.py、model.py、model_dnn.py、train.py、predict.py。Colab 專屬的 `!nvidia-smi`、`!wget`、`!unzip` 拿掉。
- 超參數（原版 → 本 repo）：`concat_nframes` 1 → **11**；`train_ratio` 0.8 → **0.9**；`batch_size` 512 → **64**；`num_epoch` 5 → **20**；`hidden_dim` 256 → **512**；`seed` 0、`learning_rate` 0.0001、`hidden_layers` 1 不變（但 hidden_layers 在本 repo 沒作用）。新增 `input_dim_lstm = 39`。
- 模型：原版 `BasicBlock`(Linear+ReLU) 疊成的 DNN（現在在 model_dnn.py，沒被用）→ **10 層單向 LSTM**，hidden 512、dropout 0.5，對 11 個位置各輸出 41 類。
- 標籤：原版每格 1 個標籤 `y` (N,) → 本 repo 拼成 (N, 11)（utils.py:73、83-84、89；原版那行留成 `- original` 註解）。
- 訓練：原版 `loss = criterion(outputs, labels)` → 本 repo 把 (64, 11, 41) 攤成 (704, 41) 對 704 個標籤算 loss（11 個位置一起算），準確率只取中間第 5 格（train.py:63-72、93-100，標 `# new`）。
- 預測：多了 `features.view(...)` 與取中間格（predict.py:39、43）。
- 原版的 `same_seeds` 也是在建 DataLoader 之後、建模型之前呼叫，本 repo 順序相同。
- 原版的 device 是 `'cuda:0' if torch.cuda.is_available() else 'cpu'`，本 repo 寫死 `"cuda"`（刻意的，見「環境」）。

## 投影片重點（hw2_slides 2022.pdf）
- p.4：資料前處理（從波形抽 MFCC）助教已經做好；學生做的是逐音框（framewise）音素分類。
- p.5：phoneme 定義與例子「Machine Learning → M AH SH IH N L ER N IH NG」，每個音素佔好幾格。
- p.6–7：39 維 MFCC（另提 80 維 filter bank）。
- p.8：每格只有 25 ms，一個音素通常橫跨好幾格 → 把相鄰的格拼起來；圖示 11 格 × 39 = 429 維、shape (1, 429)。「Finding testing labels or doing human labeling are strictly prohibited!」
- p.10：LibriSpeech train-clean-100 的子集；訓練 4268（應為 4286，見上）句 2,644,158 格；測試 1078 句 646,268 格；41 類。
- p.11–12：檔案結構；每個 .pt 是 (T, 39)；使用額外資料成績 × 0.9。
- p.14：Kaggle 4%、程式 2%、報告 4%。
- p.15 Kaggle public baselines（逐字）：Simple **0.45797**（sample code）；Medium **0.69747**（concat n frames, add layers）；Strong **0.75028**（concat n, batchnorm, dropout, add layers）；Boss **0.82324**（sequence-labeling(using RNN)）。
- p.16：評估指標 accuracy；截止 2022/3/18 23:59 (UTC+8)。p.18：每天最多 5 次上傳、選 2 個進 private leaderboard。
- p.20 報告題（逐字要點）：1. (2%) 參數量差不多的兩個模型，(A) 窄而深（例 hidden_layers=6, hidden_dim=1024）、(B) 寬而淺（例 hidden_layers=2, hidden_dim=1700），報告 training/validation accuracy。2. (2%) 加 dropout，報告 dropout rate (A) 0.25、(B) 0.5、(C) 0.75 的 training/validation accuracy。
- 本 repo **沒有** Kaggle 測試集的分數（沒有上傳紀錄）。教材只能說驗證集準確率落在哪兩條基準線之間，**不能宣稱通過了哪條 Kaggle 基準線**。

## 驗證指標檢查（Phase 0，2026-10-04；ch05）
- 結論：**沒有偏差**。2026-10-03 那次訓練（`HW02/model.ckpt`）印出的最佳 val acc 是 0.642；同一個 checkpoint 在整個驗證集上一次算完：中間格答對 **169,842 / 264,570 = 0.641955**。
- 原因（與 HW01 不同）：
  1. `val_loader` 是 `shuffle=False`（train.py:32），而且順序本來就不影響總和。
  2. `val_acc` 是逐 batch 累加**答對的個數**（train.py:104），最後除以 `len(val_set)`（train.py:108），不是各 batch 平均再平均；最後一個 58 筆的 batch 權重正確。
  3. 驗證時 `model.eval()`（dropout 關掉）＋ `torch.no_grad()`。
  4. 唯一的「挑選」是 20 個 epoch 裡挑 val acc 最高的存檔（train.py:112-115），驗證集同時被拿來選 epoch，所以 0.642 對沒看過的資料略為樂觀；但數字本身就是那個 checkpoint 在整個驗證集上的真實準確率。
- 印出的 val loss 是各 batch 平均再平均（train.py:105、108 除以 `len(val_loader)`），最後一個 batch 只有 58 筆卻佔一樣的權重；實測最佳 checkpoint：印出式（batch 平均再平均）1.299442（＝ train.py 第 15 epoch 印的值）、逐筆加權 1.299457，差 1.5e-5；最後一個 batch 58 筆、loss 0.6341。只算中間格的 val loss 是 1.197084（印出的 loss 是 11 個位置的平均，比中間格高，因為前面的位置看到的過去比較少）。不影響 acc。
- train acc（train.py:76、108）是在 `model.train()`（dropout 開著）下、邊更新邊量的，所以和 val acc 不能直接比。
- 同一個 checkpoint 各位置（第 0–10 格，第 5 格是中間）的準確率：
  `[0.4555, 0.525, 0.5692, 0.6013, 0.6244, 0.642, 0.6556, 0.6659, 0.6728, 0.6783, 0.6827]`
  單調上升：越後面的位置看得到越多過去的格子。第 10 格的 0.6827 比中間格高，但它預測的是**另一格**（t+5）的標籤，不能拿來當作中間格的答案。

## 執行實測（2026-10-04，在 HW02 的複本裡跑，libriphone 用 symlink）
- `../.venv/bin/python train.py`（`/usr/bin/time -v` 包著）：wall clock **3:04:48**、user 8,852 s、sys 2,099 s、峰值 RSS 6,029,716 KB、exit 0。GPU 同時有別的工作（前半段有 2 個 hw02_exp.py 與另一個 session），所以 3 小時只是上限的量級；乾淨的單一 epoch 時間待補（使用者選 (a)：實驗全部跑完、GPU 沒別的工作時單跑 1 epoch）。<!-- TODO(本機實測): 無干擾的 1 epoch 時間 -->
- 每個 epoch 訓練迴圈 37,182 步：有干擾時 10–14 分鐘，GPU 只剩它（與另一個 session）時約 8 分 20 秒（約 74 it/s）；驗證 4,134 步約 10–20 秒。
- stdout 逐字（tqdm 進度條在 stderr，這裡略去）：
  ```
  DEVICE: cuda
  [Dataset] - # phone classes: 41, number of utterances for train: 3857
  [INFO] train set
  torch.Size([2379588, 429])
  torch.Size([2379588, 11])
  [Dataset] - # phone classes: 41, number of utterances for val: 429
  [INFO] val set
  torch.Size([264570, 429])
  torch.Size([264570, 11])
  Classifier(
    (lstm): LSTM(39, 512, num_layers=10, batch_first=True, dropout=0.5)
    (out): Linear(in_features=512, out_features=41, bias=True)
  )
  [001/020] Train Acc: 0.495127 Loss: 1.802524 | Val Acc: 0.576195 loss: 1.500938
  saving model with acc 0.576
  [002/020] Train Acc: 0.577339 Loss: 1.492246 | Val Acc: 0.599297 loss: 1.409000
  saving model with acc 0.599
  [003/020] Train Acc: 0.597845 Loss: 1.419920 | Val Acc: 0.610678 loss: 1.376042
  saving model with acc 0.611
  [004/020] Train Acc: 0.611049 Loss: 1.373563 | Val Acc: 0.621337 loss: 1.340509
  saving model with acc 0.621
  [005/020] Train Acc: 0.622775 Loss: 1.331375 | Val Acc: 0.627588 loss: 1.316813
  saving model with acc 0.628
  [006/020] Train Acc: 0.631520 Loss: 1.300601 | Val Acc: 0.632608 loss: 1.301389
  saving model with acc 0.633
  [007/020] Train Acc: 0.638695 Loss: 1.275846 | Val Acc: 0.635012 loss: 1.293533
  saving model with acc 0.635
  [008/020] Train Acc: 0.644886 Loss: 1.255450 | Val Acc: 0.635745 loss: 1.294502
  saving model with acc 0.636
  [009/020] Train Acc: 0.649985 Loss: 1.236995 | Val Acc: 0.638398 loss: 1.290916
  saving model with acc 0.638
  [010/020] Train Acc: 0.655303 Loss: 1.220799 | Val Acc: 0.636406 loss: 1.299407
  [011/020] Train Acc: 0.659873 Loss: 1.205578 | Val Acc: 0.640451 loss: 1.291020
  saving model with acc 0.640
  [012/020] Train Acc: 0.664004 Loss: 1.191596 | Val Acc: 0.639993 loss: 1.296353
  [013/020] Train Acc: 0.668320 Loss: 1.178405 | Val Acc: 0.640420 loss: 1.296682
  [014/020] Train Acc: 0.671785 Loss: 1.165417 | Val Acc: 0.640088 loss: 1.300400
  [015/020] Train Acc: 0.675382 Loss: 1.153714 | Val Acc: 0.641955 loss: 1.299442
  saving model with acc 0.642
  [016/020] Train Acc: 0.678926 Loss: 1.142452 | Val Acc: 0.641305 loss: 1.304638
  [017/020] Train Acc: 0.682067 Loss: 1.131953 | Val Acc: 0.641006 loss: 1.303436
  [018/020] Train Acc: 0.685094 Loss: 1.121886 | Val Acc: 0.641343 loss: 1.308557
  [019/020] Train Acc: 0.688006 Loss: 1.112876 | Val Acc: 0.639249 loss: 1.318320
  [020/020] Train Acc: 0.690659 Loss: 1.104173 | Val Acc: 0.641936 loss: 1.316187
  ```
- 解讀要點：最佳是**第 15 個 epoch**（0.641955），共存檔 11 次；val loss 最低是第 11 個 epoch 的 1.291020，之後 val loss 緩升、train acc 持續上升（0.675 → 0.691）→ 輕微過擬合。val acc 從第 11 epoch 起停在 0.639–0.642。
- **可重現**：這次產生的 `model.ckpt` 與 2026-10-03 那次的 `HW02/model.ckpt` **逐位元組相同**（`cmp` 無差異），所以 `HW02/model.ckpt` 就是第 15 epoch 的權重，上面「驗證指標檢查」的數字都適用。
- 印出的 `saving model with acc` 只有 3 位小數（train.py:115），epoch 行是 6 位。
- `../.venv/bin/python predict.py`（複本裡，同一個 checkpoint）：wall clock **27.05 s**、峰值 RSS 2.1 GB。stdout：
  ```
  DEVICE: cuda
  [Dataset] - # phone classes: 41, number of utterances for test: 1078
  [INFO] test set
  torch.Size([646268, 429])
  ```
  產生的 `prediction.csv` 與 repo 裡被追蹤的 `HW02/prediction.csv` **逐位元組相同**（646,269 行，開頭 `Id,Class`、`0,0`、`1,0`）。預測裡 41 類都有出現；最多的是第 0 類 126,222 格（0.1953）、第 31 類 40,981（0.0634）、第 2 類 37,541（0.0581）。

## 實驗工具（docs/tools/hw02_exp.py、hw02_run_grid.sh）
- `hw02_exp.py` 預設參數就是 train.py；亂數順序照 train.py：preprocess（python random，seed 1337）→ 建 DataLoader → `same_seeds(0)` → 建模型 → AdamW。
- 2026-10-04 驗證：`--epochs 1` 印出 `[001/001] Train Acc: 0.495127 Loss: 1.802524 | Val Acc: 0.576195 loss: 1.500938`，與 train.py 第 1 個 epoch 的四個數字逐位相同。因為 batch 64 時亂數流與 train.py 相同，`--epochs 5` 的結果就是 20 epoch baseline 的前 5 個 epoch。
- `hw02_run_grid.sh <runs.txt> <out.jsonl> [parallel]`：ch07 的實驗清單是 `docs/tools/hw02_ch07_runs.txt`。
- 時間：本機 GPU 同時有別的 session 的工作，所以每個 epoch 的秒數只是「量級」，不是乾淨的基準（例：驗證用的 1 epoch run 花了 942 s，同時有 baseline 與另一個 session 在跑）。

## ch07 實測（2026-10-04，本機；hw02_exp.py，在 HW02/ 裡執行）
- 縮小規格（使用者選）：batch 64、**5 個 epoch**、lr 1e-4、AdamW、seed 0、train_ratio 0.9、concat 11，除非下表另註。官方 sample 兩列照 notebook 用 batch 512。
- 「最佳 val」是 5 個 epoch 裡最高的那個（照 train.py 的存檔規則）；每一組都另外用最佳權重在整個驗證集上一次算完，**全部與印出值相同**。
- 原始 JSON：`docs/tools/hw02_ch07_runs.jsonl`（含每個 epoch 的 train/val acc 與 loss）。重跑：`docs/tools/hw02_run_grid.sh docs/tools/hw02_ch07_runs.txt <out.jsonl> 1`。
- 本 repo 的 10 層 LSTM 在同一規格下 = 原規格 baseline 的第 1–5 epoch（同一條亂數流）：最佳第 5 epoch **0.627588**，train acc 0.622775，val loss 1.316813。
- 參數量都是 PyTorch 數的。秒數是整個 run（含載入資料約 1 分鐘）；dnn_c11、q1a_deep 與 baseline 同時跑，秒數偏大。

| run | 設定 | 參數量 | 最佳 epoch | 最佳 val | 第 5 epoch train acc | 第 5 epoch val | 第 5 epoch val loss | 秒 |
|---|---|---|---|---|---|---|---|---|
| sample_official | 官方 sample（DNN，concat 1，hidden 256，1 層，batch 512，ratio 0.8） | 86,569 | 5 | 0.457758 | 0.4608 | 0.4578 | 1.8898 | 143 |
| sample_r09 | 官方 sample，但 ratio 0.9（本 repo 的驗證集） | 86,569 | 5 | 0.458215 | 0.4611 | 0.4582 | 1.8867 | 144 |
| dnn_c11 | model_dnn.py + 本 repo config（concat 11，hidden 512，1 層） | 503,849 | 5 | 0.671459 | 0.6938 | 0.6715 | 1.0513 | 1279 |
| q1a_deep | 報告題 1 (A) 窄深：DNN 6 層 × 1024 | 6,779,945 | 5 | 0.687270 | 0.7683 | 0.6873 | 1.0710 | 1551 |
| q1b_wide | 報告題 1 (B) 寬淺：DNN 2 層 × 1700 | 6,584,141 | 3 | 0.687470 | 0.8128 | 0.6787 | 1.1559 | 317 |
| q2_d25 | 報告題 2 (A)：6×1024 + dropout 0.25 | 6,779,945 | 5 | 0.690010 | 0.6710 | 0.6900 | 0.9815 | 440 |
| q2_d50 | 報告題 2 (B)：6×1024 + dropout 0.5 | 6,779,945 | 5 | 0.650421 | 0.5987 | 0.6504 | 1.1445 | 415 |
| q2_d75 | 報告題 2 (C)：6×1024 + dropout 0.75 | 6,779,945 | 5 | 0.512235 | 0.4577 | 0.5122 | 1.8975 | 420 |
| strong_bn_d25 | 6×1024 + BatchNorm + dropout 0.25（strong baseline 的提示） | 6,794,281 | 5 | 0.690154 | 0.6560 | 0.6902 | 0.9748 | 537 |

- 每個 epoch 的 val acc：
  - sample_official：0.4406、0.4496、0.4538、0.4561、0.4578
  - sample_r09：0.4437、0.4517、0.4552、0.4573、0.4582
  - dnn_c11：0.6317、0.6517、0.6612、0.6680、0.6715
  - q1a_deep：0.6554、0.6775、0.6836、0.6869、0.6873
  - q1b_wide：0.6677、0.6859、0.6875、0.6835、0.6787
  - q2_d25：0.6387、0.6633、0.6754、0.6846、0.6900
  - q2_d50：0.5996、0.6230、0.6368、0.6462、0.6504
  - q2_d75：0.4218、0.4626、0.4828、0.5029、0.5122
  - strong_bn_d25：0.6428、0.6611、0.6735、0.6835、0.6902
- 解讀（寫章時可用，數字都在上表）：
  - 官方 sample 0.457758 ≈ 投影片 simple baseline 0.45797（驗證集不是 Kaggle 測試集，只能說「相當」）。ratio 0.8 與 0.9 只差 0.0005。
  - 只把 concat 1 改成 11（dnn_c11，同樣 1 個 hidden layer）：0.4582 → 0.6715，是所有改動裡最大的一步。
  - 報告題 1：A 窄深 0.687270（6,779,945 參數）與 B 寬淺 0.687470（6,584,141 參數）幾乎相同；差別在過擬合：B 第 3 epoch 後 val 下降、第 5 epoch train 0.8128 vs val 0.6787；A 第 5 epoch train 0.7683 vs val 0.6873 還在進步。
  - 報告題 2：dropout 0.25 → 0.690010（比無 dropout 的 0.687270 好），0.5 → 0.650421，0.75 → 0.512235。5 個 epoch 內 dropout 越大學得越慢；0.5、0.75 的 val loss 到第 5 epoch 都還在降，是**欠擬合**不是 dropout 無效。train acc 在 model.train() 下量（dropout 開著），所以 dropout 越大 train acc 越低，甚至低於 val acc。
  - BN + dropout 0.25：0.690154，與只加 dropout 0.25 幾乎相同（+0.00014），val loss 最低（0.9748）。
  - 5 個 epoch 下，上面所有 6×1024 的 DNN（約 680 萬參數）都**勝過**本 repo 的 10 層 LSTM（2,006 萬參數，0.627588）；LSTM 20 個 epoch 也只到 0.641955。沒有一組到 medium baseline 0.69747（驗證集）。
- **雙向 LSTM（bilstm，2026-10-04 21:50–23:03，GPU 上只有它）**：與 model.py 相同（10 層、hidden 512、dropout 0.5、loss 算 11 格），只加 `bidirectional=True`，`out` 改成 Linear(1024, 41)。參數 **59,003,945**（單向的 2.94 倍）。每個 epoch 約 864–891 秒，整個 run 4376 秒。
  - 每個 epoch：1: train 0.661983 / val 0.717266 / val loss 0.995585；2: train 0.744212 / val 0.738704 / val loss 0.922908；3: train 0.777281 / val 0.745209 / val loss 0.911529；4: train 0.804001 / val 0.744570 / val loss 0.938520；5: train 0.826498 / val 0.743554 / val loss 0.966469
  - 最佳第 3 epoch **0.745209**（整個驗證集一次算完相同）。第 1 個 epoch 的 0.717266 就勝過所有其他實驗；第 2 epoch 起超過 medium baseline 0.69747，最佳離 strong baseline 0.75028 差 0.0051（驗證集，不是 Kaggle）。第 3 epoch 後 val loss 上升（0.9115 → 0.9665）、train acc 0.8265 → 輕微過擬合。
  - 各位置準確率（最佳權重）：`[0.6648, 0.6938, 0.7155, 0.7311, 0.7401, 0.7452, 0.7457, 0.7421, 0.7315, 0.7155, 0.6931]`。**以中間為峰、左右大致對稱**（第 5 格 0.7452、第 6 格 0.7457 最高，兩端 0.66–0.69），對照單向版從 0.4555 單調升到 0.6827：兩端的格子分別缺少過去或未來的上下文，中間格兩邊各有 5 格。
- <!-- 尚未跑（等使用者放行，每組約 45 分鐘）：lstm_3layers、lstm_lossmid；以及無干擾的 train.py 1 epoch 計時 -->

## repo 問題清單（教材附錄素材；本 repo 照原樣保留，未修）
1. config.py:17 `hidden_layers = 1` 沒作用，model.py:11 寫死 10 層；`input_dim` 也沒作用（model.py:9 寫死 39）。
2. model.py 註解從 MNIST 範例抄來，與實際用途不符（見「模型」）。
3. 單向 LSTM：被評分的中間格看不到未來 5 格。
4. model_dnn.py 沒被 import（死碼）。
5. utils.py:69-70 預先配 3,000,000 × 429 float32（5.15 GB 位址空間），切片後 storage 仍是整塊；Linux 上沒寫入的頁不佔實體記憶體（峰值 RSS 6.0 GB），但寫法本身浪費且依賴 overcommit。
6. 投影片 p.10「4268」句，實際 4286 句。
7. predict.py:29-30 `test_acc`、`test_lengths` 宣告了沒用。
8. prediction.csv 被 git 追蹤（在 .gitignore 加 `*.csv` 之前就加入了），重跑 predict.py 會改到它。
9. 印出的 val loss 是 batch 平均再平均（最後一個 batch 58 筆），只影響 loss，不影響 acc。

## 已在前面章節定義過的名詞
（寫章的 session 每章追加；後續章節不必重講，可簡短回指。）
