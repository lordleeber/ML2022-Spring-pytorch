# HW03 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW03/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **狀態（2026-10-05）**：全書完成（index、outline、ch00–ch08、appendix），已推上 master。這份檔案保留當維護筆記：修訂章節、重跑實驗時先讀這裡，新的實測照「chNN 實測」的格式追加，舊數字被推翻時回頭修正對應章節並在這裡註明。
>
> **寫作方式**：本機 session 一章一章寫（使用者 2026-10-05 決定），事實一次量完（Phase 1）。這個 session 停用冷讀（使用者 2026-10-05），所以各章沒有冷讀；要打磨時再逐章跑 cold-read。
>
> **重現方法**（需要 GPU 與 `HW03/food11/`）：在 `HW03/` 裡跑
> - `PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw03_exp.py --name X [選項]`：訓練實驗，預設參數就是 train.py，不寫檔（除非給 `--save`／`--dump`）。
> - `bash docs/tools/hw03_run_grid.sh <ckpt_dir>`（在 repo 根目錄）：ch08 的全部實驗，一次一組依序跑。

教材對應 commit：`42ffd90`（HW03 程式最後一次改動是 `381aad0`，2026-10-03）。
原始碼根目錄：`HW03/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW03/HW03.ipynb`（Colab notebook，不在 repo 裡，下面「原版 vs 本 repo」列了差異）。作業投影片：`HW03/Machine Learning HW3 - Image Classification.pdf`（49 頁，已在 repo）。

## 全書約定
- 樣式：mySkills **新版共用資產**（`completed-repo-to-html-textbook/assets/style.css`、`enhance.js`；與 docs/HW02、HW09 相同，使用者 2026-10-05 選的）。
- 指令塊：讀者要貼上執行的用 `<pre class="shell cmd">`，輸出用 `<pre class="shell">`。Python 用 `<pre class="py">`。
- listing 的 `data-hot` 寫**原始碼行號**。
- 檢查：`python3 docs/tools/verify_book.py . docs/HW03/chNN.html`。
- 每本書的規則（使用者指定）：ch00 要有「模型總覽」（架構 SVG、每層 tensor 形狀、參數量：手算 + PyTorch 印出、哪個檔案定義模型），放在任務說明之後；目錄頁有「2022 vs 現在」導論；2022 寫法過時的地方加「現在的做法」框。
- 大綱：`docs/HW03/outline.html`（使用者 2026-10-05 核可：ch00–ch08 + appendix）。
- ch08 實驗規格（使用者 2026-10-05 核可）：所有對照組 **40 epoch** 並排比，另附一次 **200 epoch** 長跑；GPU 一次只跑一組，使用者授權依序自己排、不用每組等放行；每 30 分鐘回報進度。
- 圖片：ch01／ch06 放少量小縮圖（每類約 1 張、約 96px）在 `docs/HW03/img/`（使用者 2026-10-05 選的）。

## 環境（實測 2026-10-05）
- Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、Pillow 12.3.0、numpy 2.5.3、pandas 3.0.6、tqdm 4.70.1、cuDNN 91900。
- GPU：NVIDIA RTX PRO 4000 Blackwell（24,467 MiB），WSL2；24 核心；RAM 47 GB。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW03/` 裡用 `../.venv/bin/python train.py`。`train.py:68`、`test.py:22` 寫死 `device = "cuda"`，**這是刻意的設計**，教材以中性描述，不列為問題或練習。

## 資料與檔案
- 來源：本機 Kaggle zip `ml2022spring-hw3b.zip`（1,163,226,202 bytes，Windows Documents/poyi/ml_2022_data）。解到 `HW03/food11/`，約 1.2 GB。官方 notebook 用 `wget` 從 Dropbox 下載 `food11.zip`。
- `HW03/.gitignore`：`food11/*`、`sample_*.txt`、`*.csv`、`*.ckpt`。`sample_best.ckpt`（51,360,932 bytes）、`submission.csv`（23,635 bytes）、`sample_log.txt`（0 bytes）都沒被追蹤。
- `food11/training` 9866 張、`food11/validation` 3430 張、`food11/test` 3347 張，全部 `.jpg`；與投影片 p.3 一致。
- 檔名：train/val 是 `類別_序號.jpg`（例 `0_0.jpg`、`0_1.jpg`、`0_10.jpg`，排序是字串序）；test 是 `0001.jpg`–`3347.jpg`（補零四位，所以字串序 = 數字序）。
- 每類張數（training / validation）：0: 994/362、1: 429/144、2: 1500/500、3: 986/327、4: 848/326、5: 1325/449、6: 440/147、7: 280/96、8: 855/347、9: 1500/500、10: 709/232。
- 圖片（`hw03_facts.py data`，全量；PIL 的 size 是 (寬, 高)；全部是 RGB）：
  - training 9866 張、673,952,072 bytes、542 種尺寸、正方形 5567 張；最常見 (512,512) 5551、(512,384) 1355、(384,512) 519、(382,512) 320、(512,382) 216、(512,341) 142。
  - training 寬 min/中位數/max 220/512/7360、高 207/512/4912。
  - validation 3430 張、265,247,465 bytes、251 種尺寸、正方形 2125；(512,512) 2111、(512,384) 436、(384,512) 191；寬 287/512/9216、高 227/512/6144。
  - test 3347 張、249,243,175 bytes、244 種尺寸、正方形 1991；(512,512) 1978、(512,384) 427、(384,512) 192；寬 288/512/9542、高 257/512/5126。
- 標籤解析實測：`./food11/training/3_120.jpg` → 3；`./food11/test/0001.jpg` → `ValueError: invalid literal for int() with base 10: '0001.jpg'` → `except` 給 -1。Windows 路徑 `food11\training\3_120.jpg` 用原版 `split("/")[-1].split("_")[0]` 得到 `'food11\\training\\3'`，`int()` 失敗 → 每張訓練圖都會變成 -1（commit `381aad0` 修的就是這個情況）。
- 類別名稱：**repo 與投影片都沒寫**；來自 Food-11 原始資料集（EPFL MMSPG）：0 Bread、1 Dairy product、2 Dessert、3 Egg、4 Fried food、5 Meat、6 Noodles/Pasta、7 Rice、8 Seafood、9 Soup、10 Vegetable/Fruit。縮圖目視一致（0_0 麵包、1_0 起司、2_0 蛋糕、3_0 蛋、4_0 洋蔥圈、5_0 肉、6_0 義大利麵、7_0 炒飯、8_0 生蠔、9_0 湯、10_0 蔬菜）。教材要註明名稱出自原始資料集。
- 縮圖（`hw03_facts.py thumbs`）：`docs/HW03/img/class00.jpg`–`class10.jpg`，各取 training 的 `k_0.jpg`，`thumbnail((96, 96))`、JPEG quality 80，共 33,283 bytes。原圖尺寸：0_0 (512,512)、1_0 (512,384)、2_0 (512,512)、3_0 (512,384)、4_0 (512,512)、5_0 (512,384)、6_0 (512,512)、7_0 (380,273)、8_0 (512,512)、9_0 (306,512)、10_0 (1024,789)。

## 增強示範（`hw03_facts.py aug`；ch02、ch06）
- 原圖 `training/0_0.jpg`（512×512）。`test_tfm` 後 tensor (3, 128, 128) float32，min 0.003921568859368563（=1/255）、max 0.9960784316062927（=254/255）。左上角像素：`im.resize((128,128)).getpixel((0,0))` = (189, 210, 253) → tensor `[0.7411764860153198, 0.8235294222831726, 0.9882352948188782]`（= 189/255 等）。
- `torch.manual_seed(0)` 後對同一張圖跑增強 A 5 次：5 個結果兩兩不同（distinct 5）→ 符合 Q1「5 種以上」。圖檔 `docs/HW03/img/aug_resize.jpg`（test_tfm 的結果）、`aug_1.jpg`–`aug_5.jpg`，共 34,993 bytes。

## 模型（classifier.py；ch00 模型總覽、ch03）
- 每層（`hw03_facts.py shapes`，輸入 (1, 3, 128, 128)）：
  - cnn[0] Conv2d → (64,128,128) 1,792；[1] BN 128；[3] MaxPool → (64,64,64)
  - cnn[4] Conv2d → (128,64,64) 73,856；[5] BN 256；[7] → (128,32,32)
  - cnn[8] Conv2d → (256,32,32) 295,168；[9] BN 512；[11] → (256,16,16)
  - cnn[12] Conv2d → (512,16,16) 1,180,160；[13] BN 1,024；[15] → (512,8,8)
  - cnn[16] Conv2d → (512,8,8) 2,359,808；[17] BN 1,024；[19] → (512,4,4)
  - view → (8192)；fc[0] Linear → 1024 8,389,632；fc[2] → 512 524,800；fc[4] → 11 5,643
  - 卷積部分合計 3,913,728（含 BN 2,944）；全連接合計 8,920,075。total 12,833,803、全部 trainable；buffers 2,949（5 個 BN 的 running_mean + running_var 共 2×1,472 = 2,944，加 5 個 num_batches_tracked）。
- 感受野（3×3 conv、2×2 pool）：3 → 4 → 8 → 10 → 18 → 22 → 38 → 46 → 78 → 94；最後一層 4×4 的每一格看到原圖 94×94 的範圍，jump 32。
- `print(model)` 的完整輸出見 hw03_facts.py shapes（Conv2d 印成 `kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)`，BN 印 `eps=1e-05, momentum=0.1, affine=True, track_running_stats=True`，MaxPool 印 `kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False`）。
- `Classifier`：12,833,803 個參數；輸入 (B, 3, 128, 128) → 輸出 (B, 11)。
- 第一層全連接 `Linear(8192, 1024)`：8192×1024 + 1024 = 8,389,632（占 65.4%）。
- `others.py` 的 `Residual_Network`：68,259,147 個參數；`Linear(256*32*32, 256)` = 262,144×256 + 256 = 67,109,120；fc_layer 合計 67,111,947（加 `Linear(256, 11)` 2,827）。輸入 (B, 3, 128, 128) → (B, 11)。
  - 每層輸出與參數：layer1 (64,128,128) 1,920；layer2 (64,128,128) 37,056；layer3 (128,64,64) 74,112（stride 2）；layer4 (128,64,64) 147,840；layer5 (256,32,32) 295,680（stride 2）；layer6 (256,32,32) 590,592。
  - 沒有池化，只靠兩次 stride 2 降到 32×32，所以攤平後有 262,144 維。
  - `exec(others.py)` 整份：`NameError: name 'transforms' is not defined`。
- batch 數（`hw03_facts.py loader`）：train 9866/256 → 39 批、最後 138；val 3430/256 → 14 批、最後 102；test 3347/256 → 14 批、最後 19；原版 batch 64：train 155 批、最後 10。

## 原版 notebook vs 本 repo
- 原版是一個 notebook；本 repo 拆成 config.py／dataset.py／classifier.py／train.py／test.py／others.py。
- `batch_size` 64 → 256（`config.py:2`，另有註解 `# batch_size = 512  # for RTX3090`）。
- `n_epochs` 3 → 5（`train.py:71`）。
- 標籤解析：原版 `int(fname.split("/")[-1].split("_")[0])`；本 repo `int(os.path.basename(fname).split("_")[0])`（commit `381aad0`，原本是 Windows 的 `"\\"`）。
- `FoodDataset.__init__` 原版 `tfm=test_tfm` 有預設值、`if files != None`；本 repo `tfm` 無預設、`if files is not None`。
- `device`：原版 `"cuda" if torch.cuda.is_available() else "cpu"`；本 repo 寫死 `"cuda"`。
- 原版 `_exp_name`、`_dataset_dir`；本 repo `exp_name`、`dataset_dir`（在 config.py）。
- 本 repo train.py 多了 `labels_on_device`、`compare_result = torch.eq(...)`（原版 `(logits.argmax(dim=-1) == labels.to(device)).float().mean()`，原版那行留成註解）。
- `Classifier.forward`：本 repo 用 `zzz`、`vvv` 中間變數；原版一行 `out.view(out.size()[0], -1)`。
- `others.py` = 原版 notebook 最後兩格（Q1 `train_tfm`、Q2 `Residual_Network`），逐字相同（只差空白）。

## 投影片重點
- p.2 目標：CNN、資料增強、residual 等影像模型技巧。p.3 資料數量。p.4 規則：不准找測試集原始標籤、不准外部資料、**不准預訓練模型**。
- p.5 基準線：Simple 0.50099、Medium 0.73207（Training Augmentation + Train Longer）、Strong 0.81872（+ Model Design + Train Looonger (+ Cross Validation + Ensemble)）、Boss 0.88446（+ Test Time Augmentation）。
- p.6 Kaggle（P100）時間：Simple 15~20 分；Medium Augmentation A 6 小時／B 80 分；Strong 6–12 小時；Boss 12 小時。
- p.7 提交格式：`Id,Category`，Id 對應 test 的 jpg 檔名。p.8 可用 torchvision.models／timm，但 `pretrained=False`。
- p.9 增強；p.10–11 mixup（要自己寫 cross entropy，因為標籤是機率向量）；p.12–13 TTA（例：`avg_train_tfm_pred * 0.5 + test_tfm_pred * 0.5`）；p.14–15 cross validation、重切 train:valid（目前約 3:1）；p.16 ensemble（平均 logits／機率，或投票）。
- p.33–34 技巧：增強一定要做；有了增強可以放大模型；有 downsampling 的結構比較好；label smoothing、focal loss、dropout、gradient accumulation、BatchNorm、image normalization。
- p.37 Q1：train_tfm 對同一張圖要能產生 5 種以上不同結果（2%）。p.38–39 Q2：只改 `forward`，照圖接 residual（2%）。圖：layer1→ReLU→layer2→(+x1)→ReLU→layer3→ReLU→layer4→(+x3)→ReLU→layer5→ReLU→layer6→(+x5)→ReLU→fc。
- p.42 計分：四條基準線 public/private 各 0.5、code 2、report 4，共 10 分。p.45 截止 2022/03/25。

## 驗證指標檢查（Phase 0，2026-10-05；ch04）
- `valid_loader` 是 `shuffle=True`（`train.py:64`），valid acc／loss 是各 batch 平均再平均（`train.py:172-173`）；3430 = 13×256 + 102，最後一批 102 張。
- 用 2026-10-03 的 `sample_best.ckpt`（與 2026-10-05 重跑的逐位元組相同）：整個驗證集一次算 acc 0.55394（1900/3430）<!-- TODO: 確認 1900 -->、loss 1.32449；照 train.py 逐批平均、8 種打亂順序：acc 0.55246–0.55625、loss 1.31778–1.32748。訓練時印出 0.55541。
- 結論：偏差很小（±0.2 個百分點），ch00 起直接用印出的數字並註明真實值。

## 執行實測（2026-10-05，在 scratchpad 的 HW03 複本裡跑，food11 用 symlink）
- 開跑前後 `nvidia-smi --query-compute-apps` 都沒有其他程式。
- `train.py`（5 epoch）：wall 3:00.70、user 3101.94 s、sys 12.85 s、CPU 1723%、max RSS 2,121,968 KB。`test.py`：wall 0:14.92、CPU 1366%、max RSS 1,557,480 KB。
- stdout（tqdm 進度條另外印在 stderr；每個 Valid 行印兩次，第二次是 log 區塊的 print）：
```
One ./food11/training sample ./food11/training/0_0.jpg
One ./food11/validation sample ./food11/validation/0_0.jpg
[ Train | 001/005 ] loss = 1.95937, acc = 0.31425
[ Valid | 001/005 ] loss = 3.20382, acc = 0.17568
[ Valid | 001/005 ] loss = 3.20382, acc = 0.17568 -> best
Best model found at epoch 0, saving model
[ Train | 002/005 ] loss = 1.58656, acc = 0.45393
[ Valid | 002/005 ] loss = 1.62730, acc = 0.43092
[ Valid | 002/005 ] loss = 1.62730, acc = 0.43092 -> best
Best model found at epoch 1, saving model
[ Train | 003/005 ] loss = 1.37545, acc = 0.52492
[ Valid | 003/005 ] loss = 1.55324, acc = 0.47906
[ Valid | 003/005 ] loss = 1.55324, acc = 0.47906 -> best
Best model found at epoch 2, saving model
[ Train | 004/005 ] loss = 1.18828, acc = 0.58642
[ Valid | 004/005 ] loss = 1.58591, acc = 0.48411
[ Valid | 004/005 ] loss = 1.58591, acc = 0.48411 -> best
Best model found at epoch 3, saving model
[ Train | 005/005 ] loss = 1.07386, acc = 0.62643
[ Valid | 005/005 ] loss = 1.31879, acc = 0.55541
[ Valid | 005/005 ] loss = 1.31879, acc = 0.55541 -> best
Best model found at epoch 4, saving model
```
- 5 個 epoch 每次都進步，所以每個 epoch 都印「-> best」並存檔。
- 可重現：2026-10-05 重跑的 `submission.csv` 與 2026-10-03 那次逐位元組相同（`cmp`）。`submission.csv` 開頭 `Id,Category` / `0001,9` / `0002,9`。
- `sample_log.txt` 0 bytes（log bug 實證）。

## 實驗工具（docs/tools/hw03_exp.py、hw03_run_grid.sh）
- **tqdm 會改變打亂順序**（2026-10-05 發現）：`from tqdm.auto import tqdm` 在一般 script 裡是 `tqdm.asyncio.tqdm_asyncio`（`tqdm/auto.py:30`），它的 `__init__`（`tqdm/asyncio.py:37`）會先 `iter(iterable)` 一次；之後 `for` 又呼叫一次 `__iter__`。所以每個 epoch 的每個 loop 都多建一個用不到的 DataLoader iterator，`_BaseDataLoaderIter.__init__`（torch `dataloader.py:708`）多抽一次 `random_()`。實驗工具第一版沒包 tqdm，第一個 batch 的標籤就不同（train.py `[3, 9, 2, 7, 8, 3, 1, 2]`、工具 `[8, 9, 9, 0, 9, 7, 10, 3]`），epoch 1 train loss 1.97084 vs 1.95937；包上 `tqdm(..., disable=True)`（一樣會走到那個 `iter`）後一致。小實驗：`DataLoader(list(range(10)), batch_size=10, shuffle=True)`、`torch.manual_seed(1)` 後直接取第一批 `[4, 2, 0, 6, 8, 7, 9, 1, 5, 3]`，經 tqdm 取 `[9, 7, 8, 3, 5, 1, 2, 0, 4, 6]`。
- 建立 `transforms.Compose([...隨機轉換...])` 本身不消耗亂數（實測）。
- 增強 A（`--aug A`）：`RandomResizedCrop((128, 128), scale=(0.5, 1.0))`、`RandomHorizontalFlip()`、`RandomRotation(15)`、`ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)`、`ToTensor()`。
- 殘差（`--arch res`）：照 p.39，`x2 = relu(layer2(x1) + x1)`、`x4 = relu(layer4(x3) + x3)`、`x6 = relu(layer6(x5) + x5)`。
- 重切（`--resplit 1`）：驗證集 3430 張用 `random.Random(0).shuffle` 打亂索引，前 1143 張（1/3）當 holdout，其餘 2287 張併入訓練（共 12153 張）。
- **逐位元驗證（2026-10-05）**：`hw03_exp.py` 預設參數（5 epoch）印出的 15 行 Train/Valid/Best 與 train.py 的 stdout `diff` 完全相同；存下的 state_dict 與 train.py 的 `sample_best.ckpt` 41 個 tensor 全部 `torch.equal`。最佳（第 5 epoch）在整個驗證集一次算：acc 0.55394、CE 1.32449（與 Phase 0 用 sample_best.ckpt 算的一致）。
- 工具多做的事（不消耗亂數）：每個 epoch 另算整個驗證集的真實 acc（答對數 / 3430）與 CE（sum / 3430）；訓練結束後用最佳權重以 `shuffle=False` 算整個驗證集與測試集的 logits，存成 `.npz`（`--dump`）。
- `--resplit 1` 時 `full_val_*` 不是驗證數字（2/3 的驗證集被拿去訓練），比較要用 holdout 1143 張。

## ch00 實測（2026-10-05，本機；在 HW03/ 裡）
- 參數計數指令（ch00 0.2 節逐字）：`named_modules()` + `parameters(recurse=False)`，13 行 + `total 12833803`；輸出已逐字放進 ch00。
- `test.py` stdout 只有一行 `One ./food11/test sample ./food11/test/0001.jpg`。`submission.csv` 3,348 行（標題 + 3,347），開頭 `0001,9`、`0002,9`，最後 `3347,1`。預測類別分布（0–10）：386、120、182、439、351、467、166、49、469、524、194。
- 在 repo 根目錄執行 `.venv/bin/python HW03/train.py`：`FileNotFoundError: [Errno 2] No such file or directory: './food11/training'`，出自 `dataset.py:17` 的 `os.listdir(path)`（還沒碰到 GPU）。
- 全部猜最多的一類（第 2 或第 9 類，驗證集各 500 張）：500/3430 = 0.14577。
- checkpoint 51,360,932 bytes vs 參數 12,833,803 × 4 = 51,335,212 bytes。
- tqdm 進度條沒有被 `script -qc` 錄到（只剩空行），所以教材不逐字引用進度條；每個 epoch 訓練 39 批、驗證 14 批（由 batch 數推得）。
- `train.py` 跑完 stderr 沒有任何警告（`grep -ic warn` = 0）。
- 名詞首見（ch00 已定義）：影像分類、CNN、資料增強、預訓練模型、accuracy、public/private、logits、nn.Sequential、Conv2d（stride、padding、filter）、BatchNorm（buffer：移動平均／移動變異數）、ReLU、MaxPool、卷積區塊、攤平、epoch、patience、weight decay（只提名稱，ch04 解釋）、wall clock、overfitting、deterministic、TTA／cross validation／ensemble（只提名稱，ch06／ch08）。

## ch01 實測（2026-10-05，本機；在 HW03/ 裡，CPU）
- `du -sh food11/*`：test 245M、training 663M、validation 260M。`ls food11/training | head -5`：0_0.jpg、0_1.jpg、0_10.jpg、0_100.jpg、0_101.jpg；`tail -2`：9_998.jpg、9_999.jpg。test 開頭 0001.jpg、0002.jpg。
- train/val 每類序號都是連續的 0..n-1。validation 排序後第 0 個 0_0.jpg、最後 9_99.jpg、10_0.jpg 在索引 362。
- 排序後類別區塊順序（training）：0(994)、10(709)、1(429)、2、…、9；`10_0.jpg` 索引 994、`1_0.jpg` 索引 1703。
- `FoodDataset(..., tfm=T.ToTensor())`：training 第 0 筆 `torch.Size([3, 512, 512]) torch.float32 0`；test 第 0 筆 `torch.Size([3, 384, 512]) -1`。
- 給 `files=['./food11/validation/7_5.jpg']`、path 給 training：印 `One ./food11/training sample ./food11/validation/7_5.jpg`，長度 1、標籤 7；第 17 行的 listdir 照樣執行。
- `super(D)` 印 `<super: <class 'D'>, NULL>`，`.__init__()` 不報錯；`Dataset.__init__ is object.__init__` → True。
- posixpath/ntpath.basename：`'food11/training/3_120.jpg'` 兩者都 `3_120.jpg`；`'food11\\training\\3_120.jpg'` posixpath 原樣、ntpath `3_120.jpg`。`os.path is posixpath` True。
- 舊 Windows 寫法在 Linux：`'food11/training/3_120.jpg'.split('\\')[-1].split('_')[0]` → `'food11/training/3'` → -1。CPU 上 `nn.CrossEntropyLoss()(zeros(2,11), tensor([-1,-1]))` → `IndexError: Target -1 is out of bounds.`
- dataset.py 的 git 紀錄：2274ce9 2022-03-30、403765e 2022-09-29、5e5169b 2022-11-12、717528b 2024-03-28、381aad0 2026-10-03。381aad0 之前的寫法是 `split("\\")`（註解 `# windows`），原版 `split("/")` 留成註解 `# linux`。
- 尺寸極值：最小 training/7_24.jpg (270,207)；最大 validation/10_81.jpg (9216,6144)；training 最大 10_445.jpg (7360,4912)；test 最大 1598.jpg (9542,5126)。
- 驗證 ÷ 訓練（每類）：0.364、0.336、0.333、0.332、0.384、0.339、0.334、0.343、0.406、0.333、0.327；合計 0.348。訓練占比：10.1、4.3、15.2、10.0、8.6、13.4、4.5、2.8、8.7、15.2、7.2（%）。
- 配色：單一系列 `#3987e5`，validate_palette.js dark / surface #161c24 全部 PASS。
- 名詞首見（ch01）：Dataset／__len__／__getitem__、Pillow、字串排序、半監督學習（只提名稱）、ConcatDataset／Subset（只提名稱）、posixpath／ntpath、pathlib、類別不平衡。

## ch02 實測（2026-10-05，本機；在 HW03/ 裡，CPU，訓練同時在跑，所以沒有計時）
- `print(test_tfm)`：`Compose(` / `    Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)` / `    ToTensor()` / `)`。
- `training/1_0.jpg` (512,384) RGB → Resize → (128,128)；左上角像素 (18, 22, 18) → ToTensor `[0.07058823853731155, 0.08627451211214066, 0.07058823853731155]`；shape (3,128,128) float32，min 1/255、max 0.9490196108818054。自己 `permute(2,0,1).float().div(255)` 與 ToTensor `torch.equal` → True。
- 對照圖：`img/squash_keep.jpg`（`thumbnail((128,128))` → (128, 96)，3,870 bytes）、`img/squash_128.jpg`（Resize((128,128))，4,514 bytes）。
- validation、shuffle=False 的第一個 batch：`torch.Size([256, 3, 128, 128]) torch.float32 torch.Size([256]) torch.int64`，前 8 個標籤全是 0。一個 batch 的圖片 50,331,648 bytes。`len(train_loader)` 39、`len(valid_loader)` 14。
- 訓練集 9,866 張經 test_tfm 的通道平均 [0.5549, 0.4509, 0.3436]、標準差 [0.2668, 0.2694, 0.2766]。
- `torch.get_num_threads()` = 24。

## ch02 計時（2026-10-05 15:50–16:10，GPU 上沒有其他程式，前後 `nvidia-smi --query-compute-apps` 皆空）
- `hw03_facts.py timing`（test_tfm、batch 256、shuffle、pin_memory）：只讀訓練集一遍 nw=0 29.5 s、nw=4 4.2 s、nw=8 2.7 s；資料已在 RAM 的 39 步 forward+backward 6.0 s、5.3 s；peak GPU memory 5,826 MiB。
- 重量 nw=0：training 20.6、24.7 s，validation 9.1 s；`torch.set_num_threads(1)` 16.5 s。
- 3 次重複（`load_threads.py`，scratchpad）：24 threads wall 22.5/24.8/24.0 s、CPU 473.0/538.2/518.9 s（2105/2167/2165%）；1 thread wall 15.0/17.1/15.1 s、CPU 15.9/16.0/16.1 s（106/94/107%）。教材用這組。
- 768 張逐步（每 10 張取 1，`steps.py`）：24 threads — decode 0.98/1.33、resize 0.50/0.54、ToTensor 0.15/3.78、collate 0.07/1.64（wall/CPU 秒）；1 thread — 0.93/0.99、0.50/0.53、0.08/0.09、0.07/0.07。
- verify5 每 epoch（train_secs, secs）：(28.8, 38.1)、(27.2, 36.5)、(31.6, 41.1)、(26.2, 37.2)、(25.1, 36.0)。
- 整支 train.py（scratchpad 複本）：`OMP_NUM_THREADS=1` wall 147.89 s、user 137.44、sys 9.29、CPU 99%，stdout 與 sample_best.ckpt 和原版逐位元組相同。
- `num_workers=4`（sed 改兩處）：wall 46.86 s、CPU 430%，**結果不同**：epoch 1 train 1.94046/0.32487、valid 2.91845/0.16355；epoch 5 valid 1.55297/0.48967（最佳是 epoch 4 的 0.52569，epoch 5 沒有存）。
- 拿掉 tqdm（`tqdm(train_loader)` → `train_loader`、valid 同）、`OMP_NUM_THREADS=1`：nw=0 wall 150.19 s、nw=4 43.31 s，stdout 與 ckpt 完全相同；epoch 5 valid 1.39194/0.54913（和有 tqdm 的原版不同，因為少了每個 loop 一次的亂數抽取）。
- 解釋（ch04）：單行程 iterator 的 sampler 種子在第一次 `next` 才抽；多行程 iterator 在 `__init__` 就預取 batch，所以 tqdm_asyncio 丟掉的那個 iterator 也抽走了 sampler 種子（並開了 4 個 worker）。

## ch03 實測（2026-10-05，本機）
- e1.ckpt：`hw03_exp.py --epochs 1 --save`，印出 `[ Train | 001/001 ] loss = 1.95937, acc = 0.31425`、`[ Valid | 001/001 ] loss = 3.20382, acc = 0.17568`（與 train.py 第 1 epoch 相同）。存在 scratchpad/hw03ck/e1.ckpt。
- BN 兩種模式（整個集合一次算的真實 acc；train 模式用 deepcopy + no_grad、batch 256、shuffle generator seed 0）：
  - epoch1：val eval 0.17697／train-mode 0.39679；training set eval 0.18315／train-mode 0.42094。
  - epoch5（sample_best.ckpt）：val eval 0.55394／train-mode 0.57697。
  - base40 最佳（第 27 epoch）：val eval 0.67347／train-mode 0.67201。
- `cnn.1.num_batches_tracked`：39、195、1053。cnn.1 running_mean[:4]：e1 [0.2177, 0.2555, 0.1325, 0.2693]、ep5 [0.1787, 0.2861, 0.0863, 0.2041]、ep27 [0.1368, 0.3869, -0.0345, 0.1515]；running_var[:4]：e1 [0.029, 0.0546, 0.0191, 0.0435]、ep5 [0.0094, 0.0362, 0.0029, 0.0191]、ep27 [0.0061, 0.0387, 0.0028, 0.012]。
- e1、一個 shuffled val batch（seed 0）的 conv1 輸出：mean[:4] [0.2196, 0.263, 0.1266, 0.275]、var[:4] [0.0126, 0.0384, 0.0026, 0.0264]。conv5（cnn.16）batch mean[:3] [0.468, 1.097, 0.017] vs running_mean [0.575, 1.073, 0.042]；batch var [1.424, 1.362, 2.064] vs running_var [1.398, 1.139, 1.783]。
- 0.9**39 = 0.016423203268260675。
- 乘加次數（一張圖）：cnn.0 28,311,552（2.6%）、cnn.4/8/12 各 301,989,888（27.6%）、cnn.16 150,994,944（13.8%）、fc.0 8,388,608（0.8%）、fc.2 524,288、fc.4 5,632；合計 1,094,194,688。
- MaxPool2d(2,2,0) 對 arange(16).view(1,1,4,4) → [[5,7],[13,15]]。Conv2d(3,64,3,1,1) weight [64,3,3,3]、bias [64]；padding 0 → [1,64,126,126]；stride 2 padding 1 → [1,64,64,64]。新 BN：parameters weight/bias；buffers running_mean(0)/running_var(1)/num_batches_tracked。`view(2,-1)` 與 `flatten(1)` 相等。
- 名詞首見（ch03）：kernel、feature map、參數共用、局部性、γ/β、momentum、running 統計量、num_batches_tracked、感受野、跨距（jump）、MAC（乘加）、logits 不需 softmax、全域平均池化、VGG。

## ch04 實測（2026-10-05，本機）
- `hw03_exp.py --gradlog 1`（新增選項，記錄 clip 前的梯度 L2 長度，不抽亂數；15 行輸出與 train.py `diff` 相同）。每 epoch（第一步、最小、中位數、最大、>10 的步數）：1: 2.342, 1.839, 3.083, 6.748, 0；2: 2.287, 2.205, 3.541, 5.285, 0；3: 5.441, 3.306, 4.312, 5.921, 0；4: 3.149, 3.149, 4.666, 6.206, 0；5: 5.127, 3.982, 5.127, 9.962, 0。195 步全部沒有被裁剪。40 epoch 以上沒有記錄。
- 每 epoch 真實 val acc／CE（答對 ÷ 3430、CE sum ÷ 3430）：0.17697/3.19892、0.43090/1.62801、0.47988/1.54998、0.48163/1.59486、0.55394/1.32449；印出的 acc 0.17568、0.43092、0.47906、0.48411、0.55541。
- 第一個訓練 batch 的 loss 2.4208555221557617（scratchpad 的 train.py 複本 t3.py 印的）；ln 11 = 2.3979。
- tqdm：`train.py:19` `from tqdm.auto import tqdm`；`tqdm/asyncio.py` 的 `__init__` 第 37 行 `self.iterable_iterator = iter(iterable)`（教材引用 23–38 行，中間兩個 `__anext__`/`__aiter__` 分支以 `...` 省略）。追蹤 `Tensor.random_` 看到抽亂數的位置是 torch `dataloader.py:708`（`_BaseDataLoaderIter.__init__`）。
- 名詞首見（ch04）：cuDNN、deterministic／benchmark、zero_grad／backward／step、梯度累加、gradient accumulation（只提名稱）、log-softmax、Adam、weight decay、AdamW、scheduler／warmup／cosine（只提名稱）、梯度裁剪（L2 norm）、no_grad、early stopping、state_dict、tqdm_asyncio、DataLoader generator。

## ch05 實測（2026-10-05，本機）
- test 檔名排序後 == `[f"{i:04d}.jpg" for i in range(1, 3348)]`（連續、補零）。
- `pad4`：1→'0001'、42→'0042'、3347→'3347'、12345→'12345'；1..9999 全部 == `str(i).zfill(4)` == `f"{i:04d}"`。
- squeeze 邊界：(1,11) logits → `np.argmax(...,axis=1).squeeze().tolist()` → `3`（int）；`[1,2] += 3` → `TypeError: 'int' object is not iterable`。19 張 → list。3346 = 2×7×239。
- `inspect.signature(torch.load)` 的 weights_only 預設是 None（執行時決定；PyTorch 2.6 起等同 True）。
- sample_best.ckpt 在驗證集（shuffle=False）：acc 0.5539358854293823；各類 recall [0.456, 0.285, 0.228, 0.581, 0.641, 0.668, 0.544, 0.208, 0.66, 0.802, 0.651]；被猜成各類 [412, 122, 180, 453, 377, 488, 150, 38, 484, 541, 185]、真實 [362, 144, 500, 327, 326, 449, 147, 96, 347, 500, 232]。最大混淆（張數, 真, 猜）：(72, 2, 8)、(71, 2, 3)、(65, 2, 5)、(50, 3, 0)、(47, 2, 0)、(47, 0, 4)。第 7 類那一列 [23, 2, 0, 2, 2, 2, 22, 20, 7, 15, 1]。
- 隨機對應的期望準確率（用驗證集的預測與真實分布）：Σ pred_k × true_k ÷ 3430² = 1,253,923 ÷ 11,764,900 ≈ 0.1066。
- `submission.csv` 用 csv 讀：`['Id', 'Category']`、`['0001', '9']`、3,347 列。
- 名詞首見（ch05）：state_dict 載入、weights_only、recall、混淆（confusion）。
- 上傳前檢查（ch05 5.5 節逐字）：`['Id', 'Category'] 3347`、`Id == test file names: True`、`Id unique: True`、`Category values: [0, 1, ..., 10]`。

## ch06 實測（2026-10-05，本機，CPU）
- `test_tfm` 對 0_0.jpg 5 次：全部 `torch.equal`。
- 單一轉換示範圖（`torch.manual_seed(1)` 後依序產生）：`img/aug_crop.jpg`（RandomResizedCrop((128,128), scale=(0.5,1.0))）、`aug_flip.jpg`（Resize + RandomHorizontalFlip(p=1.0)，示範用）、`aug_rot.jpg`（Resize + RandomRotation(15)）、`aug_jitter.jpg`（Resize + ColorJitter(0.3,0.3,0.3)）。mixup 示範：`mixup_b.jpg`（9_0.jpg 經 test_tfm）、`mixup.jpg`（0.5×0_0 + 0.5×9_0）。這 6 張共 36,337 bytes。
- `torch.manual_seed(0)` 後 RandomResizedCrop.get_params 對 0_0.jpg ×5（上, 左, 高, 寬）：(8, 30, 410, 478)、(22, 50, 398, 430)、(77, 18, 420, 454)、(8, 56, 403, 333)、(20, 2, 415, 366)。RandomRotation(15).get_params ×5：-0.11、8.05、-12.35、-11.04、-5.78 度。
- CrossEntropyLoss 機率標籤（seed 0 的 randn(2,11)，第 0 列 0/9 各 0.5、第 1 列 one-hot 3）：2.9584498405456543，手寫 `-(y*log_softmax).sum(1).mean()` 相同；one-hot 機率標籤與整數標籤結果相同（allclose）。
- base40 vs augA40（epoch: train_loss, train_acc, 真實 val acc, 真實 val CE）：base40 5: 1.07386, 0.62643, 0.55394, 1.32449；10: 0.49362, 0.83457, 0.51545, 1.82987；20: 0.00442, 0.9999, 0.65335, 1.71685；30: 0.00037, 1.0, 0.67405, 1.78995；40: 0.00016, 1.0, 0.67259, 1.90464。augA40 5: 1.45072, 0.49833, 0.50292, 1.42699；10: 1.16577, 0.59332, 0.56589, 1.3402；20: 0.80724, 0.71823, 0.60729, 1.19178；30: 0.65052, 0.77397, 0.63907, 1.18154；40: 0.4792, 0.8346, 0.67405, 1.10745。
- 最低真實 val CE：base40 1.2509（epoch 6）、augA40 0.99139（epoch 35）。平均每 epoch train_secs：base40 27.4、augA40 30.1（augA40 第 40 epoch 41.8 s 受本機其他量測干擾）。
- 名詞首見（ch06）：RandomResizedCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、mixup、λ／Beta 分布（只提名稱）、TTA 的加權平均、TrivialAugmentWide／RandAugment／AutoAugment／CutMix（只提名稱）。

## ch07 實測（2026-10-05，本機）
- `exec(others.py)` 整份 → `NameError: name 'transforms' is not defined`（第 9 行）。
- Residual_Network 卷積部分合計 1,147,200；fc_layer 67,111,947（98.3%）；res40.ckpt 273,056,689 bytes。
- **res0_40**（`--arch res0 --aug A --epochs 40`，使用者 2026-10-05 核可加跑）：最佳 epoch 39，印出 0.64569、真實 0.64606；每 epoch 平均 45.0 s；總 1819.2 s。曲線（train_loss, train_acc, 真實 val acc, 真實 val CE）：1: 8.45978, 0.15576, 0.13644, 2.31395；5: 1.88658, 0.34072, 0.35569, 1.81957；10: 1.6369, 0.43391, 0.44227, 1.64412；20: 1.25558, 0.57164, 0.53907, 1.37907；30: 1.04897, 0.64194, 0.61312, 1.14766；40: 0.88823, 0.69775, 0.5793, 1.33965；最低 CE 1.06869（epoch 34）。
- res40 曲線：1: 19.25814, 0.13929, 0.17114, 3.83432；5: 2.01706, 0.29604, 0.31866, 1.93446；10: 1.76762, 0.39163, 0.37959, 1.75989；20: 1.42594, 0.51025, 0.46297, 1.62354；30: 1.19627, 0.59062, 0.54723, 1.37563；40: 1.05876, 0.63875, 0.5723, 1.30747；最低 CE 1.21018（epoch 35）；每 epoch 平均 46.6 s。augA40 每 epoch 平均 39.7 s。
- 初始化時（CPU、32 張訓練圖、seed 0、train 模式）：Classifier loss 2.417、logits std 0.105、|logits| max 0.34、grad norm 4.82；Residual_Network 原樣 2.498 / 0.172 / 0.55 / 29.55；+ 殘差 2.528 / 0.263 / 0.85 / 48.90。攤平特徵：殘差版 262,144 維、mean 0.654、std 0.853、每張 L2 543.3；Classifier 8,192 維、mean 0.857、std 0.742、L2 100.6。
- 前 10 步（`--max_batches 10 --gradlog 1`，增強 A、seed 6666）loss：cnn [2.425, 2.538, 2.473, 2.277, 2.289, 2.312, 2.229, 2.188, 2.141, 2.063]；res0 [2.414, 25.471, 28.612, 28.77, 24.166, 22.998, 20.159, 18.898, 15.016, 9.961]；res [2.435, 50.531, 78.717, 46.655, 49.067, 52.124, 44.079, 35.603, 31.509, 33.952]。梯度長度：cnn [2.38, 5.1, 4.9, 3.5, 2.36, 2.11, 2.09, 2.0, 2.27, 1.93]；res0 [12.06, 139.97, 262.68, 306.86, 221.19, 306.21, 165.97, 170.68, 193.5, 132.48]；res [20.41, 371.76, 530.89, 417.35, 653.68, 449.05, 558.67, 416.13, 470.4, 500.58]。
- hw03_exp.py `--gradlog` 另記 `grad_first10`、`loss_first10`。
- 名詞首見（ch07）：殘差（residual／skip connection）、projection shortcut、欠擬合（underfitting）、pre-activation（只提名稱）。

## ch08 實測（2026-10-05，本機；`hw03_run_grid.sh` + res0_40；推論用 `hw03_facts.py tta <ckdir>`）
- 全部 run 的摘要：`docs/tools/hw03_ch08_runs.txt`（由 jsonl 產生）。總時間（分）：base40 25.2、augA40 26.8、res0_40 30.3、res40 31.4、resplit40 33.6、ls40 27.2、long200 137.1。
- augA40 的 40 個 epoch 真實驗證 acc 與 long200 前 40 個完全相同（assert 過）。
- holdout（`random.Random(0).shuffle(range(3430))` 前 1143 個，與 `--resplit 1` 相同）上的 acc：base40 0.68241、augA40 0.69991、res0_40 0.65967、res40 0.62030、resplit40 0.72616、ls40 0.70079、long200 0.76203。resplit40 的 full-val 0.77901 無意義（2/3 訓練過）。
- long200：最佳 epoch 187（印出 0.76669、真實 0.76501）；第一次真實 > 0.73207 在 epoch 68（印出也是 68）；200 個 epoch 中 68 個真實 > 0.73207；共存檔 27 次；最低真實 CE 0.9887（epoch 43）；最高 train acc 0.9836；最後 20 epoch 真實 acc 平均 0.74513、範圍 0.72566–0.76501；前 40/80/120/160/200 的最佳（印出挑選）真實值：0.68805(35)、0.73411(72)、0.76093(119)、0.76093(119)、0.76501(187)；平均 41.0 s/epoch。曲線（train_acc, 真實 acc, 真實 CE）：1: 0.27928, 0.19155, 2.55572；20: 0.71823, 0.60729, 1.19178；40: 0.8346, 0.67405, 1.10745；60: 0.90848, 0.7207, 1.12334；80: 0.94132, 0.7137, 1.27017；100: 0.95808, 0.72624, 1.28519；120: 0.96829, 0.6688, 1.87311；140: 0.96839, 0.73149, 1.41682；160: 0.9812, 0.7207, 1.59096；180: 0.97751, 0.73586, 1.52248；200: 0.97731, 0.75219, 1.33576。
- TTA（K=5 增強 A 版本、`torch.manual_seed(0)`、softmax 機率）：test_tfm／單一增強版本範圍／5 版平均／0.5·aug+0.5·test／6 版等權重 — base40 0.67347 / 0.55685–0.57172 / 0.63819 / 0.67609 / 0.65831；augA40 0.68805 / 0.66472–0.67726 / 0.70379 / 0.70962 / 0.70612；ls40 0.69592 / 0.67230–0.68163 / 0.70321 / 0.71545 / 0.70933；long200 0.76501 / 0.74636–0.75423 / 0.78163 / 0.78309 / 0.78950。
- ensemble（softmax 平均、無 TTA）：augA40+ls40 0.73032；augA40+ls40+long200 0.78105；base40+augA40+ls40+long200 0.78105；augA40+res0_40+ls40+long200 0.78017；long200+ls40 0.77726。
- 圖 8.1：配色 #3987e5（long200）、#d95926（base40），validate_palette.js dark/#161c24 全部 PASS（CVD ΔE 26.8、normal 31.8）；y 軸 0.1–0.9。
- 沒有做的：mixup、cross validation、cosine 排程、多種子、TTA+ensemble 疊加、Kaggle 上傳（題庫裡標明）。
