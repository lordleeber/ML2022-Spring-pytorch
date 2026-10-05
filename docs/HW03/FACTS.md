# HW03 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW03/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **寫作方式**：本機 session 一章一章寫（使用者 2026-10-05 決定），事實一次量完（Phase 1）。這個 session 停用冷讀（使用者 2026-10-05）。
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
