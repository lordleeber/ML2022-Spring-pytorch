# HW04 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW04/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **狀態（2026-10-05）**：Phase 1 完成：outline.html、index.html、樣式、實驗工具（逐位一致）、ch08 的 10 組實驗、資料／模型事實都已量完。下一步寫 ch00。
>
> **寫作方式**：本機 session 一章一章寫，一章一停，使用者說「推」才推上 master、一次推一章。**這本書不冷讀**（使用者 2026-10-05 決定），只跑 verify_book.py。
>
> **重現方法**（需要 GPU 與 `HW04/Dataset/`）：在 `HW04/` 裡跑
> - `PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw04_exp.py --name X [選項]`：訓練實驗，預設參數就是 train.py，不寫檔（除非給 `--save`／`--save_live`／`--dump`）。
> - `PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw04_facts.py <data|mel|split|model|sched|workers|figs>`：資料與模型的事實。
> - `bash docs/tools/hw04_run_grid.sh <ckpt_dir> [name ...]`（在 repo 根目錄）：ch08 的全部實驗，一次一組依序跑；結果追加到 `docs/tools/hw04_ch08_runs.jsonl`／`.txt`。

教材對應 commit：`38cac04`（HW04 程式最後一次改動是 `a517359`，2026-10-03）。
原始碼根目錄：`HW04/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW04/hw04.ipynb`（Colab notebook，不在 repo 裡）。作業投影片：`HW04/Machine Learning HW4.pdf`（27 頁，已在 repo）。

## 全書約定
- 樣式：mySkills **新版共用資產**（`completed-repo-to-html-textbook/assets/style.css`、`enhance.js`，與 docs/HW03 相同）。頁面寫 `<!-- INLINE-ASSETS -->`，寫完跑 `inline_assets.py docs/HW04`。
- 指令塊：讀者要貼上執行的用 `<pre class="shell cmd">`，輸出用 `<pre class="shell">`；Python 用 `<pre class="py">`。
- listing 的 `data-hot` 寫**原始碼行號**。
- 檢查：`python3 docs/tools/verify_book.py . docs/HW04/chNN.html`。
- 每本書的規則：ch00 有「模型總覽」（架構 SVG、每層 tensor 形狀、參數量手算 + PyTorch、定義在哪個檔案），放在任務說明之後；目錄頁有「2022 vs 現在」導論；過時的寫法加「現在的做法」框。
- 大綱：`docs/HW04/outline.html`（使用者 2026-10-05 核可：index、ch00–ch08、appendix）。
- 實驗規格（使用者 2026-10-05 核可）：每一組都跑**原規格 70,000 步**（一組 4–10 分鐘），GPU 一次一組，依序自己排；每 30 分鐘回報。
- 圖片：放少量真實資料圖（mel 熱圖、長度分布），標註 VoxCeleb2、CC BY 4.0；放在 `docs/HW04/img/`（PNG 沒被 .gitignore 排除）。

## 環境（實測 2026-10-05）
- Python 3.12.3、torch 2.11.0+cu128（CUDA 12.8、cuDNN 91900）、numpy 2.5.3、tqdm 4.70.1、matplotlib 3.11.2（只有畫圖用）。**torchaudio 沒有安裝**。
- GPU：NVIDIA RTX PRO 4000 Blackwell（24,467 MiB），驅動 596.71，WSL2；24 核心；RAM 47 GB。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW04/` 裡用 `../.venv/bin/python train.py`。`train.py:244`、`test.py:53` 寫死 `device = "cuda"`，**這是刻意的設計**，教材以中性描述，不列為問題或練習。

## 程式檔
- `HW04/` 4 個 .py 共 524 行：dataset.py 53、classifier.py 70、train.py 310、test.py 91。最後一次改動 `a517359`（2026-10-03，"Use CUDA everywhere…"）。
- `HW04/.gitignore`（23 bytes）：`Dataset/*`、`*.csv`、`*.ckpt`。所以 `Dataset/`、`model.ckpt`、`output.csv` 都沒被追蹤（根目錄 `.gitignore` 另有 `*.log`）。
- repo 裡原有的 `HW04/model.ckpt`（509,373 bytes，2026-10-03 01:24）與 `output.csv`（400,013 bytes）是之前跑過的一次，**和今天重跑的不同**（tensor 不相等）；兩份 output.csv 的預測有 98.3375% 相同（7,867／8,000）。教材不用舊檔。

### 原版 notebook vs 本 repo
- notebook 一格一段；本 repo 拆成 dataset.py（myDataset）、classifier.py（Model 說明 + Classifier）、train.py（seed、Data 說明、collate、get_dataloader、**又一份 Model 說明**、scheduler、model_fn、valid、main）、test.py（InferenceDataset、inference_collate_batch、main）。
- notebook 用 **tab** 縮排，本 repo 換成 4 個空白（投影片 p.27「Colab 縮排問題」講的就是這個）。
- `device`：notebook 是 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`；本 repo `train.py:243` 留著這行的註解、`train.py:244` 改成 `device = "cuda"`；test.py 同樣。
- `get_cosine_schedule_with_warmup` 的 docstring（約 20 行）被刪掉；`main(...)` 的參數從一行一個縮成一行。
- test.py：notebook 用 `from tqdm.notebook import tqdm`，本 repo 用 `from tqdm import tqdm`。
- 第一個訓練進度條：notebook `unit=" step"`，本 repo `train.py:260` 是 `unit="step"`（少一個空白），所以第一段印 `353.15step/s`，之後 `train.py:299` 的進度條是 `" step"`，印 `348.98 step/s`。
- `train.py:169` 多一次 `import torch`、`train.py:109-131` 與 `classifier.py:1-23` 是同一段 Model 說明：拆檔的痕跡。
- 其餘（seed 87、segment_len 128、padding -20、batch 32、8 workers、AdamW 1e-3、warmup 1000、70,000 步、模型）全部相同。

## 投影片（HW04/Machine Learning HW4.pdf，27 頁）
- p.1 標題「Speaker Identification」；p.3 Self-attention、目標「Learn how to use Transformer」；p.4 多類別分類、從語音預測說話者。
- p.5：VoxCeleb2；Training 56,666 句有標籤；**Testing「4000 processed audio features (public & private)」**；600 類。p.20：第一行 `Id, Category`、接著 **8000 行**。實際 testdata.json 是 8,000 句 → 投影片前後不一致（可能是 public、private 各 4,000）。
- p.6 前處理圖：waveform → DFT → spectrogram → filter bank → log → mel-spectrogram（引用 2020 DLHLP）。
- p.7 資料格式（與 train.py 的說明相同）。p.8–9：長度不同、「Segment during training, Segment = 2」的示意圖。
- p.10 架構：Input Features → Encoder → Pooling Layer → Linear Layer → Prediction dim=600。
- p.11–16 基準線：Simple 0.60824（跑範例程式，Colab 30–40 分）、Medium 0.70375（調 Transformer 的參數，Colab 1–1.5 小時）、Strong 0.77750（Conformer，Colab 3–4 小時）、Boss 0.86500（Self-Attention Pooling + Additive Margin Softmax，Kaggle 約 2–2.5 小時）。
- p.15 Conformer 圖（論文圖：SpecAug → Convolution Subsampling → Linear → Dropout → Conformer Blocks ×N；block = FFN ×½ → MHSA → Convolution Module（紅框）→ FFN ×½ → Layernorm；卷積模組 = Layernorm → Pointwise Conv → Glu → 1D Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout，加殘差）。引用 2021 ML Network Compression。
- p.17 Self-Attention Pooling 圖（Encoder N× → Self-Attention Pooling Layer（紅框）→ Speaker Embeddings → DNN classifier → Speaker Posteriors）。
- p.18 AM-Softmax 圖（Original Softmax 一條決策邊界；Additive Margin Softmax 兩條邊界、中間 Fixed Decision Margin）。
- p.19 計分：四條基準線 public/private 各 +0.5；Code 2；Report 4；評估指標 @1 Accuracy。
- p.21 交 code 壓成 `<學號>_hw4.zip`，不要交模型或資料。p.22 報告兩題：1. 簡介一種 Transformer 的變體（2 分）；2. 簡要說明為什麼在 Transformer 加卷積層能提升表現（2 分）。p.23 截止 2022/04/01 23:59。p.25 規則：不准用額外資料或預訓練模型；每天最多交 5 次。p.27 Colab 縮排問題（工具 → 設定）。

## 資料（`hw04_facts.py data`）
- 本機來源：見本機的資料位置說明（Kaggle zip）。官方 notebook 從 GitHub release 下載 4 個分割檔 `Dataset.tar.gz.partaa`–`partad` 再 `cat` 起來解壓。
- `HW04/Dataset/` 約 6.6 G：`metadata.json`（7,731,814 bytes）、`testdata.json`（960,912 bytes）、`mapping.json`（20,212 bytes）、64,666 個 `uttr-*.pt`（= 56,666 訓練 + 8,000 測試，路徑都不重複），另有 `log_melspectrogram.pt`、`sox_effects.pt`。
- metadata：`{"n_mels": 40, "speakers": {speaker: [{"feature_path", "mel_len"}, ...]}}`；600 位說話者；第一筆 `id03074` → `{'feature_path': 'uttr-18e375195dc146fd8d14b8a322c29b90.pt', 'mel_len': 435}`。metadata 的說話者順序不是排序過的（`id03074, id05623, id06406, id01014, id02426, …`）。
- testdata：`{"n_mels": 40, "utterances": [...]}`，8,000 句；第一筆 `{'feature_path': 'uttr-b52ddeaacf1b42ff9c947eadce3e1966.pt', 'mel_len': 813}`。
- mapping：`speaker2id`（`id00464→0, id00559→1, id00578→2, id00905→3, id01920→4, …, id08020→598, id00206→599`）與 `id2speaker`（鍵是字串 `"0"`–`"599"`），互為反函數；speaker2id 的順序和 metadata 的順序不同。說話者編號範圍 id00036–id09271。
- 每位說話者 80–111 句，平均 94.44、中位數 94。
- 訓練語句長度（格，一格 10 ms）：min 91、max 7,193、mean 658.1、median 541；p5/25/50/75/95 = 368/441/541/740/1,338；總共 37,291,397 格 ≈ 103.59 小時。≤128 格只有 **2 句**（id02097 的 91 格 `uttr-5de554b2…`、id02475 的 116 格 `uttr-1130d082…`），>1,000 格 6,589 句、>2,000 格 724 句。
  - 直方圖（格）：[0,128) 2、[128,256) 63、[256,384) 4,009、[384,512) 20,710、[512,640) 12,059、[640,768) 6,937、[768,896) 4,048、[896,1024) 2,612、[1024,1536) 4,388、[1536,2048) 1,160、[2048,8192) 678。
- 測試語句長度：min 214、max 4,940、mean 650.1、median 538；p5/25/50/75/95 = 369/437/538/724/1,314；≈ 14.45 小時；沒有短於 128 格的。
- 一個 128 格的隨機片段，平均只看到一句話的 **23.1%**（mean of min(128, L)/L）。

### 前處理器（兩個 TorchScript 檔，資料集附帶，程式沒用到）
- `log_melspectrogram.pt`：`torch.jit.load` 可載入；`LogMelspectrogram` = torchaudio `MelSpectrogram`（sample_rate 16000、n_fft 400、win_length 400、hop_length 160、n_mels 40、f_min 50.0、f_max None、power 2.0、center True、pad_mode reflect）→ squeeze、轉置成 (T, 40) → `clamp(min=1e-9)` → `log`。所以一格 = 160/16000 = 10 ms、窗 25 ms；log(1e-9) = -20.7233，是資料的最小值。
- `sox_effects.pt`：`SoxEffects`，effects 含 `channels 1`、`rate 16000`、`norm -3`、`silence …0.1 1.0%…`（從 data.pkl 的字串看到的片段）。在這個 venv 裡 `torch.jit.load` 失敗：`Unknown builtin op: torchaudio::sox_effects_apply_effects_tensor`——**因為沒裝 torchaudio**（不是 PyTorch 不支援）。
- 用 `torch.load` 載 `log_melspectrogram.pt` 會印 UserWarning：`'torch.load' received a zip file that looks like a TorchScript archive dispatching to 'torch.jit.load'`。

### 一句話的 tensor
- `torch.load('uttr-18e3….pt')` → `torch.float32`、形狀 `(435, 40)`、min -20.7233、max 6.2613、mean -2.3047。
- 全量（`hw04_facts.py mel`，讀 56,666 + 8,000 個檔，約 11 秒）：全部是 float32、2 維、40 個 mel；每個檔的格數都等於 metadata 的 `mel_len`（0 個不符）。
  - 訓練：min **-20.72327**（= log(1e-9) = -20.723266，clamp 的下限）、max 8.92315、整體平均 -2.01732；56,666 個檔裡有 **26,838 個**（47%）的最小值就是這個下限（有能量為 0 的格子，例如靜音或某些頻帶）。
  - 測試：min -20.72327、max 8.90663。
  - 補齊用的 -20 比下限 -20.72 高一點點，和「沒有聲音」幾乎一樣，所以補的格子看起來像靜音。
  - 第一個檔 `uttr-18e3…`（id03074）前 2 格、前 6 個 mel：`[[2.8591, 3.6222, 2.3766, 1.9587, 2.6407, 2.2761], [2.9617, 2.3626, 3.1416, 2.5672, 2.6972, 2.7577]]`。

## 切分與 DataLoader（`hw04_facts.py split`；set_seed(87) 後照 train.py 做）
- `random_split` → 訓練 50,999、驗證 5,667；驗證集的前幾個 index `[43097, 46091, 22324, 24456, 274]`。
- 600 位說話者在訓練、驗證兩邊都有；驗證集每人 1–20 句（平均 9.45）。驗證集長度 mean 658.6、median 541、min 152。
- 兩句短於 128 格的（index 20902、32767）都落在**訓練集**。
- 訓練：`drop_last=True`，一個 epoch 50,999 // 32 = **1,593 步**，每個 epoch 丟掉 23 句（每次打亂後不同）；70,000 步 = **43.94 個 epoch**。
- 驗證：`drop_last=True`，177 個 batch，丟掉最後 **3 句**（沒有打亂，所以永遠是同樣 3 句）；進度條 `total=len(dataset)=5667`，`update(batch_size=32)` 177 次，停在 **`5664/5667`**。
- 177 個 batch 都是 32 句，所以「各 batch 平均再平均」= 5,664 句逐句計數（實測 printed 0.68591 = exact 0.68591）。偏差來自別處（見「驗證指標」）。
- `myDataset[0]` 連取兩次得到不同的 128 格（隨機起點）：同一句話每次被取用都是不同片段。
- worker 亂數（`hw04_facts.py workers`）：每個 worker 的 Python `random` 由 PyTorch 設成 `base_seed + worker_id`；每建一次迭代器，`base_seed` 從主行程的 torch 亂數抽一個新的；所以 8 個 worker 下也能逐位元重現。

## 模型（`hw04_facts.py model`；classifier.py）
- `Classifier(d_model=80, n_spks=600, dropout=0.1)`：`dropout` **沒有被用到**（`classifier.py:38-40` 沒傳）；`TransformerEncoderLayer` 的預設 dropout 剛好也是 0.1。
- 參數（共 **125,896**）：prenet 3,280（80×40+80）；self_attn 25,920（in_proj 240×80+240 = 19,440；out_proj 80×80+80 = 6,480）；linear1 20,736（256×80+256）；linear2 20,560（80×256+80）；norm1、norm2 各 160；pred_layer 55,080（80×80+80 = 6,480；600×80+600 = 48,600）。encoder 合計 67,536（53.6%）；最後一層 48,600（38.6%）。
- `print(model)`：TransformerEncoderLayer 裡有 self_attn（out_proj 是 `NonDynamicallyQuantizableLinear`）、linear1、dropout、linear2、norm1、norm2、dropout1、dropout2。`norm_first=False`（post-norm）、`batch_first=False`、activation relu、2 個頭、head_dim 40。
- 形狀（batch 32、128 格）：輸入 (32, 128, 40) → prenet (32, 128, 80) → permute (128, 32, 80) → encoder_layer (128, 32, 80) → transpose (32, 128, 80) → mean (32, 80) → pred_layer.0 (32, 80) → pred_layer.2 (32, 600)。測試時一句 (1, 4940, 40) → (1, 600)。
- 沒有位置編碼（positional encoding），也沒有 padding mask（forward 沒給 `src_key_padding_mask`）。
- 警告：單一 `TransformerEncoderLayer`（原狀）**不印任何警告**；照 `classifier.py:41` 打開 `TransformerEncoder(..., num_layers=2)` 會印 `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)`。（注意：打開第 41 行後 forward 仍呼叫 `self.encoder_layer`，要一起改成 `self.encoder` 才會真的用 2 層；實驗工具 `--layers 2` 是改好的版本。）

## 學習率（`hw04_facts.py sched`；warmup 1000、total 70000）
- 第 0 次 update（第一個 batch）用的學習率是 **0**：LambdaLR 建立時就把 lr 設成 lr_lambda(0) = 0。
- scheduler 第 1 步 1e-6、第 500 步 5e-4、第 999 步 9.99e-4、第 1000 步 1e-3（最高）、第 10000 步 9.586e-4、第 35500 步 5e-4（一半）、第 50000 步 1.934e-4、第 60000 步 5.094e-5、第 68000 步 2.072e-6、第 69999 步 5.2e-13、第 70000 步 0。
- 沒有 weight decay 設定 → AdamW 預設 weight_decay=0.01。

## baseline 實測（train.py 原封不動，複本在 scratchpad，2026-10-05）
- 計時：GPU 上沒有其他程式（開跑前後 `nvidia-smi --query-compute-apps` 都是空的）。第一次 **3 分 53 秒**（`/usr/bin/time`：user 646.6 s、sys 176.9 s、CPU 353%、最大 RSS 1.51 GB）；第二次 3 分 59 秒。Colab 的估計是 30–40 分（p.12）。
- 可重現：兩次的 `model.ckpt` md5 都是 `0eef222974c5365d4ee42325b269d282`。
- **終端機寬度陷阱**：用 `script -qc` 錄輸出時，非互動終端寬度是 0，tqdm 什麼都不印（log 只剩空行）；要先 `stty cols 120`。
- stdout（逐字，進度條取 2,000/2,000 那一格）：
  ```
  [Info]: Use cuda now!
  [Info]: Finish loading data!
  [Info]: Finish creating model!
  Train: 100% 2000/2000 [00:06<00:00, 353.15step/s, accuracy=0.06, loss=4.01, step=2000]
  Valid: 100% 5664/5667 [00:02<00:00, 2401.58 uttr/s, accuracy=0.15, loss=4.11]
  Train: 100% 2000/2000 [00:05<00:00, 348.98 step/s, accuracy=0.22, loss=3.59, step=4000]
  Valid: 100% 5664/5667 [00:00<00:00, 17183.29 uttr/s, accuracy=0.25, loss=3.51]
  ...
  Step 10000, best model saved. (accuracy=0.4158)
  ...
  Valid: 100% 5664/5667 [00:00<00:00, 15319.48 uttr/s, accuracy=0.67, loss=1.44]
  Train:   0% 0/2000 [00:00<?, ? step/s]
  Step 70000, best model saved. (accuracy=0.6859)
  Train:   0% 0/2000 [00:00<?, ? step/s]
  ```
  - 進度條的 accuracy／loss 是**最後一個 batch** 的值（`train.py:282-286`），不是平均。訓練第 1 步：`accuracy=0.00, loss=6.40, step=1`（ln 600 = 6.397）。
  - 第一次驗證 2 秒（worker 啟動），之後每次不到 1 秒。
  - 最後 `Train: 0% 0/2000` 是第 70,000 步驗證完又建了一個新進度條（`train.py:299`），迴圈就結束了。
- 存檔 7 次：Step 10000 0.4158、20000 0.5222、30000 0.5756、40000 0.6229、50000 0.6589、60000 0.6698、70000 0.6859。
- 35 次驗證（printed acc，hw04_exp.py 逐位一致；進度條上是 2 位小數）：2k 0.1529、4k 0.2549、6k 0.3362、8k 0.3759、10k 0.4158、12k 0.4670、14k 0.4742、16k 0.4966、18k 0.5057、20k 0.5222、22k 0.5263、24k 0.5461、26k 0.5720、28k 0.5551、30k 0.5756、32k 0.5927、34k 0.5939、36k 0.5943、38k 0.6144、40k 0.6229、42k 0.6377、44k 0.6303、46k 0.6391、48k 0.6485、50k 0.6589、52k 0.6630、54k 0.6665、56k 0.6661、58k 0.6698、60k 0.6693、62k 0.6769、64k 0.6723、66k 0.6808、68k **0.6859**（最高）、70k 0.6714。
  - 不是每次都進步：28k、44k、56k、60k、64k、70k 比前一次低。存檔的 7 個時間點裡，第 60,000 步（最佳在 58k：0.6698 > 0.6693）和第 70,000 步（最佳在 68k）「當下那一步不是最佳」，所以這兩次存檔存的是當下的權重、log 卻印最佳的準確率（`Step 60000, best model saved. (accuracy=0.6698)` 的 0.6698 是 58k 的值）。第 10k–50k 次存檔時，最佳剛好就是當下那一步，bug 沒有作用。
- 訓練 loss（每 2,000 步平均，hw04_exp.py）：2k 5.1924、4k 3.6450、…、70k 1.0016；驗證 loss（逐批平均）70k 1.4429。
- 每份 ckpt 509,373 bytes 左右（125,896 × 4 = 503,584 bytes 的 float32，加上名稱與 zip 結構）。
- PyTorch 逐模組計數（`named_modules` + `parameters(recurse=False)`）：`prenet Linear 3280`、`encoder_layer.self_attn MultiheadAttention 19440`（in_proj 掛在 MultiheadAttention 自己身上）、`encoder_layer.self_attn.out_proj NonDynamicallyQuantizableLinear 6480`、`encoder_layer.linear1 Linear 20736`、`encoder_layer.linear2 Linear 20560`、`encoder_layer.norm1 LayerNorm 160`、`encoder_layer.norm2 LayerNorm 160`、`pred_layer.0 Linear 6480`、`pred_layer.2 Linear 48600`、`total 125896`。

## 實驗工具（docs/tools/hw04_exp.py）
- `import train` 直接用 train.py 的 set_seed(87)、get_dataloader、collate_batch、scheduler、model_fn；主迴圈照抄，只多記帳。預設參數下用 `classifier.Classifier`。
- **驗證逐位一致**：預設參數的 `--save_live`（train.py 存下的那份）和 train.py 的 `model.ckpt` 18 個 tensor 全部 `torch.equal`（檔案 md5 不同只因為存的是 deepcopy 的副本，序列化位元組不同）；35 次驗證的 acc 與 train.py 印的一致。
- 預設參數一次：訓練 224.1 s，連同最後的整句評估共 276 s。

## 驗證指標與 deepcopy（orig，hw04_exp.py）
- 印出的 `best_accuracy` 0.68591（第 68,000 步）。
- **deepcopy 的 bug 實際發生了**：最後一次存檔（第 70,000 步）時，最佳在第 68,000 步，但存下的是第 70,000 步的權重（best ≠ live，tensor 不相等）。
- 用 5,667 句（含 drop_last 丟掉的 3 句）評估兩份權重：
  - 第 68,000 步（真正的 best，deepcopy）：**整句 0.85724**、固定 128 格片段 0.68467、整句 CE 0.67034。
  - 第 70,000 步（train.py 存的）：**整句 0.85707**、片段 0.68555、整句 CE 0.6705。
  - → bug 是真的，但在這次的數字上差不到 0.0002（1 句）；固定片段上甚至反過來。教材要照實說：程式錯了、這次運氣好影響很小。
- **切段 vs 整句**：同一個模型，128 格片段 0.685、整句 0.857。test.py 用的是整句。所以 train.py 印的數字嚴重低估了 test.py 的做法在驗證集上的準確率。

## ch08 實驗（hw04_run_grid.sh；結果在 docs/tools/hw04_ch08_runs.jsonl）
- 規格：全部 70,000 步、batch 32、AdamW 1e-3、warmup 1000、seed 87（train.py 原規格）；每組都用 deepcopy 保存最佳、另外記下 train.py 會存的那一份（live）。評估：printed（照 valid()：5,664 句隨機 128 格、逐批平均）、full（5,667 句整句，像 test.py）、crop（5,667 句、固定種子的 128 格片段）。
- GPU 一次一組，2026-10-05 21:05–22:23 依序跑完前 9 組；long_sap_am 22:24–22:52 另外跑（`hw04_run_grid.sh <dir> long_sap_am`）。long_sap_am 每 20k 步的 printed：20k 0.7248、40k 0.7948、60k 0.8310、80k 0.8378、100k 0.8577、120k 0.8649、140k 0.8782、160k 0.8844、180k 0.8941、200k 0.8951。`train_secs` 是訓練迴圈的時間（不含最後的整句評估）。

| 名稱 | 設定（hw04_exp.py 參數） | 參數量 | 訓練秒數 | printed 最佳（步） | full（best） | crop（best） | full CE | full（live） |
|---|---|---|---|---|---|---|---|---|
| orig | （train.py） | 125,896 | 220.0 | 0.68591（68k） | 0.85724 | 0.68467 | 0.67034 | 0.85707 |
| layers2 | `--layers 2` | 193,432 | 308.4 | 0.74117（68k） | 0.89712 | 0.74202 | 0.46493 | 0.89765 |
| med160 | `--d_model 160 --nhead 4 --ffn 512 --layers 2` | 665,304 | 324.9 | 0.79590（64k） | 0.91848 | 0.79954 | 0.39603 | 0.91918 |
| med256 | `--d_model 256 --nhead 8 --ffn 1024 --layers 3` | 2,599,768 | 570.5 | 0.55438（68k） | 0.52126 | 0.55673 | 2.23496 | 0.52056 |
| conf160 | `--arch conformer --d_model 160 --nhead 4 --ffn 640 --layers 2` | 1,326,040 | 507.5 | 0.82998（64k） | 0.93736 | 0.83325 | 0.31411 | 0.93859 |
| conf256 | `--arch conformer --d_model 256 --nhead 4 --ffn 1024 --layers 3` | 4,799,320 | 1024.6 | 0.84657（68k） | 0.94000 | 0.84419 | 0.34737 | 0.93947 |
| conf160_sap | conf160 + `--pool sap` | 1,326,201 | 516.7 | 0.86511（66k） | **0.95306** | 0.86571 | 0.24342 | 0.95412 |
| conf160_sap_am | conf160_sap + `--loss amsm`（m 0.2、s 30） | 1,325,601 | 545.9 | 0.86635（60k） | 0.94900 | 0.86377 | — | 0.95024 |
| seg256 | `--seg 256`（其餘同 orig） | 125,896 | 240.8 | 0.77331（68k） | 0.85636 | 0.76919 | 0.66713 | 0.85477 |
| med256_lr3e4 | med256 + `--lr 3e-4` | 2,599,768 | 575.0 | 0.86405（60k） | 0.94883 | 0.86377 | 0.25650 | 0.94989 |
| med256_pre | med256 + `--norm_first 1`（pre-norm） | 2,599,768 | 586.9 | 0.85999（70k） | 0.95059 | 0.86289 | 0.29837 | 0.95059 |
| long_sap_am | conf160_sap_am + `--steps 210000`（warmup 仍 1000、cosine 拉長到 210k） | 1,325,601 | 1615.5 | 0.90025（194k） | **0.96700** | 0.89571 | — | 0.96683 |

- 觀察（寫章時要照實說）：
  - 打開第 2 層（layers2）+4 個百分點（full）；加寬到 160 再 +2.1；Conformer（同寬 160、2 層）再 +1.9 到 0.937；SAP 再 +1.6 到 0.953；AMSoftmax **沒有**再進步（0.949，同一規格下略低）。
  - **med256 失敗，原因已實測**：更大的 post-norm Transformer（3 層、d 256、8 頭）學得很慢，printed 第 2k 0.1335、第 10k 0.1866、第 70k 0.5486（最佳 0.5544 在 68k），full 0.521，比原版還差。2026-10-05 23:06–23:28 加跑兩組（使用者要求），各只改一個變因：
    - `med256_lr3e4`（學習率 1e-3 → 3e-4）：full **0.94883**，printed 最佳 0.86405（60k）。
    - `med256_pre`（post-norm → pre-norm，`norm_first=True`）：full **0.95059**，printed 最佳 0.85999（70k）。
    - 前幾次驗證（printed acc／該 2k 段的平均 train loss）：
      - med256：2k 0.1335／4.985、4k 0.1879／3.984、10k 0.1866／3.488、20k 0.3287／2.867、40k 0.4663／1.983、70k 0.5486／1.405。
      - lr3e4：2k 0.2064／4.974、4k 0.3875／3.060、10k 0.5925／1.578、20k 0.7168／0.862、40k 0.8189／0.300、70k 0.8591／0.078。
      - pre：2k 0.2779／4.532、4k 0.4481／2.727、10k 0.6208／1.445、20k 0.7066／0.893、40k 0.8084／0.343、70k 0.8600／0.081。
    - 結論：同樣 260 萬參數的模型，**只要降學習率、或只要改成 pre-norm**，就從 0.521 變成 0.95（和 Conformer conf160 的 0.937、conf160_sap 的 0.953 同一級）。所以 med256 的問題是「post-norm 的深一點的 Transformer 在 lr 1e-3（warmup 只有 1,000 步）下訓練不穩」，不是模型太大。同樣 3 層、d 256、lr 1e-3 的 Conformer（conf256）訓練正常，和「pre-norm 就好」一致（Conformer 的每個子模組前先 LayerNorm）；這是推論，沒有另外拆開驗證。
    - 兩組修好的版本，最後的 train loss 都降到 0.08 左右，驗證整句 CE 0.26–0.30：開始過擬合（printed 最佳不在最後一步）。
  - conf256（480 萬參數、17 分鐘）只比 conf160 多 0.3 個百分點。
  - 長跑（Boss 配方 × 3 倍步數，27 分鐘）：full 0.967、printed 0.900；70k 時的同一配方是 0.949 → 訓練更久比換配方更有用。
  - seg256：printed 從 0.686 升到 0.773（因為驗證時看的片段變長了），但 full 幾乎不變（0.856 vs 0.857）→ printed 的提升大多是「量法」改變，不是模型變好。
  - deepcopy（best）和 live 在 full 上的差距都在 ±0.0013 以內；有幾組 live 反而較高。printed 的最佳 ≠ full 的最佳。
  - printed 與 full 的差距隨模型變好而縮小（orig 0.17、conf160_sap 0.09）。
  - layers2／med160／med256 在 stderr 印 `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)`；其他組 stderr 是空的。
- 這些都是驗證集上的數字。驗證集和訓練集是同一批 600 人的句子隨機切的；Kaggle 的測試集分數無法在本機取得，不能宣稱過了哪條基準線。
