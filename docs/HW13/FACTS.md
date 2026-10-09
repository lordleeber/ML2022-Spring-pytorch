# HW13 事實清單（不進最終教材）

量測環境：RTX PRO 4000 Blackwell（24 GB）、24 核心、共用 `.venv`（Python 3.12、torch 2.11.0+cu128、torchvision 0.26.0、numpy 2.5.3、pandas 3.0.6、Pillow 12.3.0、tqdm 4.70.1、torchsummary 1.5.1——使用者決定裝進 .venv）。

## 來源（Phase 0，2026-10-09）

- 官方 GitHub 的 HW13 只有資料（`food11-hw13.tar.gz`），沒有範例程式。投影片 p.2 的兩個 Colab 連結用 Google Drive 匿名下載成功（使用者同意）：
  - `HW13/HW13.ipynb` ← drive id `1S7J12rzL4m5BjSk0QqYw2cnrcqP46VuV`（216,273 bytes，32 格，作者 Liang-Hsuan Tseng，「modified from ML2021-HW13」，Kaggle 版 metadata：Python 3.7.12）。
  - `HW13/HW13_pruning_example.ipynb` ← drive id `1Wy4EemeVnP7xUD8bMn-725j0qeJ3AplB`（4 格，「network pruning example for report Q3-1」：LeNet + `prune.l1_unstructured(module, name='weight', amount=0.2)` 套在所有 Conv2d）。
- 投影片：`HW13/Machine Learning HW13.pdf`（23 頁）。
- 資料：本機 Kaggle zip `ml2022spring-hw13.zip` 解壓到 `HW13/food11-hw13/`（.gitignore 排除；`HW13/outputs/` 也排除）。訓練 9,866、驗證 3,430、測試（evaluation）3,347，加 `resnet18_teacher.ckpt`（44,806,605 bytes）。

## 原版 notebook vs 本 repo

| 檔案 | 對應 notebook 格 | 內容 |
|---|---|---|
| `config.py` | [5] | `cfg`（batch 64、lr 3e-4、seed 20220013、CE、wd 1e-5、grad_norm_max 10、10 epoch、patience 300） |
| `dataset.py` | [12]、[13] | `normalize`、`test_tfm`、`train_tfm`、`FoodDataset` |
| `model.py` | [16]、[19]、[22] | `dwpw_conv`、`StudentNet`、`get_student_model`；`get_teacher_model(dataset_root)` 包住 [22] |
| `kd.py` | [24] | `loss_fn_kd`，TODO 已填 |
| `train.py` | [6] [10] [14] [21] [22] [25] [27] | 種子、log、資料、summary、老師、選 loss、訓練迴圈；`--loss_fn_type KD` 打開 notebook 標 `# MEDIUM BASELINE` 的行；`--exp_name`、`--n_epochs` |
| `test.py` | [29]、[30] | 讀 `outputs/<exp>/student_best.ckpt` 寫 `submission.csv` |

改動：
- 每行行尾空白去掉（不影響語意）。
- **老師模型**：notebook 用 `torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)`，會從 GitHub 下載 torchvision v0.10.0 的原始碼；本 repo 用已安裝的 `torchvision.models.resnet18(weights=None, num_classes=11)`（`pretrained` 參數在新版已移除）。load_state_dict 成功、所有 key 對得上。**v0.10.0 與 0.26 建構時的亂數消耗是否一樣沒有實測**（要下載才能比）；本書的「參照版」也用本機 torchvision。
- **建老師模型會消耗全域亂數**（權重初始化在 load 之前），所以 `train.py` 在 CE 模式也照 notebook 的位置建老師，否則學生的訓練順序不同。
- `!wget`、`!tar` 去掉（資料已在本機）。
- notebook 的 `for dirname ... os.walk('./food11-hw13')` 寫死路徑，不用 `cfg['dataset_root']`。

### 參照版與逐位元驗證
- 參照版：`docs/tools/hw13_make_ref.py HW13/HW13.ipynb <out.py>` 把程式格原樣串起來，只註解掉 `!` 指令、把 `torch.hub.load` 換成本機 torchvision，其他一字不改。在 scratchpad 跑（`food11-hw13` 用 symlink）。
- 參照版 Simple（10 epoch）：09:10:46–09:19:05，**8 分 19 秒**（wall 8:19.22），user 7789 s、CPU 1567%（PyTorch 開滿 24 核），最大 RSS 2.2 GB；前後 `nvidia-smi --query-compute-apps` 都是空的。投影片：Simple < 1 小時（Kaggle）。
- `python train.py` + `python test.py`（HW13/）：stdout 與參照版只差一行（`One ./food11-hw13/evaluation sample ...` 移到 test.py 印）；`student_best.ckpt` 每個張量 `torch.equal`；`submission.csv` `cmp` 相同。train.py 507.29 s（同時有別的 CPU 工作，只當參考）。

### Simple baseline 逐 epoch（參照版 = train.py，%）

| epoch | train loss | train acc | valid loss | valid acc |
|---|---|---|---|---|
| 1 | 1.96877 | 0.32769 | 1.86308 | 0.34956 best |
| 2 | 1.80251 | 0.38425 | 1.75736 | 0.37405 best |
| 3 | 1.70642 | 0.42023 | 1.70051 | 0.41224 best |
| 4 | 1.64257 | 0.43858 | 1.63731 | 0.43878 best |
| 5 | 1.58321 | 0.46665 | 1.63385 | 0.44286 best |
| 6 | 1.54136 | 0.48034 | 1.65186 | 0.44082 |
| 7 | 1.50853 | 0.49260 | 1.52971 | 0.45918 best |
| 8 | 1.47209 | 0.50932 | 1.53104 | 0.46268 best |
| 9 | 1.44812 | 0.50710 | 1.47539 | 0.50408 best |
| 10 | 1.42286 | 0.51520 | 1.52090 | 0.50175 |

最佳在第 9 個 epoch（log 印「Best model found at epoch 8」，epoch 從 0 數）。驗證 0.50408 > Simple 基準線 0.44820（基準線是 Kaggle 測試集的數字，本書只能說驗證集）。

### 驗證指標沒有偏差
- valid_loader `shuffle=False`、沒有 drop_last；acc 是答對總數 / 3430、loss 是逐 batch `loss.item() * batch_len` 加總 / 3430。
- `docs/tools/hw13_facts.py`：最佳 checkpoint 在整個驗證集重算 acc **0.50408（1729/3430）**、CE **1.47539**，與印出的第 9 個 epoch 完全相同。

## 參數量（`hw13_facts.py`，torchsummary 1.5.1 與 numel）

| 模型 | parameters | torchsummary Total params | buffers（BN running mean/var + num_batches_tracked） |
|---|---|---|---|
| StudentNet | **87,907** | 87,907 | 460（float 456 + 4 個 long） |
| ResNet-18 老師 | **11,182,155** | 11,182,155 | 9,620（float 9,600 + 20） |

- torchsummary 的 Total params 只數 `parameters()`，BN 的 running mean/var 不算；「Non-trainable params: 0」。投影片 p.11 說「non-trainable parameters should also be considered」——指的是 `requires_grad=False` 的參數（例如凍結層），不是 buffer。
- 老師 / 學生 = 127.2 倍。
- 學生各層（torchsummary）：Conv 3→32 3×3：896；BN 64；Conv 32→32：9,248；BN 64；Conv 32→64：18,496；BN 128；Conv 64→100：57,700；BN 200；Linear 100→11：1,111。輸出形狀 222→220→110→108→54→52→26→1。

## 老師模型
- 驗證集 acc **0.86093（2953/3430）**、CE 0.64702（`test_tfm`，batch 64）。notebook 註解「test-acc ~= 89.9%」、投影片 p.7「test-Acc ≅ 0.899」——那是 Kaggle 測試集，本機無法驗證；驗證集低 3.8 個百分點。

## 資料與 HW03 的關係
- notebook [7] 說「We've modified the dataset ... DO NOT utilize the dataset of HW3」。實測（md5 全量比對 `HW03/food11`）：
  - training 9,866、validation 3,430：**檔名與內容完全相同**（0 張不同）。
  - evaluation 3,347：每一張的位元組都在 HW03 `test/` 裡，但**重新打亂編號**（HW13 `0000.jpg` = HW03 `0095.jpg`，`0001` = `1283`，…；HW03 從 0001 起編、HW13 從 0000 起編）。
  - 結論：只改了測試集的檔名順序（HW03 的 submission 不能直接拿來交）；訓練、驗證的事實可以沿用 docs/HW03。
- 投影片 p.4「Same as HW3」。

## KD loss（`kd.py`）
- 填法：`alpha * T² * kl_div(log_softmax(s/T), softmax(t/T), 'batchmean') + (1-alpha) * CE(s, y)`。
- 與 `nn.KLDivLoss(reduction='batchmean')` 照官方文件範例的寫法逐值相同（驗證集前 64 張、最佳學生 vs 老師 logits）：α=0.5 T=1：1.192444；α=0.5 T=4：6.928273；α=0 T=1（純 CE）：1.237240；α=1 T=1（純 KL）：1.147648。
- **文件與程式不一致**：notebook [23] 的公式寫 `KL(p || q)`，p = 學生、q = 老師；但 `kl_div(input=學生 log 機率, target=老師機率)` 算的是 KL(老師 ‖ 學生)，Hinton 的原意也是這個方向。

## 速度
- 一個 epoch（nw=0、預設執行緒）：train 155 批 + valid 54 批，整支 10 epoch 8 分 19 秒 ≈ 50 秒 / epoch。
- 只讀資料一輪（training、train_tfm、batch 64、shuffle）：nw=0 預設 24 執行緒 21.1 s；nw=0 `OMP_NUM_THREADS=1` 21.7 s；nw=8 3.4 s；nw=16 2.3 s。讀圖（Resize 256 解 JPEG）大約佔一半時間。
- `num_workers>0` 會改變亂數消耗（HW03 的教訓），所以改 worker 數的實驗不會和 train.py 逐位元一致。

## 讀程式找到的疑點（待寫章時處理）
1. 學生模型第一層在 222×222 上做 32 通道 3×3 卷積，計算量集中在前兩層，參數卻集中在最後一層（57,700 / 87,907）——「參數少 ≠ 算得快」。MACs 待量。
2. 範例學生沒有用到 `dwpw_conv`（只定義了），換成 depthwise/pointwise 是 Strong 的方向。
3. KD 的驗證 loss 也用 `loss_fn`（KD 模式下是 KD loss），所以 CE 與 KD 兩組印出的 valid loss 不能直接比；acc 可以比。
4. `test.py` 的 `preds = list(logits.argmax(dim=-1).squeeze()...)`：最後一批若只有 1 張，`squeeze()` 變 0 維、`list()` 會出錯。3347 % 64 = 19，所以這份資料不會觸發。
5. 剪枝範例用 `prune.l1_unstructured`，只加 mask（`weight = weight_orig * weight_mask`），計算量不變——報告 Q3-2 的答案，要實測推論時間。

## 實驗工具（Phase 1，2026-10-09）
- `docs/tools/hw13_exp.py`：預設（CE、sample 學生、範例增強、10 epoch、`--nw 0`）與 train.py 逐位元一致——20 行 Train/Valid 輸出 `diff` 相同、best.ckpt 每個張量 `torch.equal`（523.4 s）。兩個迴圈都包 `tqdm(..., disable=True)`。額外記錄（不碰亂數）：每個 epoch 的驗證 CE、train/valid 秒數；最佳 epoch 的驗證 logits。
- `docs/tools/hw13_students.py`：`sample` 87,907；`dw`（MobileNet-v1 風格，stem 3→24 s2 + 9 個 dwpw 區塊，寬 32/64/64/96/96/128/128/128/144，步幅 1/2/1/2/1/2/1/1/2）**99,811**；`plain`（同步幅與深度、一般 3×3 卷積，stem 16，寬 16/24/24/32/32/40/44/48/56）**99,355**；`mbv2`（inverted residual，stem 16，(t,c,n,s) = (1,16,1,1)(4,24,2,2)(4,32,2,2)(4,48,2,2)(4,64,1,2)，最後 1×1 到 160）**96,651**。dw 區塊用 model.py 的 `dwpw_conv`（兩個卷積都有 bias）再各接 BN+ReLU。
- `docs/tools/hw13_run_grid.sh`：A 組，16 workers，結果寫 `docs/tools/hw13_runs.jsonl`，log 在 `docs/tools/hw13_logs/`。
- 圖書樣板：`docs/tools/hw13_book/`（`book.css`、`book.js`、`apply_template.py`、`fill_listings.py`），由 HW14 改：強調色薄荷青 #5cc9ae、標題 Noto Serif TC、等寬 JetBrains Mono；圖表底 #181e25 與 HW14 相同，角色色沿用 HW14 驗證過的色盤（老師 #9085e9、CE #8b949e、KD #3987e5、arch1 #d95926、arch2 #199e70、剪枝 #d55181、量化 #c98500）。
