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
- ~~驗證集 acc 0.86093（2953/3430）、CE 0.64702~~ **錯誤，ch03 更正**：`hw13_facts.py` 在量準確率之前先對老師呼叫 `torchsummary.summary`（老師剛建好是訓練模式），summary 的 forward 用隨機輸入更新了每個 BatchNorm 的 running mean/var。正確值 **0.87143（2989/3430）**，見「ch03 實測」。notebook 註解「test-acc ~= 89.9%」、投影片 p.7「test-Acc ≅ 0.899」——那是 Kaggle 測試集，本機無法驗證；驗證集低 3.8 個百分點。

## 資料與 HW03 的關係
- notebook [7] 說「We've modified the dataset ... DO NOT utilize the dataset of HW3」。實測（md5 全量比對 `HW03/food11`）：
  - training 9,866、validation 3,430：**檔名與內容完全相同**（0 張不同）。
  - evaluation 3,347：每一張的位元組都在 HW03 `test/` 裡，但**重新打亂編號**（HW13 `0000.jpg` = HW03 `0095.jpg`，`0001` = `1283`，…；HW03 從 0001 起編、HW13 從 0000 起編）。
  - 結論：只改了測試集的檔名順序（HW03 的 submission 不能直接拿來交）；訓練、驗證的事實可以沿用 docs/HW03。
- 投影片 p.4「Same as HW3」。

## KD loss（`kd.py`）
- 填法：`alpha * T² * kl_div(log_softmax(s/T), softmax(t/T), 'batchmean') + (1-alpha) * CE(s, y)`。
- 與 `nn.KLDivLoss(reduction='batchmean')` 照官方文件範例的寫法逐值相同（驗證集前 64 張、最佳學生 vs 老師 logits；老師用正確的 BN 統計量，2026-10-09 重跑）：α=0.5 T=1：1.166643；α=0.5 T=4：6.807671；α=0 T=1（純 CE）：1.237240；α=1 T=1（純 KL）：1.096046。（第一版被 summary 汙染的老師：1.192444／6.928273／1.237240／1.147648。）
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

## ch00 實測（全貌，2026-10-09）
- 老師各段參數（torchvision resnet18, num_classes=11）：conv1 9,408；bn1 128；layer1 147,968；layer2 525,568；layer3 2,099,712；layer4 8,393,728（75.1%）；fc 5,643（512·11+11）。合計 11,182,155。
- 老師各段輸出形狀（輸入 1×3×224×224）：conv1/bn1/relu (64,112,112)；maxpool (64,56,56)；layer1 (64,56,56)；layer2 (128,28,28)；layer3 (256,14,14)；layer4 (512,7,7)；avgpool (512,1,1)。
- 老師 checkpoint：`OrderedDict`，122 個 key，參數 + buffer 共 11,191,775 個數，張量本身 44,767,180 bytes；檔案 44,806,605 bytes。
- 學生 `student_best.ckpt`：30 個 key，88,367 個數（87,907 參數 + 456 個 running mean/var + 4 個 int64 `num_batches_tracked`），張量 353,484 bytes，檔案 363,645 bytes。老師 / 學生檔案大小 123 倍。
- `submission.csv`（Simple，test.py）：3,348 行（表頭 + 3,347）；預測各類張數 0:636、1:38、2:531、3:331、4:263、5:202、6:50、7:21、8:303、9:734、10:238。訓練集比例（HW03 FACTS）第 1 類 4.3%、第 6 類 4.5%、第 7 類 2.8%；預測只佔 1.1%、1.5%、0.6%。
- test.py 的 stdout：`One ./food11-hw13/evaluation sample ./food11-hw13/evaluation/0000.jpg`（tqdm 進度條在 stderr）。

## ch01 實測（程式結構與訓練迴圈，2026-10-09）
- **建老師會改變學生的訓練**：照 train.py 的順序設種子、建 DataLoader、建學生後，「建老師」與「不建老師」兩種情況：學生第一層權重 `torch.equal`（學生在老師之前建）；之後全域 RNG 狀態不同；第一個訓練 batch 的前 12 個標籤：建老師 `[7, 1, 8, 0, 7, 8, 0, 3, 1, 10, 6, 3]`、不建 `[5, 5, 2, 3, 9, 9, 10, 2, 6, 9, 2, 2]`（nw=0，迴圈包 tqdm）。
- `print(test_tfm)`：`Resize(size=256, interpolation=bilinear, max_size=None, antialias=True)` / `CenterCrop(size=(224, 224))` / `ToTensor()` / `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`；train_tfm 在 CenterCrop 後多 `RandomHorizontalFlip(p=0.5)`。
- `log.txt` 只有 `log()` 寫的行（cfg、`device: cuda`、Train/Valid/Best、Finish training）；資料夾檔案數、`One ... sample`、torchsummary 表格只印在 stdout。
- test.py 的 `list(logits.argmax(dim=-1).squeeze().cpu().numpy())`：batch 只有 1 張時 `TypeError: iteration over a 0-d array`（實測 `torch.zeros(1,11)`）；2 張時 `[np.int64(0), np.int64(0)]`。
- **計時**：
  - 參照版（nw=0、預設 24 執行緒）8 分 19 秒，CPU 1567%。
  - 工具驗證那次（nw=0，與參照版逐位元一致；同時本機有其他輕量 CPU 工作）：每 epoch 訓練 34.8–46.1 s、驗證 10.3–19.3 s，總 523.4 s。
  - A10_ce（nw=16、persistent workers）：每 epoch 訓練 12.5–13.9 s、驗證 3.0–4.1 s，總 163.5 s。
  - 只讀資料一輪：nw=0 21.1 s、nw=16 2.3 s（Phase 0）。
- **只改 worker 數的雜訊**：A10_ce（nw=16）最佳驗證 0.51545（第 10 個 epoch），train.py（nw=0）0.50408（第 9 個）；第 5 個 epoch A10_ce 0.40000、train.py 0.44286。逐 epoch 驗證：0.35394 0.36793 0.43061 0.46764 0.40000 0.40554 0.48571 0.46793 0.48776 0.51545。
- 工具修正：A10_ce 寫完 jsonl 後行程卡在結束（persistent workers 收尾），手動結束；之後 `hw13_exp.py` 結尾加 `os._exit(0)`（只影響結束，不影響數字）。
- **不建老師、完整重跑**（scratchpad 複本，`train.py` 第 68 行照字面註解掉，其餘不變，nw=0，`OMP_NUM_THREADS=4`，與 A 組並行）：10:01:39–10:11:07。逐 epoch 驗證準確率 0.35423 0.38455 0.44023 0.43673 0.46589 0.47609 0.47464 0.46297 0.48105 **0.51720**（第 10 個 epoch 最佳；原版 0.50408 在第 9 個）。第 1 個 epoch 驗證 loss 1.85076（原版 1.86308）。（0.51720 = 1774/3430，和 A10_kd 的最佳值相同，巧合。）
- **梯度範數**（`docs/tools/hw13_gradnorm.py` → `hw13_gradnorm.json`；Simple 設定、nw=16，所以和 train.py 的亂數流不同）：1,550 步，clip 前的總範數 > 10 只有 **6 步**（0.39%）；平均 4.061、中位數 4.059、最大 16.875；前 10 步 2.788 1.89 2.206 2.293 2.183 1.67 1.811 1.829 1.522 1.628；每個 epoch 的最大值 4.634 5.717 10.108 11.067 8.042 13.159 14.741 16.875 9.891 12.628。
- A10_kd_T1_a0.5（nw=16）：最佳 0.51720（第 10 個 epoch），200.5 s（A10_ce 163.5 s，多 23%）。

## ch02 實測（參數怎麼數，2026-10-09；`docs/tools/hw13_count.py` → `hw13_count.txt`）
- torchsummary 1.5.1 原始碼（`torchsummary.py`）：對每個不是 `nn.Sequential`／`nn.ModuleList`、也不是 model 本身的模組註冊 forward hook（37–42 行）；hook 裡只數 `module.weight` 與 `module.bias` 的元素數（29–35 行）；trainable 只看 `weight.requires_grad`（32 行）；用 `torch.rand(2, *in_size)` 當輸入跑一次 forward（60、72 行）；Forward/backward pass size = 所有 hook 到的輸出元素數 × 2（梯度）× 4 bytes（101 行）；Params size = 參數 × 4 bytes。
- **summary 會用掉全域亂數**：`manual_seed(0)` 後「建學生」與「建學生 + summary」的 RNG 狀態不同（`torch.equal` False）。
- 邊角案例（numel vs torchsummary Total）：同一個 Conv2d(3,3,3) 呼叫兩次 + Linear(3,11)：128 vs **212**（重複算）；模型本身掛 `nn.Parameter(torch.ones(50_000))` + Linear(3,11)：50,044 vs **44**；定義了沒呼叫的 Linear(300,300)：90,344 vs **44**；學生第一個卷積 `requires_grad=False`：Total 87,907、Trainable 87,011、Non-trainable **896**；學生 `cnn[1]` 換 `BatchNorm2d(32, affine=False)`：87,843 vs 87,843（少 64 個 γβ，兩邊一致）。
- **計算量**（`torch.utils.flop_counter.FlopCounterMode`，一張 224×224，只數卷積與全連接；MACs = FLOPs/2）：
  - 範例學生 **859,378,124 MACs**（FLOPs 1,718,756,248）：cnn.0 42,581,376；**cnn.3 446,054,400（51.9%）**；cnn.7 214,990,848；cnn.11 155,750,400；fc 1,100。前兩層合計 488.6M（56.9%）。手算 cnn.3 = 32·32·9·220·220 = 446,054,400。
  - 老師 **1,813,566,976 MACs**：conv1 118,013,952；layer1 462,422,016；layer2/3/4 各 411,041,792；fc 5,632。學生／老師 = 47.4%（參數只有 1/127）。
  - dw 66,033,200（範例的 7.7%）；plain 84,333,928；mbv2 79,502,496。
- 範例學生每層輸出元素（一張圖）：cnn.0–2 各 1,577,088；cnn.3–5 各 1,548,800；pool 387,200；cnn.7–9 各 746,496；pool 186,624；cnn.11–13 各 270,400；pool 67,600；GAP 100。torchsummary 的 Forward/backward：sample 99.72 MB、dw 48.97、plain 14.44、mbv2 79.28、老師 62.79。
- `student_best.ckpt` 內容：weight 87,440 個（349,760 bytes）、bias 467（1,868）、running_mean 228（912）、running_var 228（912）、num_batches_tracked 4 個 int64（32）；檔案 363,645 bytes（張量 353,484，其餘約 10 KB 是 zip/pickle 格式）。轉 fp16 另存 186,257 bytes。
- 推論時間沒有在 ch02 量（A 組在跑，GPU 不乾淨）；移到第 6 章與剪枝的計時一起量。

## ch03 實測（老師，2026-10-09；`hw13_teacher.py` → `hw13_teacher.json`、`hw13_teacher_train.py`）
- **torchsummary 會改掉 BatchNorm 的統計量**：剛建好的模型是訓練模式；summary 的 forward（`torch.rand(2,3,224,224)`）讓每個 BN 用這筆隨機輸入更新 running mean/var（momentum 0.1）。老師 bn1：running_mean 最大變化 0.142、running_var 最大變化 1.282，`num_batches_tracked` 72,813 → 72,814。驗證準確率：新載入 **0.87143（2989/3430）**、CE 0.59343；訓練模式下 summary 之後 0.86327（2961）；Phase 0 那次 0.86093（2953，不同的隨機輸入）。先 `eval()` 再 summary：0.87143 不變。學生在 train.py 第 63 行也被 summary 改了 BN（running_mean 前三個 −0.04189 −0.01696 −0.024、running_var 0.90297…，tracked 1），但之後訓練會蓋掉，且參照版同樣如此。
- 老師 checkpoint 的 `num_batches_tracked` = 72,813（每個 BN 都一樣）：助教訓練了 72,813 步（batch 64 時約 470 個 epoch；batch 大小未知）。
- **驗證集**（test_tfm）：acc 0.87143；平均最大機率 0.9418；平均正確類機率 0.8545；最大機率 > 0.99 的佔 70.55%；logits 平均最大值 10.73。
  - 各類準確率（張數）：麵包 0.7762（362）、乳製品 0.7986（144）、甜點 0.798（500）、蛋 0.8654（327）、炸物 0.8466（326）、肉類 0.8864（449）、麵食 0.9592（147）、米飯 0.8958（96）、海鮮 0.9078（347）、湯 0.954（500）、蔬果 0.9397（232）。
  - 最常見混淆（真→預測，張數）：麵包→蛋 21、麵包→肉類 21、甜點→麵包 19、甜點→肉類 16、炸物→肉類 16、蛋→麵包 15、炸物→麵包 15、甜點→蛋 13。
  - 溫度（softmax(z/T) 的平均熵 nats／平均最大機率）：T=1 0.1598/0.9418；T=2 0.3998/0.8707；T=4 1.0101/0.6999；T=8 1.8349/0.4288；T=20 2.3175/0.1938；均勻分布 ln 11 = 2.3979。
  - 各類平均 soft label（T=1，%）：麵包 [75.68, 0.56, 3.62, 6.27, 2.84, 5.7, 0.4, 0.06, 2.69, 1.74, 0.46]；甜點 [3.73, 2.48, 77.3, …]；（完整見 json）。
- **訓練集**（test_tfm，老師在訓練時看過這些圖）：acc **0.99980（9864/9866）**，錯 2 張：2_111.jpg 甜點→海鮮、5_1311.jpg 肉類→海鮮。平均最大機率 **0.9986**；最大機率 > 0.99 的佔 **97.89%**；logits 平均最大值 13.73。
  - 溫度（熵／平均最大機率）：T=1 0.0062/0.9986；T=2 0.0843/0.9832；T=4 0.6283/0.8546；T=8 1.6736/0.5255；T=20 2.2992/0.2193。
  - 平均放在「其他 10 類」的機率：T=1 0.15%；T=2 1.69%；T=4 14.55%；T=8 47.46%。
  - 各類平均 soft label：T=1 正確類 99.54–100%（甜點最低 99.54）；T=4 正確類 77.12（甜點）–95.6（麵食），例：麵包 [83.75, 1.37, 1.87, 3.35, 1.7, 2.29, 0.79, 0.67, 1.59, 1.73, 0.88]；炸物 [4.12, 1.12, 1.62, 3.23, 80.1, 3.64, …]。
  - 例子 3_0.jpg（蛋）logits [-11.77, -6.79, 1.42, 11.39, -13.61, -1.2, -4.1, -7.8, -4.47, -5.23, -10.33]；T=1 蛋 99.99%；T=4 蛋 82.67、甜點 6.85、肉類 3.55、麵食 1.72、海鮮 1.57；T=8 蛋 44.21、甜點 12.73、肉類 9.17、麵食 6.38、海鮮 6.09。
  - 例子 7_0.jpg（米飯）logits 最大 21.1（米飯）、次大 3.27（蔬果）；T=1 100%；T=4 米飯 97.96、蔬果 1.13；T=8 米飯 72.97、蔬果 7.85、麵包 4.06。

## ch04 實測（知識蒸餾）
### KD loss 的性質（`docs/tools/hw13_kdcheck.py` → `hw13_kdcheck.txt`；Simple 學生 checkpoint 與老師，整個訓練集 9,866 張，test_tfm）
- 學生在訓練集 acc 0.50993、老師 0.99980。
- **方向**：`F.kl_div(log_softmax(s/T), softmax(t/T), 'batchmean')` = KL(老師‖學生)：T=1 1.436162（手算 KL(t‖s) 1.436162；反方向 KL(s‖t) **10.976600**）；T=4 1.390786（反方向 2.111768）。
- **reduction**：`'mean'` 0.13056（= batchmean / 11，並發出 UserWarning「'mean' divides the total loss by both the batch size and the support size ... will be changed to behave the same as 'batchmean' in the next major release」）；`'batchmean'` 1.436162；`'sum'` 14169.1758；`nn.KLDivLoss()` 預設 reduction 是 'mean' → 0.13056。
- **T=1 時 KL ≈ CE**：KL(老師‖學生) 1.43616 vs CE（硬標籤）1.44061；對學生 logits 的每張圖梯度範數 0.75147 vs 0.75272。
- **T² 的作用**（KL 項對學生 logits 的每張圖梯度平均 L2 範數；不乘 → 乘 T²）：T=1 0.75147 → 0.75147；T=2 0.41101 → 1.64402；T=4 0.18699 → **2.99186**；T=8 0.05463 → **3.49654**；T=20 0.00669 → 2.67601。KL 值本身：T=1 1.43616、2 1.60408、4 1.39079、8 0.59988、20 0.08267。
- `loss_fn_kd` 分解：T=1、α=0.5：1.43838 = 0.5·1·1.43616 + 0.5·1.44061；T=4、α=0.5：11.84659 = 0.5·16·1.39079 + 0.5·1.44061。

## ch06 實測（剪枝，準確率部分；`docs/tools/hw13_prune.py` → `hw13_prune.json`，2026-10-09）
驗證集準確率；三種剪法都只剪 Conv2d 的 weight（範例 notebook 的範圍），不微調。學生 = Simple 的 student_best.ckpt。
| 比例 | 老師 逐層 L1 | 老師 全域 L1 | 老師 逐層通道(L2) | 學生 逐層 L1 | 學生 全域 L1 | 學生 逐層通道 |
|---|---|---|---|---|---|---|
| 0 | 0.87143 | 0.87143 | 0.87143 | 0.50408 | 0.50408 | 0.50408 |
| 0.1 | 0.86443 | 0.86822 | 0.22449 | 0.46356 | 0.50525 | 0.30758 |
| 0.2 | 0.85598 | 0.86327 | 0.14402 | 0.39708 | 0.50700 | 0.14752 |
| 0.3 | 0.82711 | 0.85714 | 0.14577 | 0.24840 | 0.51720 | 0.14577 |
| 0.4 | 0.77551 | 0.82187 | 0.14577 | 0.16531 | 0.47843 | 0.14577 |
| 0.5 | 0.52216 | 0.74927 | 0.14577 | 0.18426 | 0.39796 | 0.14577 |
| 0.6 | 0.16064 | 0.64257 | 0.13090 | 0.15248 | 0.32128 | 0.14577 |
| 0.7 | 0.11516 | 0.17376 | 0.13090 | 0.14606 | 0.17551 | 0.14577 |
| 0.8 | 0.10991 | 0.12362 | 0.10117 | 0.14577 | 0.14723 | 0.14577 |
| 0.9 | 0.10233 | 0.14665 | 0.10292 | 0.14577 | 0.14577 | 0.14577 |
| 0.95 | 0.10117 | 0.10117 | 0.10117 | 0.14577 | 0.14577 | 0.14577 |
- 0.14577 = 500/3430（全部猜成甜點或湯，兩類各 500 張）；0.10117 = 347/3430（海鮮）。
- 每次剪完的零比例（卷積權重中 == 0 的比例）等於設定比例（json 的 zeros 欄）。
- **剪枝的機制**（老師、逐層 L1 0.5）：`conv1` 的參數只剩 `weight_orig`，buffer 多了 `weight_mask`，`conv1.weight` 不再是 Parameter（forward pre-hook 1 個，每次 forward 前算 `weight_orig * weight_mask`）。numel 11,182,155、torchsummary Total 11,182,155（都不變）。state_dict 142 個 key、存檔 **89,481,855 bytes**（原 44,806,605 的 2 倍）。`prune.remove` 之後 122 個 key、44,807,627 bytes，準確率 0.52216，零比例 0.5。把 4 維權重轉成 `to_sparse()`（COO）另存：**201,150,399 bytes**（4.5 倍）。

## ch07 實測（量化，準確率與檔案；`docs/tools/hw13_quant.py` → `hw13_quant.json`，2026-10-09）
CPU 量測用 4 執行緒；靜態 int8 用 FX graph mode（`prepare_fx`／`convert_fx`，x86 後端，`get_default_qconfig_mapping('x86')`），校正資料 = 訓練集每 10 張取 1（987 張，test_tfm）。weight-only = 每個輸出通道對稱、四捨五入到 2^(b−1)−1 級（模擬，權重仍以 float 存）。檔案 = `torch.save(state_dict)`。
| | 老師 acc | 老師檔案 | 學生 acc | 學生檔案 |
|---|---|---|---|---|
| fp32（GPU） | 0.87143 | 44,800,651 | 0.50408 | 361,765 |
| fp32（CPU） | 0.87143 | | 0.50408 | |
| fp16（GPU，`.half()`） | 0.87143 | 22,414,667 | 0.50408 | 184,613 |
| bf16（GPU） | 0.87026 | | 0.50496 | |
| dynamic int8（`quantize_dynamic({nn.Linear})`，CPU） | 0.87143 | 44,784,581 | 0.50437 | 359,325 |
| **static int8**（FX，CPU） | **0.87201** | **11,308,361** | 0.49563 | 101,085 |
| weight-only int8（GPU） | 0.87085 | | 0.49942 | |
| weight-only int6 | 0.86647 | | 0.50058 | |
| weight-only int4 | 0.71458 | | 0.42420 | |
| weight-only int3 | 0.33382 | | 0.34198 | |
| weight-only int2 | 0.10146 | | 0.12128 | |
- 警告（torch 2.11）：「torch.ao.quantization is deprecated and will be removed in 2.10. For migrations of users: 1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please …」（實際上 2.11 仍可用）；以及 observer 的「reduce_range will be deprecated」。`torch.ao.quantization.quantize_pt2e` 不存在，torchao 沒裝。
- dynamic int8 只換掉 `nn.Linear`（老師的 fc 512×11），所以檔案只小 16 KB。

### A 組（`hw13_run_grid.sh`，16 workers，`docs/tools/hw13_runs.jsonl`，2026-10-09 09:52–12:33）
| run | 最佳驗證 acc | 最佳 epoch | 最後 5 個 epoch 平均 | 最後 train acc | 最後 valid CE | 平均每 epoch 訓練秒數 |
|---|---|---|---|---|---|---|
| A10_ce | 0.51545 | 10 | 0.4725 | 0.5221 | 1.43347 | 13.0 |
| A10_kd_T1_a0.5 | 0.51720 | 10 | 0.4727 | 0.5224 | 1.43112 | 15.7 |
| A50_ce | 0.64286 | 50 | 0.5983 | 0.69826 | 1.08615 | 17.7（與不建老師重跑、ch02/03 的量測並行） |
| A50_kd_T1_a0.5 | 0.63732 | 50 | 0.5974 | 0.69785 | 1.10063 | 17.5 |
| A50_kd_T2_a0.5 | **0.64840** | 50 | 0.5985 | 0.68113 | 1.18314 | 16.6 |
| A50_kd_T4_a0.5 | 0.63032 | 50 | 0.6029 | 0.66227 | 1.60786 | 15.8 |
| A50_kd_T8_a0.5 | 0.61516 | 50 | 0.5801 | 0.64413 | 1.77971 | 15.7 |
| A50_kd_T1_a0.9 | 0.64023 | 50 | 0.5994 | 0.69886 | 1.09612 | 15.7 |
| A50_kd_T2_a0.9 | 0.63878 | 50 | 0.6048 | 0.67697 | 1.31925 | 15.8 |
| A50_kd_T4_a0.9 | 0.61691 | 45 | 0.5966 | 0.66065 | 1.86218 | 15.8 |
| A50_kd_T8_a0.9 | 0.61108 | 45 | 0.5490 | 0.62913 | 2.07965 | 15.7 |
- 總時間：A50_ce 1066.0 s；KD 各組 986–1092 s。A10：CE 163.5 s、KD 200.5 s。
- B 組用 A 組最佳的 T=2、α=0.5（差距在雜訊內，照規格取最佳者），12:35 開始。
- **各類驗證準確率與和老師的一致率**（`hw13_kd_perclass.py` → `hw13_kd_perclass.jsonl`，最佳 epoch 的 logits）：
  - A50_ce：一致率 0.6464；麵包 0.304、乳製品 0.486、甜點 0.512、蛋 0.630、炸物 0.531、肉類 0.840、麵食 0.714、米飯 0.604、海鮮 0.718、湯 0.826、蔬果 0.810；預測佔比 甜點 12.0%、米飯 1.9%。
  - A50_kd_T1_a0.5：0.6411；麵包 0.296 … 米飯 0.604。
  - A50_kd_T2_a0.5：0.6481；麵包 0.265、乳製品 0.500、甜點 0.660、蛋 0.581、炸物 0.586、肉類 0.768、麵食 0.673、米飯 0.583、海鮮 0.703、湯 0.798、蔬果 0.871；甜點預測佔比 17.4%。
  - A50_kd_T4_a0.5：0.6259；乳製品 0.375、麵食 0.456、米飯 0.469；甜點佔比 19.7%。
  - A50_kd_T8_a0.5：0.6155；乳製品 0.326、甜點 0.702、米飯 **0.281**；甜點佔比 **21.6%**、米飯 0.9%。
  - A50_kd_T2_a0.9：0.6312；炸物 0.721、蔬果 0.922、麵包 0.238。
  - 驗證集真實佔比：甜點 500/3430 = 14.6%、米飯 96/3430 = 2.8%。
- **假設被推翻**：「高 T 的 soft label 把機率推給大類別」不成立（`hw13_soft_mass.py`，訓練集平均 soft target %）：米飯 T=1 2.84、T=2 2.85、T=4 3.15、T=8 **4.67**（硬標籤 2.84）；乳製品 4.36 → 6.18；甜點 15.14 → 14.78 → 12.95 → **10.63**（硬標籤 15.20）。T 越高，目標分布越偏向小類別、越不偏向甜點，學生卻反過來。原因本書沒有查明。

## ch05 實測（架構設計；B 組，`hw13_run_grid_B.sh`，2026-10-09 12:34–15:23）
200 epoch、`--aug hw03`、16 workers、固定學習率 3e-4（沒有排程）、KD 用 A 組最佳 T=2 α=0.5。
| run | 參數 | MACs | 最佳驗證 acc | 最佳 epoch | 最後 10 個 epoch 平均 | 最後 10 個 epoch 訓練 acc 平均 | 第 50／100 個 epoch | 最低驗證 CE | 每 epoch 訓練秒數 | 總秒數 |
|---|---|---|---|---|---|---|---|---|---|---|
| B200_dw_kd | 99,811 | 66,033,200 | 0.78513 | 198 | 0.7746 | 0.8873 | 0.68367／0.75364 | 0.88662 | 7.7 | 2216.9 |
| B200_dw_ce | 99,811 | | 0.77201 | 195 | 0.7654 | 0.8871 | 0.66239／0.74052 | 0.77789 | 5.3 | 1602.9 |
| B200_mbv2_kd | 96,651 | 79,502,496 | 0.80000 | 195 | 0.7902 | 0.8740 | 0.67493／0.74490 | 0.82120 | 9.4 | 2562.1 |
| B200_mbv2_ce | 96,651 | | **0.80437** | 188 | 0.7868 | 0.8793 | 0.67901／0.75306 | 0.69044 | 6.8 | 1916.4 |
| B200_plain_kd | 99,355 | 84,333,928 | 0.79125 | 187 | 0.7699 | 0.8511 | 0.65394／0.76793 | 0.83840 | 6.0 | 1833.6 |
- 同參數對照：plain（一般卷積）0.79125 > dw（depthwise）0.78513（都是 KD）。
- KD vs CE：dw +1.31（0.78513 vs 0.77201），mbv2 −0.44（0.80000 vs 0.80437）。
- 使用者 15:25 核可加跑 B2（範例學生同訓練方式 KD／CE），排在 C 組與計時之後。

## ch06 實測（續）：C 組微調與計時
### C 組（使用者核可加跑；`hw13_run_grid_C.sh`，20 epoch、CE、範例增強、16 workers，15:23–15:38）
- 實體拿掉通道（`hw13_shrink.py`：每個卷積依權重 L1 範數留下前 keep 比例的輸出通道，BN 與下一層跟著裁）：keep75 → 通道 [24, 24, 48, 75]、49,949 參數、491,384,409 MACs、未微調 0.17085；keep50 → [16, 16, 32, 50]、22,647 參數、225,490,150 MACs、未微調 0.14577。
| run | 最佳驗證 acc | 最佳 epoch | 總秒數 |
|---|---|---|---|
| C20_ft_keep75（Simple 權重剪完微調） | **0.57843** | 20 | 264.8 |
| C20_ft_keep50 | **0.54373** | 19 | 205.7 |
| C20_scratch_keep75（同寬度從頭訓練） | 0.53324 | 20 | 262.7 |
| C20_scratch_keep50 | 0.48601 | 15 | 205.4 |
- 對照：Simple 原模型 10 epoch 0.50408（87,907 參數）；A50_ce 50 epoch 0.64286。微調的起點是 Simple 的 student_best（已訓練 9 epoch），所以 ft 共經歷 29 epoch 的訓練。

### 計時（`hw13_speed.py` → `hw13_speed.json`，15:39:21–15:40:23，前後 `nvidia-smi --query-compute-apps` 皆空；RTX PRO 4000 Blackwell；CPU 4 執行緒；GPU 100 次、CPU 20 次取中位數，ms）
| 模型 | 參數 | MACs | GPU b64 fp32 | GPU b1 fp32 | GPU b64 fp16 | CPU b1 fp32 | CPU b1 int8 |
|---|---|---|---|---|---|---|---|
| 老師 | 11,182,155 | 1,813,566,976 | 16.531 | 1.115 | 7.443 | 9.386 | 3.742 |
| 老師 剪 50%（mask） | 11,182,155 | 1,813,566,976 | 16.761 | 1.468 | — | 12.751 | |
| 老師 剪 50%（prune.remove 後的稠密權重） | 11,182,155 | 1,813,566,976 | 16.568 | 1.101 | 7.429 | 9.615 | |
| 範例學生 | 87,907 | 859,378,124 | **22.369** | 0.303 | 15.728 | 5.698 | 2.621 |
| dw | 99,811 | 66,033,200 | 6.373 | 0.731 | 3.657 | 1.633 | |
| plain | 99,355 | 84,333,928 | **2.901** | 0.464 | 1.430 | 0.931 | |
| mbv2 | 96,651 | 79,502,496 | 8.920 | 0.888 | 4.569 | 2.038 | |
| 範例學生 keep75（實體裁） | 49,949 | 491,384,409 | 17.278 | 0.302 | 11.954 | 3.778 | |
| 範例學生 keep50（實體裁） | 22,647 | 225,490,150 | 10.800 | 0.263 | 6.695 | 2.249 | |
- （計時用隨機權重的 dw/plain/mbv2；權重值不影響時間。範例學生與老師用訓練好的權重。）
- 剪枝後的模組不能 `copy.deepcopy`：`RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.`（`weight` 是 `weight_orig * weight_mask` 算出來的非 leaf 張量。）

### 各學生的層別分解（`docs/tools/hw13_arch.py` → `hw13_arch.txt`；「每搬一個數做幾次乘加」= MACs ÷（輸入 + 輸出 activation 元素 + 權重），一張 224×224）
- sample：4 個 k×k 卷積，參數 86,340（98.2%），MACs 859,377,024，每搬一個數 131.6 次。
- dw：stem 3×3（648 參數、8,128,512 MACs、12.3%、18.0）；9 個 depthwise（7,600 參數 7.6%、7,225,344 MACs 10.9%、**3.4**）；9 個 pointwise（86,640 參數 86.8%、50,677,760 MACs 76.7%、26.3）。
- plain：10 個 3×3（98,064 參數 98.7%、84,333,312 MACs、**55.9**）。
- mbv2：stem（432、5,419,008 6.8%、15.4）；17 個 pointwise（81,664 84.5%、65,529,856 82.4%、16.5）；8 個 depthwise（8,208 8.5%、8,551,872 10.8%、**3.0**）。

### B2（使用者核可加跑；`hw13_run_grid_B2.sh`，15:40–17:41）範例學生，同 B 組訓練方式
| run | 最佳驗證 acc | 最佳 epoch | 最後 10 個 epoch 平均 | 最後 10 個 epoch 訓練 acc | 第 50／100 個 epoch | 每 epoch 訓練秒數 | 總秒數 |
|---|---|---|---|---|---|---|---|
| B200_sample_kd | 0.73440 | 194 | 0.6999 | 0.7423 | 0.62420／0.68630 | 15.8 | 3962.4 |
| B200_sample_ce | 0.73411 | 194 | 0.6962 | 0.7495 | 0.61662／0.66997 | 13.1 | 3251.8 |
- 拆解（KD、最佳驗證 acc）：範例學生 50 epoch 範例增強 0.64840 → 200 epoch + HW03 增強 0.73440（訓練方式 +8.6 點）→ 換架構 dw 0.78513／plain 0.79125／mbv2 0.80000（架構 +5.1～+6.6 點）。CE 版：0.64286 → 0.73411 → dw 0.77201／mbv2 0.80437。
- 訓練時間：範例學生每 epoch 13.1–15.8 s，新學生 5.3–9.4 s。
- **老師在增強過的訓練圖上**（`hw13_teacher_aug.py`，torch.manual_seed(0)，一輪）：範例 train_tfm（翻轉）acc 0.99959、平均最大機率 0.9985、>0.99 佔 97.66%、其他類機率 T=1 0.17%／T=2 1.73%；HW03 增強 acc **0.99179**、平均最大機率 0.9863、>0.99 佔 87.39%、其他類機率 T=1 1.74%／T=2 **6.39%**。
