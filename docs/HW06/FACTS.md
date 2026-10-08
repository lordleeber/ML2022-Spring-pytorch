# HW06 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW06/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **狀態（2026-10-07 19:12）**：實驗全部跑完並評估（baseline、grid 四組、StyleGAN2）。大綱還沒寫。
>
> **重現方法**（需要 GPU 與 `HW06/faces/`；評估另需 .venv 外的 pylib，見「評估工具」）：
> - `cd HW06 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw06_exp.py --name X --out DIR [選項]`：訓練實驗，預設參數就是 train.py。
> - `PYTHONPATH=.:<pylib> ../.venv/bin/python ../docs/tools/hw06_eval.py real|gen ...`：FID／AFD。
> - `bash docs/tools/hw06_run_grid.sh <out_dir> <pylib> <cascade.xml> [name ...]`（repo 根目錄）：一組一組訓練＋評估；結果追加到 `docs/tools/hw06_runs.jsonl`、`hw06_eval.jsonl`，逐步紀錄在 `docs/tools/hw06_logs/`。

原始碼根目錄：`HW06/`（程式最後一次改動 `381aad0`，2026-10-03，只改 test.py）。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW06/HW06.ipynb`。投影片：`HW06/Machine Learning HW6.pdf`（26 頁）。投影片重點、原版差異、疑點清單見 [PLAN.md](PLAN.md)。

## 環境（同 HW04，2026-10-07 沿用）
- Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、scipy 1.18.1、numpy 2.5.3。GPU NVIDIA RTX PRO 4000 Blackwell（24,467 MiB），WSL2，24 核心，RAM 47 GB。
- .venv 外（scratchpad 的 pylib，session 結束會消失，要重裝）：pytorch-fid 0.3.0、opencv-python-headless 4.14.0.94（5.x 沒有 `cv2.CascadeClassifier`，要裝 `"opencv-python-headless<5"`；`uv pip install --no-cache --target <pylib> --no-deps ...`；不加 `--no-cache` 時 uv 卡在 cache lock）。
- Inception 權重：`~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth`（95,628,359 bytes，sha256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`）。
- AFD 偵測器：nagadomi/lbpcascade_animeface 的 `lbpcascade_animeface.xml`（sha256 `9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30`），參數照它 README 的範例（equalizeHist、scaleFactor 1.1、minNeighbors 5、minSize 24×24）。
- StyleGAN2：lucidrains `stylegan2-pytorch` 1.9.0（`--no-deps`，另裝 einops 0.8.2、kornia 0.8.3、kornia-rs 0.2.0、contrastive-learner 0.1.1、vector-quantize-pytorch 0.1.0、fire 0.7.1、retry 0.9.2、decorator、py、termcolor）。它在模組最上面 `import aim`（只有 `--log` 才用），裝了一個空的 `aim/__init__.py` 讓它載入。

## 資料
- `HW06/faces/`：71,314 張，檔名 `0.jpg`–`71313.jpg`，96×96 RGB JPEG（抽查前 2,000 張全是），560 MB。被 `HW06/.gitignore` 排除。
- `utils.get_dataset`：`glob` 不排序（第一個是 `faces/56141.jpg`，檔案系統順序）→ `ToPILImage` → `Resize((64, 64))` → `ToTensor` → `Normalize(0.5, 0.5)`，值域 [-1, 1]。
- 一個 epoch：71,314 / 64 = 1,114.3 → **1,115 步**（最後一批 18 張）；100 epoch = 111,500 步。

## Baseline（照跑 `train.py`，2026-10-07 11:00:28–11:29:29）
- 開跑前後 `nvidia-smi --query-compute-apps` 都是空的。
- `/usr/bin/time -v`：**Elapsed 29:01.38**；User 3241.57 s、System 513.83 s、CPU 215%；最大 RSS 1,699,640 KB。進度條約 59–65 it/s，每個 epoch 約 17–18 秒。
- 輸出：`logs/2026-10-07_11-00-32_GAN/Epoch_001.jpg`–`Epoch_100.jpg`（每張 10×10 格，`z_samples` 固定的 100 個 z），`checkpoints/2026-10-07_11-00-32_GAN/G_{0,4,9,...,99}.pth`、`D_*.pth`（21 對）。印出的是 logging 的 `INFO: Save some samples to ...`，沒有其他警告。
- 進度條上的 loss（每 10 步更新一次）：第 1 個 epoch 中段 loss_D 0.32–0.56、loss_G 4.8–11.2；第 26 epoch loss_D 0.0388、loss_G 8.62；第 50 epoch loss_D 0.032、loss_G 16.5；第 72 epoch loss_D 0.00204、loss_G 13.4（D 遠強過 G）。
- **可重現**：2026-10-03 那次（`checkpoints/2026-10-03_01-25-33_GAN`，01:25:33–01:54:51，約 29 分）和今天這次的 G_0、D_0、G_99、D_99 全部 `torch.equal`，`Epoch_100.jpg` 位元組相同。→ 同一台機器、2 個 worker 下逐位元可重現。

## 實驗工具驗證
- `hw06_exp.py --n_epoch 1`：G_0.pth、D_0.pth 與 baseline 的 `torch.equal`（key 順序也相同），`Epoch_001.jpg` 位元組相同；1 個 epoch 17.6 s，最後 loss_D 0.0908、loss_G 5.0907。

## 評估：真實資料（`hw06_eval.py real`，4 分 14 秒）
- 統計量用全部 71,314 張（sorted 檔名）。real64 = `get_dataset` 的轉換再還原到 [0, 1]；real96 = 原始檔。
- **真實圖的 AFD**：real64 **0.5375**（38,331／71,314），real96 **0.4890**（34,876）。→ 這個偵測器連真實圖都只認出約一半，投影片的 AFD 基準線（Medium 0.4、Strong 0.5、Boss 0.6）不能拿來對照；本書的 AFD 只比相對高低。
- 本工具的 FID 是一般尺度（幾十到幾百），投影片基準線是上萬（Simple ≤ 30000），兩者不能對照。

## 評估：baseline 每個存檔（`hw06_eval.py gen`，n = 1000、z 種子 0，21 個 checkpoint 共 4 分 7 秒）
逐行紀錄：`docs/tools/hw06_baseline_eval.jsonl`（`checkpoints/2026-10-07_11-00-32_GAN`）。

| 存檔 | epoch | FID64 | FID96 | AFD |
|---|---|---|---|---|
| G_0 | 1 | 363.23 | 369.75 | 0.007 |
| G_4 | 5 | 207.56 | 225.69 | 0.276 |
| G_9 | 10 | 119.29 | 147.93 | 0.416 |
| G_14 | 15 | **104.95** | **134.21** | 0.366 |
| G_19 | 20 | 347.12 | 354.81 | 0.0 |
| G_24 | 25 | 276.63 | 289.37 | 0.0 |
| G_29 | 30 | 284.91 | 300.85 | 0.058 |
| G_34 | 35 | 278.83 | 295.07 | 0.007 |
| G_39 | 40 | 226.54 | 240.02 | 0.421 |
| G_44 | 45 | 178.26 | 194.34 | 0.596 |
| G_49 | 50 | 205.03 | 221.43 | 0.298 |
| G_54 | 55 | 218.04 | 230.28 | 0.0 |
| G_59 | 60 | 199.46 | 223.97 | 0.29 |
| G_64 | 65 | 230.37 | 251.93 | 0.358 |
| G_69 | 70 | 241.82 | 257.36 | 0.378 |
| G_74 | 75 | 246.43 | 265.91 | **0.728** |
| G_79 | 80 | 218.88 | 242.63 | 0.21 |
| G_84 | 85 | 185.63 | 203.13 | 0.445 |
| G_89 | 90 | 211.6 | 224.95 | 0.0 |
| G_94 | 95 | 216.94 | 228.7 | 0.208 |
| G_99 | 100 | 211.93 | 233.24 | 0.666 |

- **最佳 FID 在第 15 個 epoch**；第 20 個 epoch 崩到 347（接近第 1 個 epoch 的 363）。test.py 取的是「最新」的 G_99（FID64 211.93），是第 15 個 epoch 的兩倍。
- 目視 `logs/2026-10-07_11-00-32_GAN/Epoch_XXX.jpg`（同一組 z_samples）：epoch 15 一百張各不相同；**epoch 20 一百張目視幾乎相同（mode collapse），是同一張花掉的圖的小變化**。量化（G_14／G_19／G_99，z 種子 0 的 1000 張，[0,1] 乘 255）：每像素標準差平均 55.03／**16.07**／46.50；與平均圖的平均絕對差 46.17／12.81／40.89；前 200 張兩兩 L2 距離中位數 33.63／10.20／32.13。→ 不是逐像素全同，但多樣性只剩約 3 成；epoch 45 只剩少數幾種臉；epoch 100 多樣性部分恢復，但較糊。
- AFD 和 FID 不同步：AFD 0.728 的 G_74 FID 是 246；AFD 0 的存檔有 G_19、G_24、G_54、G_89。崩成同一張臉時，AFD 只取決於那一張被不被偵測到（全有或全無）。
- 評估速度：每個存檔約 12 秒，大部分是 CPU 上 2048×2048 的 `sqrtm`。
- pytorch-fid 0.3.0 的 `calculate_frechet_distance` 在 scipy 1.18 會 `TypeError: sqrtm() got an unexpected keyword argument 'disp'`；hw06_eval.py 複製了同一段算式（只拿掉 `disp`）。

## 實驗工具驗證（100 epoch，grid 的 gan 組，11:44:10–12:13:04）
- `hw06_exp.py`（預設參數）100 epoch 的 21 對 G／D 存檔，和 train.py 的 `checkpoints/2026-10-07_11-00-32_GAN` 全部 `torch.equal`（42/42），`Epoch_100.jpg` 位元組相同。工具計時 1,732.3 s（含 log 紀錄），111,500 步。
- **評估的可重現度**：同一批存檔評估兩次，AFD 完全相同，FID 在小數第 3 位左右不同（最大差 0.006，G_64：230.3678 vs 230.3617）——Inception 在 GPU 上的浮點不確定性。**教材的 FID 一律寫到小數 1 位**。

## 改良實驗：grid（`hw06_run_grid.sh`，2026-10-07 11:44–14:02，每組 100 epoch = 111,500 次 D 更新）
設定（全部從同一組初始權重出發；變體在 TrainerGAN() 建完模型後才換，不消耗亂數）：
- **gan**：train.py 原樣（BCE + Sigmoid、Adam 1e-4 (0.5, 0.999)、n_critic 1）。
- **wgan**：拿掉 Sigmoid、WGAN loss、RMSprop lr 5e-5、clip ±0.01、n_critic 5（WGAN 論文的設定）。
- **wgangp**：拿掉 Sigmoid、D 的 BatchNorm2d 換 InstanceNorm2d(affine=True)、Adam lr 1e-4 (0, 0.9)、λ = 10、n_critic 5（WGAN-GP 論文的設定）。
- **wgan_sigmoid**：WGAN loss 但**保留 Sigmoid、不 clip**、Adam 照原樣、n_critic 1（使用者筆記「只換 loss 不能用」的情況）。

| 組 | 訓練秒數 | 每 epoch | G 更新次數 | 最後 loss_D | 最後 loss_G |
|---|---|---|---|---|---|
| gan | 1,732.3 | 約 17.3 s | 111,500 | 0.0012 | 12.7586 |
| wgan | 1,180.8 | 約 11.9 s | 22,300 | -0.4406 | 0.2046 |
| wgangp | 2,647.5 | 約 26.5 s | 22,300 | -5.7069 | 188.288 |
| wgan_sigmoid | 1,734.1 | 約 17.3 s | 111,500 | 0.0 | -1.0 |

（每組另加約 4 分鐘評估 21 個存檔。逐行：`docs/tools/hw06_runs.jsonl`、`hw06_eval.jsonl`；逐步 log：`docs/tools/hw06_logs/<組>.jsonl`，每 10 步一行，每 100 步多一個 `gn`。）

FID64（n = 1000、z 種子 0；FID96 與 AFD 見 jsonl）：

| epoch | gan | wgan | wgangp | wgan_sigmoid |
|---|---|---|---|---|
| 1 | 363.2 | 382.6 | 389.6 | 307.8 |
| 5 | 207.6 | 321.2 | 295.8 | 337.2 |
| 10 | 119.3 | 234.8 | 240.7 | 337.3 |
| 15 | **105.0** | 187.0 | 202.1 | 336.8 |
| 20 | 347.1 | 155.5 | 180.7 | 337.3 |
| 25 | 276.6 | 136.7 | 162.4 | 337.2 |
| 50 | 205.0 | 111.4 | 119.4 | 337.0 |
| 75 | 246.4 | 96.5 | 102.0 | 336.3 |
| 90 | 211.6 | 91.6 | 96.1 | 337.4 |
| 95 | 216.9 | 90.8 | **94.1** | 337.1 |
| 100 | 211.9 | **90.3** | 96.7 | 337.4 |

- **gan**：存檔和 baseline 逐位元相同，評估同上表（第 15 epoch 最佳，之後崩、震盪）。
- **wgan**：FID64 幾乎單調下降，最佳是最後一個（90.3，FID96 120.7）；AFD 0.36–0.40（第 100 epoch 0.364）。
- **wgangp**：同樣穩定下降，最佳第 95 epoch 94.1（FID96 123.0），第 100 epoch 96.7；AFD 0.30–0.39（第 100 epoch 0.355）。最佳值比 wgan 稍差，每 epoch 慢 2.2 倍。
- **wgan_sigmoid**：第 1 epoch 之後 FID64 一直 336–338、AFD 全部 0；最後 loss_D 0.0、loss_G -1.0 = D 對所有輸入都輸出 1（Sigmoid 飽和），梯度全部是 0（見下）。
- 三種 GAN 的 AFD（本書偵測器）都在 0.3–0.4；gan 單一存檔有過 0.728（G_74），但 FID 246。

### 判別器各層梯度範數（報告第 2 題；每 100 步記一次 5 個 conv 的 weight.grad L2 範數，第 1 層 = 輸入端）
中位數，取第 91–100 epoch（約 112 個紀錄點）：

| 組 | conv1 | conv2 | conv3 | conv4 | conv5（輸出） |
|---|---|---|---|---|---|
| gan | 2.41 | 0.457 | 0.243 | 0.17 | 0.237 |
| wgan（clipping） | **49.5** | 18.2 | 3.68 | **0.316** | 0.614 |
| wgangp（GP） | 32.7 | 13.8 | 12.9 | 13.9 | 5.21 |
| wgan_sigmoid | 0 | 0 | 0 | 0 | 0 |

第 1 個 epoch 的中位數：gan 3.6／2.49／2.13／1.77／2.31；wgan 9.12／2.69／0.73／0.136／1.34；wgangp 68.6／53.5／24.4／12.8／7.55；wgan_sigmoid 0.00463／0.00306／0.00226／0.00184／0.00173。第 0 步：gan 11.6／11.6／9.67／8.87／8.68；wgan 41.8／40.9／33.6／30.7／29.3；wgangp 2.7e3／1.3e3／720／635／922。
- **clipping**：從 conv4 往輸入端每層放大好幾倍（0.316 → 3.68 → 18.2 → 49.5，conv1／conv4 ≈ 157 倍）——WGAN-GP 論文 Fig. 1(b) 說的梯度爆炸／消失。**GP**：conv1–conv4 都在 13–33 之間，差不到 2.5 倍。
- 注意：這裡的 D 有 BatchNorm（wgan）／InstanceNorm（wgangp），和論文的純 MLP 不同；clip 也套在 BN 的 weight／bias 上。

## StyleGAN2（`hw06_sg2_train.sh`，lucidrains stylegan2-pytorch 1.9.0，2026-10-07 14:42:52–19:01:30）
- 設定：64×64、batch 32、`gradient_accumulate_every 1`、50,000 步、每 2,500 步存檔（`model_0.pt`–`model_20.pt`，各 197,352,625 bytes）與樣本圖；其餘是套件預設（network_capacity 16、lr 2e-4、ttur_mult 1.5、mixed_prob 0.9、seed 42、`trunc_psi` 0.75）。從零訓練，沒有預訓練權重。
- 計時：開跑前 `nvidia-smi --query-compute-apps` 空；`/usr/bin/time -v` **Elapsed 4:18:37**、CPU 101%、最大 RSS 2,110,352 KB；進度條穩定在 3.22–3.26 it/s（前 200 步的試跑 3.16 it/s）。50,000 × 32 = 1,600,000 張 ≈ **22.4 個 epoch**（DCGAN 100 epoch 是 7,131,400 張送進 D）。GPU 記憶體 7.4 GB。
- 參數量：mapping 網路 S 2,101,248、G 11,216,556、D 22,679,184（另有 EMA 的 SE、GE，以及和 D 同大小的 D_aug）。DCGAN：G 5,142,080、D 2,766,529。
- loss 紀錄：`docs/tools/hw06_logs/sg2.txt`（1,000 行，每 50 步一行 `G | D | GP | PL`）。前 3 行 `G: 5.45 | D: 0.37 | GP: 11.18` …；最後 `G: 0.67 | D: 1.99 | GP: 0.01 | PL: 0.41`。過程中沒有 NaN。
- 評估（`hw06_eval.py sg2`，EMA 的 SE／GE、套件自己的 `generate_truncated`、全域亂數種子 0、n = 1000；逐行 `docs/tools/hw06_sg2_eval.jsonl`）：

| 步數 | FID64 ψ0.75 | FID96 ψ0.75 | AFD ψ0.75 | FID64 ψ1.0 | FID96 ψ1.0 | AFD ψ1.0 |
|---|---|---|---|---|---|---|
| 0 | 302.4 | 323.1 | 0.000 | 291.6 | 312.8 | 0.000 |
| 2,500 | 268.5 | 292.1 | 0.006 | 262.0 | 284.2 | 0.014 |
| 5,000 | 203.6 | 221.4 | 0.297 | 192.0 | 211.2 | 0.221 |
| 7,500 | 216.9 | 234.7 | 0.361 | 200.0 | 219.4 | 0.269 |
| 10,000 | 130.1 | 154.3 | 0.582 | 125.5 | 149.4 | 0.442 |
| 12,500 | 128.6 | 156.3 | 0.601 | 122.9 | 150.1 | 0.481 |
| 15,000 | 101.8 | 131.9 | 0.603 | 101.6 | 129.6 | 0.428 |
| 17,500 | 125.6 | 149.8 | 0.575 | 121.2 | 145.5 | 0.450 |
| 20,000 | 106.9 | 134.2 | 0.477 | 98.7 | 124.0 | 0.406 |
| 22,500 | 100.4 | 126.5 | 0.503 | 96.3 | 122.1 | 0.420 |
| 25,000 | 89.0 | 116.1 | 0.541 | 83.1 | 111.2 | 0.439 |
| 27,500 | 89.0 | 117.2 | 0.567 | 81.2 | 109.3 | 0.465 |
| 30,000 | 87.6 | 115.7 | 0.563 | 78.6 | 106.2 | 0.456 |
| 32,500 | 85.3 | 112.3 | 0.570 | 76.1 | 102.9 | 0.479 |
| 35,000 | 86.1 | 112.6 | 0.580 | 74.2 | 101.1 | 0.471 |
| 37,500 | 85.0 | 111.7 | 0.585 | 75.0 | 102.0 | 0.465 |
| 40,000 | 82.5 | 108.6 | 0.569 | 73.9 | 101.1 | 0.456 |
| 42,500 | 80.7 | 106.5 | 0.582 | 70.7 | 97.2 | 0.455 |
| 45,000 | 79.7 | 105.6 | 0.572 | 70.5 | 97.5 | 0.443 |
| 47,500 | 79.8 | 105.8 | 0.579 | 69.4 | 95.9 | 0.472 |
| 50,000 | **76.7** | **102.3** | 0.559 | **66.3** | **92.8** | 0.467 |

- 到最後還在下降（沒有收斂）。**ψ 的取捨**：截斷（ψ 0.75）把 w 拉向平均，臉比較「標準」→ AFD 高約 0.1；但多樣性變少 → FID 反而比 ψ 1.0 差約 10。
- 第 15,000 步 FID64 101.8 → 第 17,500 步 125.6 的回升是單一存檔的波動（之後持續下降），不是 DCGAN 那種崩潰。

## 總比較（各組最佳存檔，n = 1000、種子 0）
| 組 | 最佳 FID64 | 在哪 | 該點 FID96 | 該點 AFD | 訓練時間 |
|---|---|---|---|---|---|
| gan（train.py） | 105.0 | epoch 15 | 134.2 | 0.366 | 29 分（100 epoch） |
| gan 的 G_99（test.py 實際用的） | 211.9 | epoch 100 | 233.2 | 0.666 | |
| wgan | 90.3 | epoch 100 | 120.7 | 0.364 | 19.7 分 |
| wgangp | 94.1 | epoch 95 | 123.0 | 0.362 | 44.1 分 |
| wgan_sigmoid | 307.8 | epoch 1 | 315.4 | 0.0 | 28.9 分 |
| StyleGAN2 ψ1.0 | **66.3** | 50,000 步 | 92.8 | 0.467 | 4 時 18 分 |
| StyleGAN2 ψ0.75 | 76.7 | 50,000 步 | 102.3 | 0.559 | （同一次訓練） |
| 真實圖（參考） | — | — | — | real64 0.5375 | |

## 模型（ch00 模型總覽用；forward hook 實測，batch 2）
**Generator**（generator.py，`Generator(100)`，feature_dim 64）共 **5,142,080** 參數：

| 模組 | 層 | 輸出形狀 | 參數 |
|---|---|---|---|
| l1.0 | Linear(100 → 8192, bias=False) | (B, 8192) | 819,200 |
| l1.1 | BatchNorm1d(8192) | (B, 8192) | 16,384 |
| l1.2 | ReLU | (B, 8192) → view (B, 512, 4, 4) | 0 |
| l2.0 | ConvTranspose2d(512→256, k5, s2, p2, op1, no bias) + BN + ReLU | (B, 256, 8, 8) | 3,276,800 + 512 |
| l2.1 | ConvTranspose2d(256→128) + BN + ReLU | (B, 128, 16, 16) | 819,200 + 256 |
| l2.2 | ConvTranspose2d(128→64) + BN + ReLU | (B, 64, 32, 32) | 204,800 + 128 |
| l3.0 | ConvTranspose2d(64→3) | (B, 3, 64, 64) | 4,800 |
| l3.1 | Tanh | (B, 3, 64, 64) | 0 |

**Discriminator**（discriminator.py，`Discriminator(3)`）共 **2,766,529** 參數（conv 都有 bias）：

| 模組 | 層 | 輸出形狀 | 參數 |
|---|---|---|---|
| l1.0 | Conv2d(3→64, k4, s2, p1) + LeakyReLU(0.2) | (B, 64, 32, 32) | 3,136 |
| l1.2 | Conv2d(64→128) + BN + LeakyReLU | (B, 128, 16, 16) | 131,200 + 256 |
| l1.3 | Conv2d(128→256) + BN + LeakyReLU | (B, 256, 8, 8) | 524,544 + 512 |
| l1.4 | Conv2d(256→512) + BN + LeakyReLU | (B, 512, 4, 4) | 2,097,664 + 1,024 |
| l1.5 | Conv2d(512→1, k4, s1, p0) | (B, 1, 1, 1) | 8,193 |
| l1.6 | Sigmoid | (B, 1, 1, 1) → view(-1) (B,) | 0 |

- 檔案裡的形狀註解是錯的：discriminator.py 寫 `(batch, 3, 32, 32)`、`(batch, 3, 16, 16)`…（實際 64／128／256／512 channel），generator.py 寫 `(batch, feature_dim * 16, 8, 8)`…（實際 256／128／64 channel）。
- 「判別器至少 4 層」（報告第 2 題）：這個 D 有 5 個 conv，梯度範數表記的就是這 5 個。

## 照註解字面改會怎樣（HW06 複製到 scratchpad、faces 用 symlink、照跑 train.py）
- **WGAN-GP**：把 `trainer_gan.py:128-130` 的 GAN loss 換成註解裡的兩行（`gradient_penalty = self.gp(r_imgs, f_imgs)`、`loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty`）→ 第一步就
  `TypeError: TrainerGAN.gp() takes 1 positional argument but 3 were given`（`def gp(self)` 只收 self，而且本體是 `pass`）。
- **WGAN clipping**：把第 142–143 行的註解拿掉 → 第一步就 `KeyError: 'clip_value'`（config.py 沒有這個鍵）。
- **只換 loss_G**（使用者筆記「WGAN, WGAN-GP 只替換 loss_G 是不能使用的」的字面做法：第 164 行改成 `loss_G = -torch.mean(self.D(f_imgs))`，D 仍是 Sigmoid + BCE），config 改 20 epoch：
  - 第 1 個 epoch 結束時 `loss_D=6.96e-7, loss_G=-3.85e-7`，`Epoch_001.jpg` 全是灰色雜訊（D 完全壓制，-mean(sigmoid) 在 D(f)≈0 時梯度幾乎是 0）。
  - 但之後恢復：第 5、10、15、20 epoch 結束 loss_D 0.0542／0.0777／0.0387／1.51e-5。FID64（n 1000、種子 0）：epoch 1 319.9、5 266.1、10 136.7、15 131.0、20 **122.9**（FID96 146.7、AFD 0.385）；`Epoch_020.jpg` 是清楚、各不相同的臉。逐行：`docs/tools/hw06_lit_lossg_eval.jsonl`。
  - → **不是「不能用」**，是第 1 個 epoch 看起來不能用；原版 notebook `n_epoch` 是 1，照 notebook 跑只會看到雜訊。和 gan 比：同樣第 20 epoch，gan 已崩（347.1），只換 loss_G 的是 122.9；但它在 epoch 15 是 131.0，比 gan 的 105.0 差。
- 第 1 個 epoch 速度：45.03 it/s（其他 epoch 約 62 it/s；第一個 epoch 含 cudnn／worker 暖身）。

## 速度與可重現（`hw06_exp.py --n_epoch 5`，2026-10-07 19:41 起依序，GPU 前後 `--query-compute-apps` 空）
| 設定 | 5 epoch 秒數 | 各 epoch 秒數 | G_4／D_4 與 baseline |
|---|---|---|---|
| workers 2（train.py） | 86.1 | 17.6／17.0／17.0／17.3／17.3 | 相同 |
| workers 2 + `--detach 1` | **73.8** | 15.1／14.7／14.7／14.7／14.7 | **相同** |
| workers 0 | 144.9 | 29.4／28.9／28.8／28.8／28.9 | 相同 |
| workers 4 | 86.7 | 17.6／17.2／17.3／17.2／17.5 | 相同 |
| workers 8 | 87.5 | 17.8／17.4／17.5／17.3／17.5 | 相同 |
| workers 8 + `OMP_NUM_THREADS=1` | 87.5 | 17.8／17.4／17.4／17.3／17.5 | 相同 |
| workers 2 + `OMP_NUM_THREADS=1` | 87.1 | 17.7／17.3／17.4／17.3／17.4 | 相同 |

（「相同」= G_4.pth、D_4.pth 全部 `torch.equal`；w2／detach／w0／w4／w8 的 `Epoch_005.jpg` 也位元組相同。）
- **detach**（疑點 3）：D 那一步改成 `D(f_imgs.detach())`，每 epoch 17.2 → 14.7 s，**快約 15%，結果逐位元不變**——loss_D 對 G 算的梯度本來就會在 `self.G.zero_grad()`（G 那一步 backward 之前）被清掉，白算。
- **num_workers**（疑點 10）：0 → 2 快 1.7 倍；2 → 4 → 8 沒有再快（2 個 worker 時瓶頸已經不在讀圖）。worker 數也**不影響結果**：transform 沒有隨機性，shuffle 的排列由主行程的全域亂數決定。
- `OMP_NUM_THREADS=1` 沒有差別（HW03 快了 18%，這裡沒有）。
- **n_critic > 1 時印的是舊 loss_G**（疑點 5）：只有 `steps % n_critic == 0` 那步才重算 loss_G，而 `set_postfix` 在 `steps % 10 == 0` 印。n_critic 是 1、2、5、10 時，10 的倍數一定是 G 更新步，印出的永遠是新的；n_critic 是 3、4、6… 時（例如 n_critic 3 的第 10 步，上一次 G 更新在第 9 步）才會印到舊值。純推論，本書沒有實際跑。

## FID 的雜訊與樣本數（`docs/tools/hw06_fid_sensitivity.jsonl`）
**換 z 種子**（n = 1000，種子 0–4；種子 0 取自主表）：

| 存檔 | 5 個種子的 FID64 | 平均 | 標準差 | 範圍 |
|---|---|---|---|---|
| gan G_14 | 105.0／107.5／107.2／105.8／106.9 | 106.48 | 1.05 | 105.0–107.5 |
| wgan G_99 | 90.3／86.9／90.7／89.7／88.8 | 89.28 | 1.51 | 86.9–90.7 |
| wgangp G_94 | 94.1／92.2／95.3／92.9／93.3 | 93.56 | 1.19 | 92.2–95.3 |
| StyleGAN2 model_20 ψ1.0 | 66.3／68.5／68.0／66.4／67.6 | 67.36 | 0.98 | 66.3–68.5 |

- 種子造成的波動約 ±1.5；wgan 和 wgangp 的範圍不重疊，「wgan 的最佳 FID 比 wgangp 好」在 n = 1000 下站得住（但只差約 4）。
- AFD 隨種子：gan G_14 0.352–0.366、wgan G_99 0.353–0.383、wgangp G_94 0.375–0.395、StyleGAN2 ψ1.0 0.444–0.474。

**n = 10,000**（種子 0）：

| 存檔 | FID64 n=1000 → n=10000 | FID96 n=10000 | AFD n=10000 |
|---|---|---|---|
| gan G_14 | 105.0 → **90.8** | 121.2 | 0.3586 |
| gan G_99 | 211.9 → 205.9 | 227.4 | 0.6818 |
| wgan G_99 | 90.3 → **74.0** | 104.6 | 0.3616 |
| wgangp G_94 | 94.1 → **78.4** | 108.0 | 0.383 |
| StyleGAN2 ψ1.0 | 66.3 → **52.3** | 79.3 | 0.4617 |
| StyleGAN2 ψ0.75 | 76.7 → **64.1** | 89.9 | 0.5623 |

- 樣本少 → FID 系統性偏高（共變異矩陣估不準）：n 從 1000 到 10000，多數下降 14.0–16.3；崩掉的 gan G_99 只降 6（本來就多樣性低）。**排名不變**。作業規定交 1000 張，所以本書的主表一律用 n = 1000。

## test.py 與提交檔
- 照跑 `test.py`（在 scratchpad 的複本裡，checkpoints 只連 2026-10-07 那次）：印 `Inference with ./checkpoints/2026-10-07_11-00-32_GAN/G_99.pth`，產生 **100 張**（`output/1.jpg`–`100.jpg`），不是作業要的 1000 張。
- **兩次 test.py 的輸出不同**（`output/1.jpg` md5 `b8699eec…` vs `f773f383…`）：test.py 沒有呼叫 `same_seeds`，`inference` 的 `torch.randn` 每次不同。
- 100 張 tar.gz：116,043 bytes。
- **1000 張**（hw06_eval 的 `--keep`，`torchvision.utils.save_image` 存 JPEG，n 1000、種子 0）照投影片的 `tar -zcf ... *.jpg`：gan G_99 jpg 共 1,736,900 B → tgz **1,138,801 B**；wgan G_99 1,761,607 → **1,180,344**；StyleGAN2 ψ1.0 1,747,508 → **1,167,445**。都在 2MB 限制以內（約 1.1–1.2 MB）。tar 裡是 `1.jpg`、`10.jpg`…（沒有資料夾，符合 p.12 第 3 點）。

## Crypko（ch01 用；2026-10-08 網路查證）
- Preferred Networks（PFN，日本）的動漫角色生成服務，以 GAN 為基礎；2018 年開發出產生臉部插圖的模型，**2018 年 5 月公開 beta**；第一代服務 **2019 年 3 月結束**，當時用 blockchain 智慧合約記錄角色的生成、融合與使用者的對應（PFN 2019 公告，沒寫是哪一條鏈）。來源：https://www.preferred.jp/en/news/pr20190403
- 2022 年 4 月在日本以瀏覽器平台重新上線（可生成臉或上半身、融合、編輯髮色表情等 30 多項屬性；第六代可生成上半身），**2022-06-29 推出英文與簡中版**。來源：https://www.preferred.jp/en/news/pr20220629
- 2022 年 8 月加入可商用的 Premium Plan。來源：https://www.preferred.jp/en/news/pr20220825
- **2025 年 6 月結束一般使用者服務**（PFN entertainment 頁）；萌娘百科寫 2025-06-30 起無法連上（另有寫 7 月的版本，日期以 PFN 為準）。來源：https://www.preferred.jp/en/industries/entertainment
- 查不到的：PFN 用的是哪一種 GAN 架構（公告沒寫 StyleGAN）、是哪一條鏈。教材不寫這兩點。
- 投影片 p.9：「Website which can generate anime face by yourself」「Thanks Arvin Liu for collecting the dataset」。→ 投影片把資料歸給 Crypko，但**資料集何時、怎麼收集沒有公開來源**（2026-10-08 再查一次仍查不到）；目視前 24 張有幾張很像東方 Project 的角色（Flandre、Marisa 的帽子）。所以教材只能說「投影片說來自 Crypko 這個 GAN 生成服務」，**不能斷言每張都是 GAN 的輸出**。index 目錄與 outline 原本的說法已改。
- 使用者決定（2026-10-08）：書裡放少量 Crypko 原圖。

## ch00 實測（2026-10-08）
- baseline 進度條（`baseline_stderr`，\r 換行後取每個 epoch 最後一格）：
  - `Epoch 1: 100%|██████████| 1115/1115 [00:20<00:00, 58.45it/s, loss_D=0.171, loss_G=5.12]`
  - `Epoch 2: … [00:17<00:00, 64.78it/s, loss_D=0.175, loss_G=3.25]`；`Epoch 15: … 64.65it/s, loss_D=0.085, loss_G=4.37]`；**`Epoch 20: … 64.41it/s, loss_D=3.03e-7, loss_G=49.9]`**（崩潰的那個 epoch）；`Epoch 99: … 64.13it/s, loss_D=0.0833, loss_G=14]`；`Epoch 100: … 64.41it/s, loss_D=0.0402, loss_G=24.4]`。
  - 第 1 個 epoch 第 0 步：`loss_D=0.929, loss_G=3.43`。每個 epoch 開頭先出現一格沒有描述的 `  0%|          | 0/1115`（tqdm 建好後才 `set_description`），在終端機上會被同一行覆蓋。
  - logging 的 INFO（stderr，時間到分鐘）：`2026-10-07 11:00 - INFO: Save some samples to ./logs/2026-10-07_11-00-32_GAN/Epoch_001.jpg.`；Epoch_015 是 11:04、Epoch_020 是 11:06、Epoch_100 是 11:29；最後 `2026-10-07 11:29 - INFO: Finish training`。stdout 是空的。
- 訓練時 GPU 記憶體約 760 MiB（nvidia-smi）。
- 輸出大小：`logs/2026-10-07_11-00-32_GAN/` 100 張共 14 MB（Epoch_001.jpg 158,176 B、662×662）；`checkpoints/…` 21 對共 636 MB，G 每個 20,644,700 B，D 11,080,476（D_0）／11,080,507 B。
- `test.py`（scratchpad 複本）：印 `Inference with ./checkpoints/2026-10-07_11-00-32_GAN/G_99.pth`，**7.59 s**，`output/1.jpg` 1,797 B。
- 參數計數指令（ch00 的 shell cmd）輸出逐字已在 ch00；轉置卷積權重 (in, out, k, k)、卷積 (out, in, k, k)。
- 判別器吃 128×128：輸出 `torch.Size([100])`（4 張 × 5×5），BCELoss 報 `ValueError: Using a target size (torch.Size([4])) that is different to the input size (torch.Size([100])) is deprecated. Please ensure they have the same size.`
- `HW06/.gitignore`：`checkpoints/*`、`faces/*`、`logs/*`、`output/*`、`*.zip`。`HW06/筆記.txt` 日期 2022/10/01，三條（n_epoch 提升有效；WGAN/WGAN-GP 只換 loss_G 不能用；StyleGAN2 在 win10 + py3.8 裝不起來）。
- 圖：`docs/HW06/img/ch00_epoch015.png`、`ch00_epoch100.png`（`Epoch_015/100.jpg` 前 3 列，662×200）。

## ch01 實測（`docs/tools/hw06_facts.py`，2026-10-08）
- `data`：71,314 個檔，**全部** 96×96 RGB JPEG（逐一打開）；總 453,033,731 bytes（`du` 560M 是區塊配置）、平均 6,352.7、最小 2,074、最大 9,219。glob 順序前 5：`56141, 6620, 54470, 32619, 50504`；既不是數字順序也不是字串排序。
- `faces/0.jpg` 逐步：`read_image` uint8 (3, 96, 96) 8–255 → `ToPILImage` PIL 96×96 RGB → `Resize` 64×64（`InterpolationMode.BILINEAR`、antialias True）→ `ToTensor` float32 (3, 64, 64) 0.0980–1.0000 → `Normalize` -0.8039–1.0000。
- `stats`（全部 71,314 張轉換後）：mean RGB **0.4476、0.2342、0.1957**，std 0.5115、0.5166、0.4814，min -1.0000、max 1.0000；換回 [0,1] 的平均 0.7238、0.6171、0.5979。
- `loader`（只跑 DataLoader，batch 64、shuffle）：len 1,115、最後一批 (18, 3, 64, 64)；一個 epoch：workers 0 **14.2 s**、2 **9.3 s**、4 **4.7 s**、8 **2.6 s**。對照完整訓練每 epoch 28.9／17.2／17.3／17.4 s（速度表）。
- `figs`：`docs/HW06/img/ch01_crypko96.png`（0–23.jpg 原圖，12×2）、`ch01_crypko64.png`（轉換後再 (x+1)/2）。
- 71,313 = 3 × 11 × 2,161；batch 3、11、33 時最後一批剩 1 張。生成器（train 模式）吃 1 個 z：`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 8192])`。
- 解碼後大小：96×96 全部 ≈ 1.97 GB（uint8）、64×64 ≈ 0.876 GB。

## ch02 實測（`hw06_facts.py model` 與單次驗證，2026-10-08）
- 轉置卷積 k5 s2 p2：`output_padding=1` 4×4 → 8×8；`output_padding=0` → **7×7**。公式 (n−1)·2 − 4 + 5 + 1：4→8→16→32→64。卷積 k4 s2 p1：64 → 32。
- **conv_transpose2d = conv2d 的梯度**：W (8,4,5,5)、16×16 → 8×8 的 conv2d，`autograd.grad` 對輸入的梯度與 `F.conv_transpose2d(g, W, stride=2, padding=2, output_padding=1)` 最大差 **0.00e+00**（形狀都是 (1, 4, 16, 16)）。
- 一維 `conv_transpose1d`，輸入 [1, 10, 100, 1000]、kernel 全 1、s2：無 padding 11 格 `[1, 1, 11, 11, 111, 110, 1110, 1100, 1100, 1000, 1000]`；p2 op1 → 8 格 = 完整 11 格的第 2–9 格。保留的 8 格重疊次數 2、2、3、2、3、2、2、1。
- `weights_init`（same_seeds(2022) → Generator → Discriminator）：G `l1.0` Linear **沒被處理**，std 0.0577（= 0.1/√3，|w| 最大 0.1000）；ConvTranspose2d／Conv2d std 0.0199–0.0200、mean ±0.0004 以內；BN weight mean 0.997–1.0015、std 0.017–0.021、bias 全 0；D 的 conv bias 不是 0（未動）。
- 剛初始化（CPU、z 64 個、訓練模式）：G 輸出 mean −0.0511、std 0.2703、min −0.9416、max 0.9381、|x|>0.99 的比例 0。D 對前 64 張真圖平均 0.5134（0.0652–0.8986），對 G 的假圖 0.6320（0.1602–0.9226）；BCE loss_D 0.9949、loss_G 0.5257（ln2 = 0.6931）。
- **重現 train.py 第 0 步（GPU，照 train.py 的亂數順序）**：D 更新前 D(real) 0.5164、D(fake) 0.6125、loss_D **0.9293**（進度條印 0.929）、用更新前的 D 算的 loss_G 0.5622；D 用 Adam 更新一步後，新的一批假圖 D(fake) **0.0520**、loss_G **3.4329**（進度條印 3.43）。
- G 四層 output_padding 全改 0：輸出 (2, 3, 49, 49)；送進 D：`RuntimeError: Calculated padded input size per channel: (3 x 3). Kernel size: (4 x 4). Kernel size can't be greater than actual input size`。
- 自訂 `class ConvBlock(nn.Module)`（只包一層 Conv2d）`.apply(utils.weights_init)`：`AttributeError: 'ConvBlock' object has no attribute 'weight'`。
- `Epoch_020.jpg` 放大：有規則重複的格狀紋路（週期大於 2 px），**未驗證**是否為 checkerboard artifact。

## ch03 實測（2026-10-08）
- `hw06_logs/gan.jsonl` 的欄位：`d_real` = 判別器那一步（更新前）對真圖的平均；`d_fake` = **生成器那一步、D 更新後**對新一批假圖的平均（第 0 步 0.0520 即此）。每 epoch 112 筆（最後一個 111 筆）。
- 每 epoch 彙整（D 分數取平均、loss 取中位數）：ep1 0.875／0.016／loss_G 5.58；ep5 0.865／0.038／4.47；ep10 0.879／0.041／4.48；ep15 0.910／0.048／4.38；ep18 0.896／0.045／4.46；ep19 0.949／0.027／6.21；**ep20 1.000／0.000／49.53**；ep21 1.000／0.000／48.67；ep22 1.000／0.000／39.86；ep23 0.936／0.014／38.56；ep24 0.841／0.069／3.90；ep25 0.942／0.031／5.06；ep30 0.982／0.002／8.06；ep50 0.979／0.004／6.75；ep70 0.981／0.005／7.08；ep90 0.978／0.005／6.78；ep100 0.989／0.004／7.92。
- 非崩潰 epoch（排除 19–23）：D(real) 0.834–0.992、D(fake) 0.0012–0.0852；ep≥30 D(fake) 0.0012–0.0110；loss_G 中位數 3.90–8.06（ep<19：3.91–5.58；ep≥30：6.20–8.06）；loss_D 中位數 0.0043–0.2337。
- 崩潰區間：logged step 21,080（ep19）起 loss_G > 20 持續，到 25,370（ep23）結束（約 4,300 步）；之後 ep25–26 有零星 loss_G > 30 的單點。**最大 loss_D 11.2757 在 step 18,970（ep18）**。
- `G.eval()` vs `G.train()`（同 100 個 z，種子 0）：G_14 平均 |差| 7.09／255（最大 103.9）、G_19 1.49（23.3）、G_99 14.69（190.2）。
- `G_99.pth` 的 `l1.1.num_batches_tracked` = **223,000** = 2 × 111,500。
- non-saturating vs minimax 梯度（對 Sigmoid 前的 a）：D(G(z)) = 0.052 時 −σ = −0.052、−(1−σ) = −0.948，約 18.2 倍。
- 圖 3.1 由 scratchpad 的 `ch03_chart.py` 產生（讀 gan.jsonl），配色 #3987e5／#d95926 以 validate_palette.js 在 #161c24 上驗證通過（CVD ΔE 26.8、normal 31.8）。
- 原始碼行尾空白：trainer_gan.py 第 43、119、121、123 行行尾有一個空白，引用時要保留。

## ch04 實測（2026-10-08；`hw06_eval.py` 加了 `realsub`、`--parts`、`--nojpeg`，預設輸出不變：G_14 重跑 fid64 104.95、afd 0.366；逐行 `docs/tools/hw06_ch04_eval.jsonl`）
- **FID64 的兩項**（`--parts`；mean term = ‖μ−μ_real‖²，cov term = FID − mean term；trace Σ_real64 = 123.3）：
  | 對象 | FID64 | mean term | cov term | tr Σ |
  |---|---|---|---|---|
  | gan G_14 | 104.95 | 44.2 | 60.7 | 138.6 |
  | gan G_19（崩潰） | 347.12 | **251.1** | 96.0 | **55.6** |
  | gan G_99 | 211.93 | 82.2 | 129.7 | 123.1 |
  | StyleGAN2 model_20 ψ1.0 | 66.34 | 22.8 | 43.5 | 138.2 |
  | 真圖 1000 張（seed 0，JPEG） | 31.60 | 8.2 | 23.4 | 139.5 |
- **真圖的 FID 底線**（`realsub`，get_dataset 轉換後當成生成圖，同樣存 JPEG 讀回）：n 1000 seed 0–4 → FID64 31.60／31.95／31.97／31.99／31.90，FID96 59.87–60.56，AFD 0.412／0.401／0.412／0.416／0.459；n 10000 seed 0 → FID64 **15.78**（mean 7.9、cov 7.9）、FID96 44.48、AFD 0.419。不經 JPEG（`--nojpeg`）n 1000 seed 0 → FID64 **16.05**、FID96 30.96、AFD 0.523。
- **JPEG 的影響**（n 1000、種子 0）：G_14 JPEG 104.95 vs 不經 JPEG **118.20**（FID96 134.21 vs 133.96；AFD 0.366 vs 0.382）；G_99 211.93 vs **232.89**（FID96 233.24 vs 242.63；AFD 0.666 vs 0.656）。→ 生成圖存成 JPEG 後 FID64 變好，真圖存成 JPEG 後變差（16.05 → 31.60）。
- `torchvision.utils.save_image` 存 JPEG：`ndarr = grid.mul(255).add_(0.5).clamp_(0, 255)…`、`im.save(fp, format=format)`；與 PIL `quality=75` 存出的檔案位元組相同（90、95 不同）→ **品質 75**。
- pytorch-fid 的 InceptionV3 預設 `resize_input=True`（雙線性放大到 299×299）、`normalize_input=True`。
- AFD 範例：G_99 的 1000 張（種子 0、`--keep`）偵測到 666 張、沒偵測到 334 張；前 10 張偵測到的是 2, 4, 6, 7, 9, 10, 11, 12, 13, 16，沒偵測到的是 1, 3, 5, 8, 14, 15, 20, 22, 23, 28。圖 `docs/HW06/img/ch04_afd_yes.png`（畫了偵測框）、`ch04_afd_no.png`。兩排目視差不多。
- `inference` 的 `show=True` 分支：`row = n_output // 10 + 1` = 4 被當成 `make_grid` 的 `nrow`（每列張數），`figsize=(row, col)` = (4, 10)（推論，`show` 預設 False，沒有實際跑）。

## ch05 實測（2026-10-08）
- `hw06_facts.py div <run_dir>`：每張樣本格 `Epoch_XXX.jpg`（662×662、10×10、padding 2）切成 100 張 64×64，算 100 張之間每個像素的標準差平均（0–255）與兩兩 L2 中位數。結果存 `docs/tools/hw06_logs/div_{gan,wgan,wgangp,wgan_sigmoid}.jsonl`（gan 用 HW06/logs/2026-10-07_11-00-32_GAN，其餘用 grid 的輸出資料夾）。
- DCGAN：std < 30 的 epoch = **19–22、25–27、31、34–35**；最低 17.83（ep19）；最高 60.66（ep16）；ep2–18 46.35–60.66；ep23 53.38、ep24 57.29（恢復）；ep27 20.89、ep28 44.68、ep29 36.86、ep30 38.16、ep31 29.94、ep32 30.19、ep33 30.24、ep34 28.60、ep35 21.23、ep36 44.59；ep36–100 35.6–54.8，其中 78% 在 40–50；ep55 37.21、ep90 51.26、ep100 47.00。
- WGAN：全程 ≥ 51.75（ep1），ep≥10 最低 56.35，ep100 57.8；最高 65.69（ep4）。WGAN-GP：全程 ≥ 53.45（ep6），ep≥10 最低 53.91，ep100 58.24。wgan_sigmoid：全程 < 22（最低 15.71 ep13、最高 21.67）。
- 真圖參考（get_dataset 轉換後 tensor、不經 JPEG）：隨機 100 張（種子 0／1／2）std 58.96／57.10／58.19，兩兩 L2 中位數 35.14／34.27／34.99。
- 21 個存檔依 FID64 排序，G_99（211.9）是第 9 名；G_44 178.3 是第 40 epoch 之後最低。
- 樣本格前 2 列（20 個 z）：`docs/HW06/img/ch05_epoch{015,020,024,026,035,100}.png`（662×134）。目視：ep20 花掉的同一張；**ep24 恢復且同位置與 ep15 相像**；ep26 幾乎同一張棕髮瞇眼微笑臉；ep35 同一種模糊臉；ep100 較糊、偏淡。
- 四色配色（DCGAN #3987e5、WGAN #d95926、WGAN-GP #199e70、StyleGAN2 #c98500）以 validate_palette.js 在 #161c24 上全部通過（最差相鄰 CVD ΔE 8.4、normal 19.8）。圖由 scratchpad 的 `ch05_chart.py` 產生。

## ch06 實測（2026-10-08）
- WGAN log（`hw06_logs/wgan.jsonl`）：第 0 步 loss_D 0.4686、loss_G 0.0034、d_real 0.0655、d_fake −0.0034。−loss_D 每 epoch 中位數：ep1 1.034、ep2 1.317、ep3 1.331、**ep4 1.346（最高）**、ep5 1.310、ep10 1.250、ep15 1.192、ep20 1.121、ep30 0.895、ep50 0.713、ep70 0.614、ep100 0.524（最低 0.519）。
- **Spearman（21 個存檔的 FID64 vs 同 epoch 的 loss 中位數）**：WGAN −loss_D **0.986**；DCGAN loss_D −0.223、loss_G 0.442。
- WGAN FID64 只有一次回升：ep60 100.6 → ep65 101.1。
- wgan_sigmoid：ep1 D(real) 0.7623、D(fake) 0.0010、loss_D 中位數 −0.9998；ep2 1.0000／0.4234／−1.0000；**ep5 起 D(real)、D(fake) 都 1.0000、loss_D −0.0000、loss_G −1.0000**。梯度範數第一次全為 0 在 step 7,400（ep7），之後到 ep≈8 仍有 37 筆非零，ep91–100 的 111 筆全部是 0。float32 `torch.sigmoid(17.)` == 1.0（16.5 不等於）。
- WGAN 判別器卡在 ±0.01 的比例（不含 running stats）：D_0 11.4%、D_49 22.3%、D_99 22.0%；各層 weight：D_0 l1.0 .04／l1.2.0 .04／l1.2.1(BN) .60／l1.3.0 .05／l1.3.1 .95／l1.4.0 .13／l1.4.1 1.00／l1.5 .99；D_49 .04／.03／.45／.06／.59／.27／1.00／1.00；D_99 .04／.02／.26／.05／.52／.27／1.00／.99。max |w| 全部 0.0100。
- 只換 loss_G（lit_lossg）每 epoch 結束：ep1 `45.03it/s, loss_D=6.96e-7, loss_G=-3.85e-7`；ep2 `62.17it/s, loss_D=0.133, loss_G=-7.75e-5`；ep3 `63.99it/s, loss_D=0.266, loss_G=-0.00744`；ep4 `63.52it/s, loss_D=0.193, loss_G=-0.0681`；ep5 `62.48it/s, loss_D=0.0542, loss_G=-0.0168`；ep10 `62.26it/s, loss_D=0.0777, loss_G=-0.0179`；ep20 `61.88it/s, loss_D=1.51e-5, loss_G=-6.74e-6`。圖 `docs/HW06/img/ch06_lossg_epoch{001,020}.png`（樣本格第一列）。
- 圖 6.3 由 scratchpad 的 `ch06_chart.py` 產生。

## ch07 實測（2026-10-08）
- 判別器 5 層權重梯度範數（訓練中每 100 步，`loss_D.backward()` 後；WGAN-GP 含 penalty），**ep91–100 的 111 筆，中位數 [四分位]**：
  - clipping（wgan）：49.5 [41.6, 59.4]、18.2 [17.3, 19.1]、3.68 [3.44, 3.88]、0.316 [0.294, 0.331]、0.614 [0.575, 0.661]；conv1/conv4 = **156.7**；相鄰層比 2.72、4.95、11.6。
  - GP（wgangp）：32.7 [28.0, 39.8]、13.8 [12.7, 15.6]、12.9 [12.2, 14.0]、13.9 [13.3, 14.9]、5.21 [4.92, 5.55]；conv1/conv4 = **2.4**（前 4 層最大/最小 2.55）。
  - DCGAN：2.41 [1.04, 5.78]、0.457、0.243、0.170、0.237；conv1/conv4 = 14.2。
  - 全程（1,115 筆）中位數：wgan 42.9、16.5、3.58、0.318、0.801；wgangp 40.6、16.0、14.3、14.0、5.34；gan 4.43、0.961、0.485、0.330、0.469。
- **critic 對輸入的梯度 ‖∇ₓD(x)‖**（載入 D_e 與 G_e、critic 訓練模式、256 張真圖〔get_dataset 轉換，sorted 檔名 randperm 種子 0〕、256 張生成圖〔z 種子 1〕、內插 α 種子 2，每批 64；平均 ± 標準差）：
  - wgan D_0 真 0.0375±0.0100、假 0.0619±0.0095、內插 0.2064±0.0318；D_49 0.5814±0.0401、0.2052±0.0405、0.3357±0.0932；D_99 0.5419±0.1240、1.4672±0.1200、1.3438±0.1949。
  - wgangp D_0 2.7512±0.5986、2.1959±0.2491、1.7788±0.4937；D_49 1.7202±0.4498、1.5733±0.3969、1.1718±0.2999；D_99 1.7021±0.4476、1.6916±0.4316、**1.2015±0.2707**。
- WGAN-GP 的 FID64 回升只有三次：ep40→45 125.5→129.3、ep80→85 98.8→101.1、ep95→100 94.1→96.7。
- 時間比：wgangp 2,647.5 s ÷ wgan 1,180.8 = 2.24；÷ gan 1,732.3 = 1.53。
- 照註解字面改 WGAN-GP 的 traceback（lit_gp）：`    gradient_penalty = self.gp(r_imgs, f_imgs)` / `                       ^^^^^^^^^^^^^^^^^^^^^^^` / `TypeError: TrainerGAN.gp() takes 1 positional argument but 3 were given`。
- lucidrains stylegan2-pytorch 1.9.0 的 `gradient_penalty(images, output, weight = 10, center = 0.)`：以 0 為目標；`apply_gradient_penalty = self.steps % 4 == 0`；`gp = gradient_penalty(image_batch, real_output) + gradient_penalty(generated_images, fake_output)`（真圖與生成圖都算）。
- 圖 7.1 由 scratchpad 的 `ch07_chart.py` 產生（WGAN 橘、WGAN-GP 青綠、DCGAN 藍虛線）。
