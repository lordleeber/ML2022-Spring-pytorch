# HW14 事實清單（不進最終教材）

量測環境：RTX PRO 4000 Blackwell、共用 `.venv`（Python 3.12、torch 2.11.0+cu128、torchvision 0.26.0、matplotlib 3.11.2、numpy 2.5.3、tqdm 4.70.1）。

## 原版 notebook vs 本 repo（Phase 0，2026-10-08）

`HW14/HW14.ipynb` 是官方 notebook（44 格）。本 repo 把程式拆成：

| 檔案 | 對應 notebook 格 | 內容 |
|---|---|---|
| `config.py` | [13] 前半 | `Args`（5 任務、每任務 10 epoch、lr 1e-4、batch 128、test_size 8192）、`angle_list` |
| `utils.py` | [8] | `same_seeds` |
| `dataset.py` | [11] | `_rotate_image`、`get_transform`、`Pad`、`Data` |
| `model.py` | [18] | `Model`（784→1024→512→256→10） |
| `trainer.py` | [21]、[23] | `train`、`evaluate` |
| `methods/*.py` | [26] [29] [32] [35] [38] [41] | 六個方法的類別，逐字複製 |
| `train.py` | [13] 後半、[18] 的 `example`、[27] [30] [33] [36] [39] [42] | 依 notebook 順序跑六種方法，結果存 `output/acc.json` |
| `plot.py` | [44] | `draw_acc`，`lineStyle` 改成 `linestyle` |

改動（都為了在一般 Python 跑起來，或是作業本身要寫的）：
- `trainer.py` 加 `import tqdm.auto`：notebook 只 `import tqdm`，在 Colab 有別的套件先載入了 `tqdm.auto`；照原樣用 python 跑，第一次呼叫 `tqdm.auto.trange` 就 `AttributeError: module 'tqdm' has no attribute 'auto'`（2026-10-08 實測）。
- `methods/mas.py` 的 TODO 填入 global 版 MAS（輸出平方 L2 範數對參數的梯度取絕對值，除以 batch 數累加）。
- `plot.py`：Matplotlib 3.11 不接受 `lineStyle`（`AttributeError: Line2D.set() got an unexpected keyword argument 'lineStyle'`，實測）；改 `linestyle`，用 Agg 存檔，不 `plt.show()`。
- `train.py` 保留 notebook 第 [18] 格的 `example = Model()`：它會消耗全域亂數，拿掉就和 notebook 的數字不同。

MNIST：torchvision 0.26 下載到 `HW14/data/MNIST/MNIST/raw`（`root=os.path.join(path, "MNIST")`，torchvision 再加一層 `MNIST/`），訓練 60,000、測試 10,000；`HW14/data/` 共 64 MB，加進 `.gitignore`。

## 投影片重點（`HW14/HW14.pdf`，35 頁）
- p.10：5 個任務、每任務 10 epoch；每種方法 T4 約 20 分、K80 約 60 分。
- p.12：每個任務比前一個多轉 20°。
- p.14：評估「用一個特別的指標，請讀程式並在報告描述」。
- p.19–27：每個方法都問「需不需要 label？」。
- p.20、p.32：MAS 只准改 TODO 區塊，報告貼 TODO。
- p.29：選擇題另考 iCaRL、LwF、GEM、DGR、三種持續學習情境（van de Ven & Tolias 2019）。
- p.30–32：20 題選擇題 8 分（每題 0.4，全對才給分）；報告 2 分（學習曲線 0.5、描述指標 0.5、MAS TODO 1）。

## 讀程式找到的疑點（Phase 0 要實測確認）

1. **任務數的註解錯**：notebook 第 [9]、[13] 格說「5 different rotations to generate 10 different rotated MNISTs」「from 10 different rotations」，[11] 註解「generate 10 tasks」，EWC 說明寫「learn 10 tasks」；實際 `task_number = 5`、`angle_list = [0, 20, 40, 60, 80]`。
2. **指標**：每個 epoch 結束，對「已經學過的任務」的測試集各算準確率再平均（`test_dataloaders[:train_indexes+1]`）。所以曲線在換任務時，分母從 k 個任務變 k+1 個，不是同一個量；看不到個別任務的遺忘。
3. **guard 重複累加**：EWC／MAS／RWalk／SCP 的迴圈把每個物件的 `_precision_matrices`（已經含前面所有 guard）append 進 `prev_guards`，下一個物件再全部加總，所以舊任務的重要度一直加倍。學完第 4 個任務時：4·F1 + 2·F2 + F3 + F4（推導，待實測量級）。
4. **SCP 先平均再平方**：`total_scalar` 是 L=100 個方向投影的平均，等於「100 個單位向量的平均向量」這一個方向的投影，然後才 backward、平方；L 個 slice 的作用幾乎消失（平均向量長度約 1/√L）。論文版本待查（OpenReview 頁面擋爬蟲，notebook 的虛擬碼圖在 i.ibb.co）。
5. **EWC 的 Fisher 用 batch 平均梯度的平方**，不是逐樣本梯度平方的平均（empirical Fisher 的 batch 近似）；用的是真實 label。公式有 λ/2，程式沒有 1/2。
6. **SI 的 docstring 抄成 EWC 的引用**（kirkpatrick2017overcoming）。
7. **各方法 λ 不同**：baseline 0、EWC 100、MAS 0.1、SI 1、RWalk 100、SCP 100（notebook 設定，未說明怎麼選）。
8. **測試 DataLoader `shuffle=True`**，每次評估都消耗全域亂數（DataLoader 每次建 iterator 都抽 `_base_seed`，shuffle 再抽一次），所以評估會影響訓練的 batch 順序。
9. **`evaluate` 沒有 `torch.no_grad()`**；`train()` 只在開頭 `model.train()`，第一次評估後模型就停在 eval 模式（這個模型沒有 dropout／BN，所以結果不受影響）。
10. **每個任務重建 Adam**（動量歸零）。
11. **plot 函式在新版 Matplotlib 會壞**（見上）；印出的串列在 numpy 2 是 `np.float64(92.58)` 的樣子。

## Baseline 與六種方法（2026-10-08 實測）

- **參照版**：notebook 的程式格原樣串成腳本（拿掉 `!nvidia-smi` 與 plot 格；補 `import tqdm.auto`；MAS TODO 填同一段），`MPLBACKEND` 用 Agg。21:22:07–22:14:14，**52 分 07 秒**（6 種方法），前後 `nvidia-smi --query-compute-apps` 都是空的。單一方法約 7–8 分（工具記錄 421.6–464.1 秒，兩個程式並行時量的，只當參考）。投影片說 T4 約 20 分／方法。
- **逐位元驗證**：`python train.py` 的 stdout 與參照版 `diff` 完全相同；`docs/tools/hw14_exp.py`（不加選項）六種方法的 `acc` 串列與參照版逐值相等（`==`）。工具另外記錄的逐任務矩陣，取前 k 個任務平均後與印出值最大差 5.2e-6（float32 平均的誤差）。
- 參數量 **1,462,538**（784·1024+1024 + 1024·512+512 + 512·256+256 + 256·10+10，工具 `numel` 實測）。
- 原始串列：`docs/tools/hw14_ref_acc.json`；含逐任務矩陣與重要度統計：`docs/tools/hw14_runs.jsonl`（tag `ref`）。

各任務學完時（第 10、20、30、40、50 個 epoch）的平均準確率（notebook 的指標，%）：

| 方法 | λ | 任務1 | 任務2 | 任務3 | 任務4 | 任務5（最後） |
|---|---|---|---|---|---|---|
| baseline | 0 | 98.03 | 95.58 | 87.51 | 77.70 | **70.59** |
| EWC | 100 | 97.77 | 95.31 | 87.38 | 80.40 | **71.66** |
| MAS | 0.1 | 97.89 | 95.74 | 92.70 | 88.20 | **83.06** |
| SI | 1 | 97.69 | 95.95 | 89.34 | 82.30 | **71.98** |
| RWalk | 100 | 97.32 | 96.25 | 91.98 | 85.42 | **78.37** |
| SCP | 100 | 97.64 | 95.32 | 88.64 | 81.31 | **77.20** |

學完全部 5 個任務後，各任務的準確率（%）、剛學完該任務時的準確率（對角線），ACC = 最後一列平均，BWT = 前 4 個任務（最後 − 對角線）的平均：

| 方法 | 最後：任務1..5 | 對角線 | ACC | BWT |
|---|---|---|---|---|
| baseline | 35.4 / 51.6 / 73.1 / 94.4 / 98.4 | 98.0 / 98.1 / 98.3 / 98.3 / 98.4 | 70.58 | −34.55 |
| EWC | 36.4 / 53.7 / 75.1 / 94.9 / 98.2 | 97.8 / 98.3 / 98.2 / 98.4 / 98.2 | 71.66 | −33.15 |
| MAS | 68.7 / 78.5 / 86.4 / 91.9 / 89.7 | 97.9 / 96.8 / 94.6 / 92.2 / 89.7 | 83.04 | −14.00 |
| SI | 37.1 / 54.4 / 75.8 / 94.7 / 97.9 | 97.7 / 98.2 / 97.7 / 98.1 / 97.9 | 71.98 | −32.42 |
| RWalk | 57.0 / 65.7 / 81.7 / 93.0 / 94.4 | 97.3 / 96.7 / 95.5 / 94.5 / 94.4 | 78.36 | −21.65 |
| SCP | 45.3 / 64.9 / 82.6 / 95.5 / 97.7 | 97.6 / 98.1 / 98.0 / 97.7 / 97.7 | 77.20 | −25.77 |

Baseline 學完任務 1 時，各任務是 98.03 / 90.53 / 55.66 / 25.68 / 14.02：轉 20° 的任務 2 不學也有 90%，轉 80° 的任務 5 只有 14%。

重要度矩陣（全部參數合計）在每個任務學完後的總和／最大值（第 5 個在最後才算出來，沒有用到）：

| 方法 | 總和（任務 1→5） | 最大值（任務 1→5） |
|---|---|---|
| EWC | 0.444 / 0.609 / 1.19 / 2.41 / 4.77 | 3.76e-4 / 4.35e-4 / 8.29e-4 / 1.68e-3 / 3.33e-3 |
| MAS | 4.40e5 / 8.53e5 / 1.62e6 / 3.17e6 / 6.31e6 | 63.7 / 118 / 217 / 422 / 844 |
| SI | 265 / 478 / 705 / 953 / 1220 | 0.0162 / 0.0252 / 0.0321 / 0.0388 / 0.0447 |
| RWalk（Ω+F） | 135 / 233 / 308 / 359 / 397 | 0.0093 / 0.0133 / 0.0165 / 0.0198 / 0.0230 |
| SCP | 16.0 / 35.9 / 71.2 / 141 / 282 | 0.0211 / 0.0427 / 0.0804 / 0.156 / 0.313 |

觀察（待實驗確認）：EWC、MAS、SCP 的總和每個任務大約加倍，符合疑點 3 的「guard 重複累加」；EWC 的重要度只有 0.44，乘上 λ=100 仍很小，結果和 baseline 幾乎一樣；MAS 的重要度大 6 個數量級，λ=0.1 也壓不住，舊任務保住了，但新任務學不好（對角線 97.9 → 89.7）。各方法的 λ 與重要度量級沒有可比性。

## ch01 實測（資料，2026-10-08）
- `transforms.functional.rotate` 在 torchvision 0.26 的簽名：`interpolation=NEAREST, expand=False, center=None, fill=None`；正角度逆時針（第 2 列第 14 行的亮點轉 +90° 到 `[13, 2]`）。
- `ToTensor` 後：`torch.Size([1, 28, 28])`、float32、min 0.0、max 1.0；第一張訓練圖 label 5；沒有 mean/std 正規化。
- `torch.equal(Pad(28)(x), x)` 為 True。改成 `expand=True`：畫布 28/36/40/40/34，`Pad(28)` 的負 padding 裁回 28×28，五個角度都與 `expand=False` 的輸出 `torch.equal`。
- 五份訓練集前五個 label 都是 `[5, 0, 4, 1, 9]`（同一批圖）；`raw_folder` = `data/MNIST/MNIST/raw`。
- `len(DataLoader)`：訓練 469（最後一批 96 張）、測試 2（8192 + 1808）。
- 測試集前 2,000 張，旋轉後總亮度 / 原圖：0.9995（20°）、0.9988（40°）、1.0000（60°）、1.0057（80°）。
- 攤平 cosine similarity（同一張圖，轉 k° vs 0°）：1.000 / 0.604 / 0.408 / 0.323 / 0.293；參考：同 label 不同圖 0.520、隨機兩張 0.391（`torch.Generator().manual_seed(0)` 配對）。
- 樣本圖 `docs/HW14/img/tasks.png`：訓練集每個 label 第一次出現的索引 label0..9 = 1, 3, 5, 7, 2, 0, 13, 15, 17, 4。
- A 組 `A_baseline_s0` 最後 70.592，與參照版 baseline 完全相同（兩者都是種子 0 之後第一個跑的方法）。
