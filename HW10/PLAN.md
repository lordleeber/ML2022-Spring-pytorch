# HW10 對抗攻擊（Adversarial Attack）— 研究筆記

> 2026-10-08 由前一個 session 整理。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。候選（本機可量、實驗便宜）。

## 題目（本資料夾 `HW10.ipynb`、`HW10.pdf`（31 頁），從官方 repo 複製）
- 200 張 CIFAR-10（32×32，10 類各 20 張），對每張產生對抗樣本，**L∞ ≤ ε = 8**（0–255 尺度；程式裡要換算成 8/(255·std)，因為先 ToTensor 再 Normalize）。
- 評分：助教用看不到的黑箱模型分類你交的 200 張，**準確率越低越好**。基準線：Simple ≤ 0.70（FGSM）、Medium ≤ 0.50（Ensemble＋隨機幾個模型＋I-FGSM）、Strong ≤ 0.30（Ensemble＋挑對代理模型〔paper B: Query-Free Adversarial Transfer via Undertrained Surrogates〕＋I-FGSM，或 Ensemble＋很多模型＋MI-FGSM）、Boss ≤ 0.15（Ensemble＋paper B＋DIM-MI-FGSM）。每條都應在 20 分鐘內完成。
- 代理模型：`pytorchcv` 的 `*_cifar10` 預訓練模型（作業**允許**用預訓練模型）。
- 交 `.tgz`（< 2MB），`<class>/<class><id>.png` 結構。
- 報告：攻擊（怎麼產生可轉移的雜訊、JudgeBoi 準確率）；防禦（`resnet110_cifar10` 對 `dog/dog2.png` 做 vanilla FGSM：預測錯了嗎、變成哪類；再實作 JPEG 壓縮〔imgaug 的 compression=70，換算成 PIL 品質 31〕當前處理，預測恢復了嗎）。

## 本機狀態
- 本資料夾只有官方 notebook 與投影片，**還沒拆成 .py**。
- 資料：`/mnt/c/Users/valtec/Documents/poyi/ml_2022_data/ml2022spring-hw10.zip`。
- `.venv` 沒有 `pytorchcv`、`imgaug`；代理模型權重要從網路下載（模型權重，不是資料，但仍要先問使用者）。

## 為什麼值得做
- 本機可量：把一部分 pytorchcv 模型當代理、另一部分當「本機黑箱」量轉移性。
- 200 張小圖，每組實驗分鐘級，可大量並排比較。
- 接得上 HW03（CNN）、HW09（梯度）、HW06 第 7 章（對輸入取梯度）。
- 對應 ML2026 HW1「LLM Malicious Instruction Defense」（從騙圖片分類器到 LLM 越獄）。

## 風險
- 要建資料夾、拆 notebook、裝 pytorchcv、下載多個權重；imgaug 可能和新版 NumPy 不相容（未測）。

## 狀態（2026-10-10 Phase 0）
- 使用者 2026-10-10 選定 HW10，同意下載 pytorchcv 權重。pytorchcv 0.0.74 裝進共用 .venv；imgaug 0.4.0 與 opencv-python-headless<5、shapely 裝在 scratchpad（`--target`），NumPy 2 需要 `docs/tools/hw10_npshim_sitecustomize.py`（補回 `np.sctypes`）。
- notebook 拆成 `config.py`、`dataset.py`、`attack.py`（mifgsm 的 TODO 已補）、`ensemble.py`（TODO 已補，logits 相加）、`hw10.py`、`report.py`（JPEG TODO 已補）。參照版 `docs/tools/hw10_make_ref.py`。
- **GPU 上預設不可重現**：兩次跑參照版，fgsm/ 有 28 張 PNG 不同（cuDNN 反向）；`docs/tools/hw10_det.py` 決定性模式下兩次完全相同，hw10.py 與參照版逐位元一致，`docs/tools/hw10_exp.py` 與 hw10.py 逐位元一致（fgsm、ifgsm）。
- Baseline（resnet110 白箱，決定性模式）：benign 0.95000／0.22679；印出 fgsm 0.59500、ifgsm 0.00500；存成 PNG 後重量 fgsm 0.59000、ifgsm 0.00500；L∞ = 8。
- 報告題：dog2 benign dog 99.64% → fgsm cat 72.20% → JPEG(70) dog 99.24%。imgaug compression=70 = PIL quality 31（不是品質 70）；工具內的 `jpeg()` 與 imgaug 800 組逐位元相同。
- 模型庫（`docs/tools/hw10_zoo.jsonl`）：85 個 `*_cifar10`，70 個能載入（benign 0.915–0.98），13 個下載 404、2 個沒有預訓練權重；權重共 2.5 GB 在 `~/.torch/models`。
- 試跑（`docs/tools/hw10_pilot.jsonl`，受害池 8 個與代理不同家族的模型）：resnet110 單一代理 FGSM 0.627 → I-FGSM 0.492 → MI 0.341 → DIM-MI 0.259；6 模型 ensemble I-FGSM 0.057；但受害者前面加 JPEG70 後全部回到 0.51–0.65。

## 使用者決定（2026-10-10）
- 大綱核可：index、ch00 全貌／ch01 資料與 ε／ch02 FGSM／ch03 本機黑箱／ch04 I-FGSM 與過擬合／ch05 Ensemble（含 paper B）／ch06 MI-FGSM 與 DIM／ch07 防禦（JPEG）／ch08 總結；不寫附錄、不冷讀、樣板自己設計。
- paper B 要實測：下載 CIFAR-10 訓練集（torchvision），自己訓練「訓練不足」的代理模型。
- 受害池依家族分開：代理用 resnet／preresnet／se*／densenet／nin；受害者 8 個（wrn28_10、wrn40_8、pyramidnet110_a48、resnext29_32x4d、ror3_110、rir、shakeshakeresnet26_2x32d、diaresnet56），各量無防禦與 JPEG70。
- 節奏照 HW07：每章寫完驗證就推、接著寫下一章；規格內的 GPU 實驗一組一組自己排。長任務每 30 分鐘回報。

## 狀態（2026-10-10 14:50）
- 已推上 master：index、outline、ch00–ch04（樣板在 docs/tools/hw10_book/，珊瑚紅；圖表 make_charts.py）。
- 實驗：A（98 組）完成；B（ensemble 19 組）、C（MI／DIM 34 組）依序在跑（hw10_queue.sh）；U：resnet20／56 × 3 種子從頭訓練中（hw10_queue_U.sh，C 完成後跑 66 組 checkpoint 攻擊）。B、C、U 的 attack_s 因與訓練並行而不乾淨，計時最後另外量。
- 作業 200 張 = CIFAR-10 測試集每類前 20 張（hw10_overlap.json）；CIFAR-10 在 HW10/cifar10/（gitignore），checkpoint 在 HW10/surrogates/（gitignore）。
- ch07 工具已寫好：hw10_bpda.py（JPEG 前向、反向恆等）、hw10_jpeg_sweep.py；D 組規格等 B、C、U 結果出來再定。
