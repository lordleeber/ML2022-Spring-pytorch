# HW10 對抗攻擊（Adversarial Attack）— 研究筆記

> 2026-10-08 由前一個 session 整理。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。候選（本機可量、實驗便宜）。

## 題目（本資料夾 `HW10.ipynb`、`HW10.pdf`（31 頁），從官方 repo 複製）
- 200 張 CIFAR-10（32×32，10 類各 20 張），對每張產生對抗樣本，**L∞ ≤ ε = 8**（0–255 尺度；程式裡要換算成 8/(255·std)，因為先 ToTensor 再 Normalize）。
- 評分：助教用看不到的黑箱模型分類你交的 200 張，**準確率越低越好**。基準線：Simple ≤ 0.70（FGSM）、Medium ≤ 0.50（Ensemble＋隨機幾個模型＋I-FGSM）、Strong ≤ 0.30（Ensemble＋挑對代理模型〔paper B: Query-Free Adversarial Transfer via Undertrained Surrogates〕＋I-FGSM，或 Ensemble＋很多模型＋MI-FGSM）、Boss ≤ 0.15（Ensemble＋paper B＋DIM-MI-FGSM）。每條都應在 20 分鐘內完成。
- 代理模型：`pytorchcv` 的 `*_cifar10` 預訓練模型（作業**允許**用預訓練模型）。
- 交 `.tgz`（< 2MB），`<class>/<class><id>.png` 結構。
- 報告：攻擊（怎麼產生可轉移的雜訊、JudgeBoi 準確率）；防禦（`resnet110_cifar10` 對 `dog/dog2.png` 做 vanilla FGSM：預測錯了嗎、變成哪類；再實作 JPEG 壓縮〔品質 70，imgaug〕當前處理，預測恢復了嗎）。

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
