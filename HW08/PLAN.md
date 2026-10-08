# HW08 非監督異常偵測（Autoencoder）— 研究筆記

> 2026-10-08 由前一個 session 整理。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。**目前不建議**：本機量不到主指標，且和 HW06 重疊多。

## 題目（本資料夾 `Homework 8 Anomaly Detection.pdf`，21 頁）
- 訓練只給正常人臉，測試時給每張圖一個異常分數；方法是用 autoencoder 的**重建誤差**當分數。
- 資料（`data/`，gitignore）：`trainingset.npy` (100000, 64, 64, 3) uint8；`testingset.npy` (19636, 64, 64, 3)，約一半正常（label 0）、一半異常（label 1），**標籤不公開**。
- 評分：Kaggle ROC AUC（不需門檻）。基準線**投影片沒寫分數**：Simple 照跑、Medium 調模型結構、Strong multi-encoder autoencoder、Boss 加隨機雜訊＋額外分類器（real/fake）或論文方法。
- 報告：1) 介紹 VAE，一個優點、一個問題；2) 訓練全連接 autoencoder，調整 latent 至少兩個維度，畫原圖與重建圖並描述差別。

## 本機狀態
- 6 個 .py 共 354 行：`fcn_autoencoder.py`、`conv_autoencoder.py`（3 層 stride-2 conv 到 48×8×8，3 層轉置卷積＋Tanh）、`vae.py`、`dataset.py`（[0,255]→[-1,1]）、`train.py`（目前 `cnn`、50 epoch、batch 2000、Adam 1e-3、MSE）、`test.py`（用 `last_model_cnn.pt`，誤差加總開根號 → `prediction.csv`）。`test.py` 的選項列了 `resnet` 但沒有實作。
- 已有 2026-10-03 的 `best_model_cnn.pt`、`last_model_cnn.pt`、`prediction.csv`。`381aad0` 修過 `torch.load(..., weights_only=False)`（存的是整個模型物件）。

## 已看到的問題
1. **本機無法算 AUC**（測試集無標籤）——要自造代用的尺（例如合成異常），結論打折。
2. `train.py` 用**訓練 loss** 挑 best model，與異常偵測能力無關；`test.py` 又用 last。
3. autoencoder 太強時連異常圖都重建得好，分數反而變差（好教的反直覺現象）。

## 2026 視角
- 異常偵測任務本身仍大量使用（工業瑕疵、詐欺、資安、醫療、時間序列）。圖像主流已轉向預訓練特徵比對（PaDiM、PatchCore、EfficientAD、CLIP 類 WinCLIP/AnomalyCLIP；Intel Anomalib；MVTec AD），autoencoder 重建誤差多作為基線（寫進書前要查證）。
