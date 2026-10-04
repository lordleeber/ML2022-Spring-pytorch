# HW02 教材計畫

HW02（phoneme classification，LibriSpeech 音框分類）的 HTML 教材還沒開始寫。流程照 [docs/TEXTBOOK_WORKFLOW.md](../TEXTBOOK_WORKFLOW.md)，這份檔案只記 HW02 特有的事。開始寫之後，量到的數字放在同一個目錄的 `FACTS.md`。

## 狀態（2026-10-04）

- Phase 0 完成：大綱 `outline.html`（ch00–ch07 + appendix）使用者已核可；驗證指標沒有偏差（印出 0.642 = 整個驗證集一次算完 0.641955，見 FACTS「驗證指標檢查」）。
- 使用者的決定：用新版共用樣式；事實一次量完；ch07 用同一套縮小規格（batch 64 × 5 epoch）並排比，另附一次原規格 20 epoch 的完整紀錄。
- Phase 1：style/enhance、FACTS.md、`docs/tools/hw02_exp.py`（已驗證與 train.py 逐位一致）、`hw02_facts.py`、`hw02_run_grid.sh` + `hw02_ch07_runs.txt` 已建立。
- 進行中：原規格 baseline（20 epoch，約 4–5 小時）與 ch07 實驗清單在本機背景執行；跑完補進 FACTS 的「執行實測」「ch07 實測」，以及峰值 RSS、predict.py 的時間與輸出。
- 下一步：baseline 數字補齊後，給雲端寫 index + ch00 的 prompt。

## Phase 0 要做的事

1. 讀 HW02/ 的程式：config.py、data_loader.py、model.py、model_dnn.py、train.py、utils.py、predict.py，共約 400 行。
   - 有兩個模型檔（model.py、model_dnn.py），先弄清楚 train.py 和 predict.py 用的是哪一個，另一個是做什麼的。
2. 跑一次 baseline，量出訓練要多久。資料量比 HW01 大很多，訓練可能慢得多（還沒量過），這決定了 ch07 那類改良實驗能跑幾組、要不要多跑幾個 seed。
3. **檢查驗證指標有沒有偏差**（HW01 最大的教訓）：train.py 印出的 val acc，和最佳 checkpoint 在整個驗證集上一次算完的 accuracy 是否一致？要看 valid loader 有沒有打亂、是不是各 batch 平均再平均、最佳是不是從很多次量測裡挑出來的。有偏差的話，從 ch00 起就用真實的數字。
4. 讀作業說明 `HW02/hw2_slides 2022.pdf`：作業提示、Kaggle 基準線（HW01 的基準線在 PDF 的截圖裡，要把圖片抽出來看）。
5. 寫大綱，等使用者核可。

## HW02 特有的注意事項

- **資料不在 repo 裡**：`HW02/libriphone/`（約 511M）被 gitignore 排除，來源見本機的資料位置說明。雲端完全看不到資料，所以「看資料」的章節內容全部要靠 FACTS，資料相關的實測要做得比 HW01 更完整。
- **評估指標是 accuracy**，不是 MSE。HW01 那套「印出的值 vs 一次算完的真實值」的檢查要改寫成 accuracy 的版本。
- **大檔案**：`HW02/model.ckpt` 約 80M（沒有被 git 追蹤）；`HW02/prediction.csv` 約 6M，**有被 git 追蹤**，重新執行 predict.py 會改到它。實驗時要在複本裡跑，或事後還原。
- **實驗工具**：照 HW01 的 `docs/tools/hw01_exp.py` 寫一份 HW02 版，先驗證它能逐位元重現 train.py。

## 待使用者決定

- **樣式版本**：用新版共用樣式（有 Python 上色與 `pre.shell.cmd`，HW09 選了這個），還是照抄 HW01 的舊版？建議用新版。
- **事實要分章量，還是一次量完**：HW01 是分章量，HW09 是一次量完。一次量完的話，雲端可以連續寫好幾章；HW02 要等 Phase 0 量完訓練時間後再決定。
