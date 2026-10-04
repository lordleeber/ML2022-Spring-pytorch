# HW03 教材計畫

HW03（Image Classification，food-11 食物分類 CNN）的 HTML 教材還沒開始寫，預定在新的本機 session 開始。流程照 [docs/TEXTBOOK_WORKFLOW.md](../TEXTBOOK_WORKFLOW.md)，特別是「HW02 的調整」一節（本機一章一章寫、事實一次量完）；這份檔案只記 HW03 特有的事。開始寫之後，量到的數字放在同一個目錄的 `FACTS.md`。

## 狀態（2026-10-05）

- 還沒開始。下一步是 Phase 0。
- 程式：`HW03/` 下 6 個檔共 435 行（config.py 4、dataset.py 43、classifier.py 49、others.py 97、train.py 196、test.py 46）。作業說明 `HW03/Machine Learning HW3 - Image Classification.pdf`（已在 repo）。
- 2026-10-03 在共用 .venv 跑過一次：5 個 epoch 後 val acc 0.555。
- commit `381aad0` 修過 `dataset.py`：標籤原本用 Windows 的 `"\\"` 切路徑，改成 `os.path.basename`。這是「為了在 2026 年跑起來而改的地方」，寫進 index 的「2022 vs 現在」。

## Phase 0 要做的事

1. 讀 HW03/ 的程式，弄清楚 `others.py` 是做什麼的、`train.py` 和 `test.py` 用到哪些檔案。
2. 跑一次 baseline 並計時（照 HW02 的教訓：計時那次 GPU 上不能有任何其他程式，開跑前用 `nvidia-smi --query-compute-apps` 確認並記下來）。
3. **檢查驗證指標有沒有偏差**：`train.py:64` 的 `valid_loader` 是 **`shuffle=True`**，這是 HW01 偏差的同一種寫法。要看 valid acc／loss 是不是「各 batch 平均再平均」、最後一批多大、最佳是不是從很多次量測裡挑出來的，並用最佳 checkpoint 在整個驗證集上一次算完比對。有偏差的話，從 ch00 起就用真實的數字。
4. 讀作業 PDF：Kaggle 基準線、作業提示、報告題（基準線若在截圖裡，要把圖片抽出來看；本機沒有 poppler，用 `uv pip install --target <scratch>/pylib pymupdf`）。
5. 和官方原版比對：`~/poyi/GitHubPublic/ML2022-Spring/HW03/`。
6. 寫大綱、問使用者三個決定（樣式版本、事實分章量或一次量完、改良實驗的規格），等核可。

## HW03 特有的注意事項

- **資料不在 repo 裡**：`HW03/food11/`（約 1.2 G）被 `.gitignore` 排除，來源是本機 Kaggle zip（`hw3b.zip`，見本機的資料位置說明）。
- `.gitignore` 也排除了 `*.ckpt`、`*.csv`、`sample_*.txt`，所以 `sample_best.ckpt`（約 49 M）、`submission.csv` 沒有被追蹤（和 HW02 的 prediction.csv 不同）。實驗仍建議在複本裡跑。
- 圖片資料：教材裡「看資料」的章節可以放真實圖片的縮圖（參考 HW09 的 `img/` 做法），但要確認授權與檔案大小。
- `device = "cuda"` 寫死是刻意的，不列為問題。
