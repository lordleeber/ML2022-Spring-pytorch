# HW02 教材計畫

HW02（phoneme classification，LibriSpeech 音框分類）的 HTML 教材還沒開始寫。流程照 [docs/TEXTBOOK_WORKFLOW.md](../TEXTBOOK_WORKFLOW.md)，這份檔案只記 HW02 特有的事。開始寫之後，量到的數字放在同一個目錄的 `FACTS.md`。

## 狀態（2026-10-05）

- **全書完成**：index、outline、ch00–ch07、appendix，由本機 session 依序撰寫（使用者 2026-10-04 決定 HW02 不走雲端；這個 session 暫停冷讀，所以各章**沒有冷讀**，日後要補可以逐章跑 cold-read skill）。
- 驗收（2026-10-05）：11 頁 verify_book.py 全過、站內錨點全部存在、check_links 只報跨書連結（../HW01、../HW09，檔案實際存在）與 ch00 回指 outline（同 HW09 的寫法）、每頁一組內嵌 style/js、無 TODO。
- 實測全部在 FACTS.md；實驗原始資料 `docs/tools/hw02_ch07_runs.jsonl`。圖表用 dataviz 驗證過的配色，SVG 用 PyMuPDF 渲染目視檢查過。
- 使用者的決定：新版共用樣式；事實一次量完；ch07 縮小規格（batch 64 × 5 epoch）；不寫雲端 prompt。

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
