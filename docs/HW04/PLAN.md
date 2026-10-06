# HW04 教材計畫

HW04（Speaker Classification，用 Transformer 從 mel-spectrogram 辨認 600 位說話者）的 HTML 教材還沒開始寫，預定在新的本機 session 開始。流程照 [docs/TEXTBOOK_WORKFLOW.md](../TEXTBOOK_WORKFLOW.md)，特別是「HW02 的調整」「HW03 的調整」兩節（本機一章一章寫、事實一次量完、只在使用者明講時才推）；這份檔案只記 HW04 特有的事。開始寫之後，量到的數字放在同一個目錄的 `FACTS.md`。

## 狀態（2026-10-06）

- **Phase 0、Phase 1 完成**（同一個本機 session）。使用者決定：照大綱（index、ch00–ch08、appendix）、新版共用樣式、事實一次量完、ch08 每組都跑**原規格 70,000 步**（另加一組 210,000 步長跑）、放少量真實 mel 熱圖、**這本書不冷讀**。
- 事實全部在 `FACTS.md`；實驗工具 `docs/tools/hw04_exp.py`（與 train.py 逐位一致）、`hw04_facts.py`、`hw04_run_grid.sh`、`hw04_ch08_runs.jsonl/.txt`。
- 下面「已經看到的疑點」的實測結論：1 成立（第 60k、70k 次存檔存的不是最佳，但影響 ≤ 1 句）；2 drop_last 只丟 3 句、平均再平均沒有偏差，**真正的偏差是驗證切 128 格、測試用整句**（0.686 vs 0.857）；3 單層不印警告，打開 2 層才印 nested tensor 警告；4 8 個 worker 下逐位元可重現；5 確認；6 一個 epoch 1,593 步、70,000 步 = 43.94 epoch；7 只有 2 句短於 128 格（都在訓練集）；8 確認。
- baseline 只要 3 分 53 秒（Colab 估 30–40 分）。
- ch00–ch05 已推；ch06 寫完（本機 commit）。下一步：ch07（一章一停，使用者說「推」才推）。
- 2026-10-06 scratchpad 被清空：grid／snaps 的 ckpt 都沒了，需要時用 `hw04_exp.py --save/--save_live/--snap_dir` 重跑（每組約 4 分鐘，逐位元相同）。

## 投影片重點（只讀了文字，圖還沒看）

- 基準線（public）：Simple 0.60824（照跑範例程式，Colab 30–40 分）、Medium 0.70375（調 Transformer 的參數，1–1.5 小時）、Strong 0.77750（改成 Conformer，3–4 小時）、Boss 0.86500（Self-Attention Pooling + Additive Margin Softmax，Kaggle 2–2.5 小時）。
- 計分：四條基準線 public/private 各 0.5、code 2、report 4。
- 報告兩題（p.22）：1. 簡介一種 Transformer 的變體；2. 說明為什麼在 Transformer 加卷積層能提升表現。兩題都是**文字題**，不像 HW03 要交程式碼。
- p.15、p.17、p.18 的提示是圖片（Conformer、Self-Attention Pooling、AMSoftmax），要用 PyMuPDF 把頁面轉成圖片看。

## Phase 0 要做的事

1. 讀 HW04/ 的程式，並和官方原版比對：`~/poyi/GitHubPublic/ML2022-Spring/HW04/`（notebook）。列出本 repo 改了什麼。
2. 跑一次 baseline 並計時（70,000 步；GPU 上不能有其他程式，開跑前後用 `nvidia-smi --query-compute-apps` 記錄）。先估算時間再決定要不要整個跑完。**問使用者放行**，一次只跑一個。
3. **驗證指標與存檔的檢查**（下面「已經看到的疑點」1、2），用最佳 checkpoint 在整個驗證集上一次算完來比對。
4. 讀作業 PDF：基準線、提示的圖、報告題；本機沒有 poppler，用 `uv pip install --target <scratch>/pylib pymupdf`。
5. 寫大綱、問使用者決定（樣式版本、事實分章量或一次量完、改良實驗的規格、要不要放圖），等核可。寫作方式已定（本機一章一章寫），不用再問。

## 已經看到的疑點（Phase 0 要實測確認）

1. **「最佳」checkpoint 可能不是最佳的**：`train.py:297` 是 `best_state_dict = model.state_dict()`，沒有 `copy.deepcopy`。`state_dict()` 回傳的是參數 tensor 的參照，之後的訓練會繼續改動它們，所以第 303 行每 10,000 步存檔時，存下的可能是**存檔當下**的權重，而不是驗證準確率最高時的權重。要實測：存下的 `model.ckpt` 在驗證集上的準確率，是否等於 log 印的 `best_accuracy`。
2. **驗證集 `drop_last=True`**（`train.py:101`）：驗證時最後不滿一批的語句被丟掉，`valid()` 的準確率又是各 batch 平均再平均（`train.py:219`），而且 `pbar` 用 `dataloader.batch_size` 累加。和 HW01／HW03 一樣要檢查印出值與真實值的差距。
3. **只有一層 Transformer**：`classifier.py:38-41` 建的是單一個 `TransformerEncoderLayer`，第 41 行 `TransformerEncoder(..., num_layers=2)` 被註解掉；`nhead=2`、`dim_feedforward=256`、`d_model=80`。`batch_first` 沒設，所以 forward 裡要先 `permute`（第 60 行）。PyTorch 2.x 可能會對這種寫法印警告，要看實際輸出。
4. **亂數有兩個來源**：`dataset.py:42` 用 Python 的 `random.randint` 隨機取 128 格的片段，`train.py:231` 開 8 個 worker；`random_split`（`train.py:86`）用的是 PyTorch 全域亂數。要確認結果能不能逐位元組重現（PyTorch 會替每個 worker 設 Python `random` 的種子，但要實測）。
5. **tqdm 用的是 `from tqdm import tqdm`**（標準版，不是 HW03 的 `tqdm.auto`），而且 `train.py:260` 是手動 `update()`，不是包住 DataLoader。HW03 那種「進度條改變亂數順序」的問題在這裡應該不會發生，寫實驗工具時仍要驗證逐位元一致。
6. **訓練是以 step 為單位**（`total_steps=70000`、每 2,000 步驗證一次、每 10,000 步存檔），不是 epoch；`train_iterator` 用完就重建（第 264-268 行）。一個 epoch 有幾步、70,000 步是幾個 epoch，要算出來。
7. **padding 值 -20**（`train.py:74`）：訓練時用 `pad_sequence` 補到同一批最長的長度，但 `myDataset` 已經把長句切成 128 格，所以只有短於 128 格的句子會被補；mean pooling（`classifier.py:66`）會把補的 -20 也平均進去。測試時 `batch_size=1`（`test.py:62`）不補。要看有多少句短於 128 格。
8. **說明文字重複**：`classifier.py:1-23` 與 `train.py:109-131` 是同一段 Model 說明；`train.py:169` 又 `import torch` 一次。
9. `device = "cuda"` 寫死是刻意的，不列為問題。

## HW04 特有的注意事項

- 資料是 VoxCeleb2 的子集（授權 CC BY 4.0，見 `train.py:37-38`），已經轉成 mel-spectrogram，沒有原始聲音；教材「看資料」的章節以形狀、長度分布、說話者分布為主，可以畫 mel-spectrogram 的圖。
- 一次完整訓練（70,000 步）可能要一小時以上，改良實驗的規格（縮小步數並排比、另附一次完整紀錄）要在 Phase 0 量完時間後和使用者一起決定。
- 實驗產生的 checkpoint 放在 scratchpad，session 結束就沒了；要留的數字一定要寫進 FACTS 或 `docs/tools/hw04_*` 的紀錄檔。
