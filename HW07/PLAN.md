# HW07 BERT 抽取式問答 — 研究筆記與計畫

> 2026-10-08 由前一個 session 整理，尚未開始寫書。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。排序第三（原本第一，看過 ML2025／2026 後下調）。

## 題目（官方 `~/poyi/GitHubPublic/ML2022-Spring/HW07/HW07.ipynb`、`HW07.pdf`，34 頁）
- 給一段中文文章和問題，從文章裡**框出一段**當答案（extractive QA）。
- 資料：DRCD（台達閱讀理解資料集）＋DRCD-TTS（train／dev），測試集另含 ODSQA。本資料夾已有：
  | 檔 | 文章 | 問題 |
  |---|---|---|
  | `hw7_train.json` | 10,524 | 31,690 |
  | `hw7_dev.json` | 1,490 | 4,131（**有答案，本機可量 EM**） |
  | `hw7_test.json` | 1,586 | 4,957 |
  每題有 `answer_text`、`answer_start`、`answer_end`。
- 評分：Kaggle，Exact Match。基準線：Simple 0.45139、Medium 0.65792、Strong 0.78136、Boss 0.84388。
- 提示：Medium＝線性學習率衰減＋調 `doc_stride`（視窗重疊）；Strong＝改善前處理（答案不要總在視窗中央）＋換預訓練模型；Boss＝改善後處理（end < start 等）。另建議 fp16、梯度累積、ensemble。
- 報告：1) 你用什麼規則從 start／end 機率決定答案範圍；2) 換一個 HF 預訓練模型，說明它與 BERT 的差別。

## 本機狀態（2026-10-08 檢查）
- `train.py` 200 行、`dataset.py` 86 行、`test.py` 94 行。**已不是原版**：模型換成 `luhua/chinese_pretrain_mrc_roberta_wwm_ext_large`，lr 1e-4、batch 32、3 個 epoch；註解記了使用者試過的 Kaggle 分數：原版 bert-base-chinese 0.49495、ckiplab 0.57160、**luhua RoBERTa 0.60508（目前）**、macbert 0.53206、albert 0.55990。
- `dataset.py`：`max_question_len` 40、`max_paragraph_len` 150、`doc_stride` 150（= 段落長度，視窗不重疊）。前處理、後處理仍是範例程式的寫法。
- `saved_model/`（model.safetensors 1,297,946,304 bytes）與 `result.csv` 是 2026-10-03 的結果。`381aad0` 修過 AdamW（改從 torch.optim）與 test.py 的載入方式。
- transformers 5.18.0 在 .venv；HF 快取已有 `bert-base-chinese`、`luhua/chinese_pretrain_mrc_roberta_wwm_ext_large`（2.5 GB）。

## 要先決定／查證
1. 書的主線用原版 bert-base-chinese（容易教、可重現 Simple→Boss），還是 repo 現在的 luhua large？
2. luhua 這類「已用閱讀理解資料預訓練」的模型，訓練資料是否包含 DRCD（會影響「換模型為何變好」的解讀）。
3. large 模型一次訓練多久（要量）。

## 2026 視角
- 抽取式問答已被 LLM 生成式問答＋RAG 取代；BERT 類 encoder 仍大量用於 embedding、reranker、分類（ModernBERT 等，寫進書前要查證）。
- 延續的觀念：tokenizer、預訓練＋微調、長文切塊（= RAG chunking）、後處理規則。ML2025 有 HW5「Fine tune is powerful」，ML2026 沒有直接對應。

## 使用者決定（2026-10-09 晚）
- **主線還原成官方範例**：train.py／dataset.py／test.py 改回 notebook 原版（bert-base-chinese、1 epoch、最後才存檔），做參照版逐位元比對；Simple→Medium→Strong→Boss 的改良當實驗；luhua 等換模型放在「換預訓練模型」那章，repo 舊註解裡的 Kaggle 分數當史料引用。
- 節奏照 HW13：大綱核可後每章寫完驗證就推、接著寫下一章；規格內的 GPU 實驗直接一組一組跑；不冷讀；長任務每 30 分鐘回報。
- 舊的 luhua 結果（HW07/saved_model、result.csv，2026-10-03）不動；實驗都在 scratchpad 的複本目錄跑。

## Phase 0 紀錄
- 現行 skill 與 HW13 時相同（mySkills 27d96f9）。
- notebook 用 `from transformers import AdamW`（transformers 4.5.0）；transformers 5.x 已刪除。舊實作的 eps 預設 1e-6、weight_decay 0，而且 eps 加在偏差修正之前，和 `torch.optim.AdamW`（eps 1e-8、weight_decay 0.01）不同。381aad0 改成 torch 版時改變了行為。還原版用 `HW07/legacy_adamw.py`（抄自本機 ML2025 .venv 的 transformers 4.47.0 `optimization.py`，演算法與 4.5.0 相同；4.5.0 原檔未逐字核對）。
- 原版一次 dev EM 0.416–0.464（只因 GPU 非決定性）；決定性模式下 train.py＋test.py 與參照版逐位元一致。細節在 docs/HW07/FACTS.md。

## 使用者決定（2026-10-09 晚，大綱核可）
- 大綱：index（2022 vs 現在）、ch00 全貌／ch01 tokenizer／ch02 Dataset 與視窗／ch03 模型與訓練迴圈／ch04 評估與後處理（Boss）／ch05 Medium（LR 衰減、doc_stride、epoch）／ch06 Strong 前處理（答案置中的偏差、隨機視窗）／ch07 換預訓練模型＋fp16／梯度累積／ch08 總結。不寫附錄、不冷讀、樣板自己設計。
- **全書用繁體中文（台灣用語）**。
- 實驗：決定性模式、種子 0/1/2。N 原版×3；P 後處理與評估 stride（用現成 checkpoint）；M LR 衰減、1/2/3 epoch ×3 種子；S 隨機視窗 ×3 種子；D 換模型。
- D 組可下載：hfl/chinese-roberta-wwm-ext、hfl/chinese-macbert-base、ckiplab/bert-base-chinese-qa（加上快取裡的 luhua large）。
