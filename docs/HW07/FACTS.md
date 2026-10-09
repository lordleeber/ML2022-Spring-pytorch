# HW07 實測事實（本機，2026-10-09）

環境：RTX PRO 4000 Blackwell 24 GB、torch 2.11.0+cu128、transformers 5.18.0、Python 3.12（共用 .venv）。

## Phase 0：程式還原與比對
- 官方 notebook：`~/poyi/GitHubPublic/ML2022-Spring/HW07/HW07.ipynb`。原版：bert-base-chinese、`num_epoch = 1`、lr 1e-4、batch 32、`from transformers import AdamW`（transformers 4.5.0）、訓練完才存檔。
- repo 原本的 train.py 已改成 luhua RoBERTa-large、3 epoch、`torch.optim.AdamW`、每個 epoch 存最佳；舊註解的 Kaggle 分數（使用者 2026-10-03 前後實測）：bert-base-chinese 0.49495、ckiplab/bert-base-chinese-qa 0.57160、luhua/chinese_pretrain_mrc_roberta_wwm_ext_large 0.60508、luhua/chinese_pretrain_mrc_macbert_large 0.53206、wptoux/albert-chinese-large-qa 0.55990。舊檔備份只在 scratchpad。
- 還原後：`HW07/train.py`（訓練＋dev＋存檔）、`test.py`（載入 saved_model、寫 result.csv）、`dataset.py`（read_data、QA_Dataset）、`postprocess.py`（evaluate，多一個 tokenizer 參數）、`legacy_adamw.py`（transformers 4.47.0 的舊 AdamW，演算法同 4.5.0）。
- 參照版：`docs/tools/hw07_make_ref.py` 把 notebook 程式格串成腳本（`!` 行註解掉、AdamW 改 import 舊版）。
- **訓練不可重現**：同一份 train.py 跑兩次，100 步後權重總和 -12803.044 vs -12797.275（loss 1.474 vs 1.471）。原因是 CUDA 上非決定性的運算（embedding 反傳的 atomic add 等）；`cudnn.deterministic = True` 不夠。
- `docs/tools/hw07_det.py`：開 `torch.use_deterministic_algorithms(True)`（需 `CUBLAS_WORKSPACE_CONFIG=:4096:8`）再 runpy 執行腳本，不改原檔。兩次 100 步完全相同（-12801.912）。
- **逐位元一致**（決定性模式）：參照版 vs train.py＋test.py，stdout 相同、result.csv 相同、saved_model 每個張量 `torch.equal`。
- 原版一次的結果（dev EM，即 Exact Match）：

  | 執行 | dev EM | 時間 |
  |---|---|---|
  | 參照版，預設（非決定性） | 0.445 | 全程 6:36（含測試） |
  | train.py＋test.py，預設 | 0.464 | train 334 s、test 73 s |
  | 參照版，決定性模式 | 0.416 | 413 s |
  | train.py＋test.py，決定性模式 | 0.416 | train 338 s、test 77 s |

  → 只因 GPU 非決定性，dev EM 就在 0.416–0.464 之間（差近 5 點）。Kaggle Simple 基準線 0.45139。
- 訓練每 100 步印的 loss／acc（決定性模式）：1.474/0.484、0.834/0.674、0.777/0.689、0.686/0.719、0.677/0.717、0.664/0.726、0.625/0.740、0.589/0.747、0.567/0.742。訓練 acc（start 與 end 都對）0.74，但 dev EM 只有 0.42：訓練視窗把答案放在中央（見下）。
- 991 步／epoch（31,690 ÷ 32 進位），約 4 it/s；峰值 RSS 3.1 GB；saved_model/model.safetensors 406,737,656 bytes。

## 資料（`docs/tools/hw07_data_facts.py` → `hw07_data_facts.txt`）
| | 文章 | 問題 | 文章 token 平均／p50／p99／max | 問題 token 平均／>40 被截 |
|---|---|---|---|---|
| train | 10,524 | 31,690 | 390.2／355／860／1679 | 20.7／596 |
| dev | 1,490 | 4,131 | 414.4／407／708／1126 | 20.5／114 |
| test | 1,586 | 4,957 | 416.7／396／828／950 | 22.3／148 |
- 幾乎所有文章都超過 150 token（train 0.9999、dev 1.0）；超過 512 token 的文章 train 1,548、dev 199、test 294。
- 測試集沒有答案：前 3,493 題答案欄是 None，後 1,464 題是字串 `'null'`，文字有語音辨識錯字（「梵語」→「當犯人」），即 ODSQA 的口語部分。本機只能量 dev。
- 每題視窗數（dev）：stride 150 平均 3.32（最多 8）、100 → 4.75、75 → 6.17、50 → 8.99、32 → 13.79。
- 答案：dev 平均 5.55 字（p99 23、max 52）；train 4.58 字（max 118）。`answer_text` 與原文切片全部一致；char_to_token 沒有 None。
- **答案不完整落在任何一個評估視窗內**（無法答對）：stride 150 → dev 74 題（1.79%）、train 431；stride 100 → 0；stride 75 → dev 0（train 1，因為 115 token 的長答案）。
- **decode 還原不了答案**：dev 44 題（1.07%），其中 39 題含 `[UNK]`（英文字母、罕用字：`'Duff Roblin' → '[UNK][UNK]'`、`'朱允炆' → '朱允[UNK]'`）；其餘是數字併成大 token（train 例：`'12' → '128'`）。train 311 題（0.98%）。這是範例「postprocessing 有 bug」的來源之一。
- `[UNK]` 出現在問題：train 718、dev 149、test 104；在文章：train 2,823、dev 473、test 436。
