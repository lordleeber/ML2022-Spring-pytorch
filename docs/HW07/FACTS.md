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

## 投影片（`~/poyi/GitHubPublic/ML2022-Spring/HW07/HW07.pdf`，34 頁）
- p.6 資料：train = DRCD + DRCD-TTS（10,524 文章／31,690 題）；dev = DRCD + DRCD-TTS（1,490／4,131）；test = DRCD + ODSQA（1,586／4,957）。p.7「Testing: No answer!」。p.8 範例文章（新加坡、馬來西亞的簡繁體）。
- p.10 tokenization 範例：'李宏毅教授2022機器學習' → ['李','宏','毅','教','授','2022','機','器','學','習'] → [3330, 2131, 3675, 3136, 2956, 10550, 3582, 1690, 2119, 5424]。p.11 input_ids／token_type_ids／attention_mask。
- p.12 BERT 上限 512、self-attention O(n²)。p.13 訓練：在答案附近畫視窗（假設：答題需要的資訊在答案附近）。p.14 測試：切視窗，每個視窗 start score＋end score 取最大，表中 window 2 總分 1.0 勝出。
- p.16 提示與估計時間（K80／T4／T4 fp16／P100／V100）：Simple、Medium 40m／20m／8m／10m／7m；Strong 2h／1h／25m／35m／20m；Boss 12.5h／6h／2.5h／4.5h／2h。訓練技巧：fp16、梯度累積、ensemble。
- p.17 線性衰減：手動減 `optimizer.param_groups[0]["lr"]`，或 scheduler（推薦 huggingface）；「檢查訓練完 lr 是否非常接近 0」。p.18 doc_stride（範例 = max_paragraph_len，不重疊；提示：重疊視窗）。p.19 前處理：答案不要總在視窗中間。p.20 只能用 huggingface 上的預訓練模型（違規學期成績 ×0.9）。p.21 後處理：打開預測檔看錯在哪（end_index < start_index）。p.22 AMP 約 1.5–3.0 倍。p.23 梯度累積。
- p.25 評分：報告 4、程式 2、四條基準線 public／private 各 0.5。p.26 Kaggle：4,957 題（public／private 約各半），Exact Match；基準線 0.45139／0.65792／0.78136／0.84388；一天最多 5 次（p.32）。
- p.30 報告：(1) 2% 從 start／end 機率決定最終位置的規則（要和範例不同）；(2) 2% 換一個 HF 上的預訓練模型，說明模型、表現、與 BERT 的差別（架構、預訓練 loss 等）。
- p.32 規則：不准額外資料、不准 huggingface 以外的預訓練模型、不准手改預測檔。

## ch00 實測（`docs/tools/hw07_ch00.py` → `hw07_ch00.txt`，用決定性模式 base 種子 0 的 checkpoint，CPU）
- 參數：embeddings 16,622,592；encoder 12 層 85,054,464（每層 7,087,872）；qa_outputs 1,538（Linear 768→2）；合計 101,678,594。BertForQuestionAnswering 沒有 pooler。
- 載入時 transformers 5 印「LOAD REPORT」：UNEXPECTED 9 個（`cls.*` 預訓練頭、`bert.pooler.*`）、MISSING 2 個（`qa_outputs.weight/bias`，隨機初始化）。
- 決定性模式的時間：訓練 991 步 4:31（3.65 it/s）、dev 4,131 題 0:59；非決定性：4:26、0:56；test 4,957 題 1:07。
- dev 前 8 題（seed 0 決定性 checkpoint）：0 福岡（gold 天神地區，預測**空字串**）；1 白蓮教 失敗 ✓；2 摔角 職業摔角比賽 ✓；3 1960 重點大學（gold 64所，預測 華南理工大學）；4 權勢象徵（gold 納妾制度，預測 中共將其看作統戰工作）；5 中華民國首都（gold 臺北市，預測 菲律賓）；6 微積分（gold 17世紀，預測 1670年）；7 幕府（鐮倉幕府 ✓）。dev 每題 3–4 個視窗，start_logits 形狀 (視窗數, 193)。
- dev 第 0 題拆解（同一個 checkpoint）：文章 460 token → 4 個視窗（[0,150)、[150,300)、[300,450)、[450,460)，最後一個只有 10 個文章 token）；答案「天神地區」在 token 337–340，落在第 3 個視窗（k=2）。各視窗 (start, end, start_logit, end_logit, 和)：k=0 (44, 33, 1.82, −1.95, −0.13) → end < start，decode 出空字串；k=1 (42, 43, −1.34, −3.98, −5.31)「福岡」；k=2 (103, 104, −0.81, −3.15, −3.96)「填海」；k=3 (14, 35, −0.07, −1.48, −1.55) 跨進問題與 [SEP]。最大和是 k=0 的 −0.13，所以答案是空字串。原文：「福岡市的兩大中心地區是中央區的天神地區和博多站附近的博多地區」。
- 非決定性的來源（`torch.use_deterministic_algorithms(True, warn_only=True)`，bert-base QA 跑 3 步隨機輸入）：只有一個警告「Memory Efficient attention defaults to a non-deterministic algorithm」（`torch/autograd/graph.py`，觸發於 `attention_backward.cu:900`）。transformers 5 的 BERT 預設用 SDPA，GPU 上選 memory-efficient attention，反向是非決定性的。（2022 的 transformers 4.5 是手寫的 eager attention。）
