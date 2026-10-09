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

## ch01 實測（`docs/tools/hw07_ch01.py`、`hw07_ch01b.py` → `.txt`，CPU）
- 詞表 21,128：[PAD] 0、[UNK] 100、[CLS] 101、[SEP] 102、[MASK] 103；單一中日韓字 7,321 個、`##` 開頭 9,614 個（其中 `##`+單字 7,321 個，中文用不到）、[unusedN] 99 個、小寫英文字 2,084 個、純數字串 873 個（如 2022、1993、3400）。
- **bert-base-chinese 的 tokenizer_config.json 是 `{"do_lower_case": false, "model_max_length": 512}`**：不轉小寫，而詞表裡的英文字全是小寫，所以大寫英文幾乎都變 [UNK]（HTTP、GDP、NHL、Duff Roblin → [UNK] [UNK]）。改 `do_lower_case=True`：HTTP → http、GDP → gdp、Duff Roblin → du ##ff ro ##b ##lin。
- 切法例：'李宏毅教授2022機器學習' → 李 宏 毅 教 授 2022 機 器 學 習（與投影片 p.10 的 id 完全相同）；'1338年…' → 133 ##8 年 …；'128所' → 128 所；'iPhone 13' → [UNK] 13；'張騫' → 張 [UNK]；'朱允炆' → 朱 允 [UNK]；'「HD-Ready」' → 「 [UNK] - [UNK] 」；全形 '１２３ＡＢＣ' → 一個 [UNK]；'台灣臺灣' → 台 灣 臺 灣（繁簡異體都在詞表）；decode 一律在 token 之間插空白（'李 宏 毅 教 授 2022 機 器 學 習'）。
- offsets：每個 token 對應原文的 (起, 迄) 字元位置，[UNK] 也有（'Duff Roblin' → (0,4)、(5,11)）。
- 加特殊符號：'哪一地區?' + '天神地區' → [CLS] 哪 一 地 區 ? [SEP] 天 神 地 區 [SEP]，token_type_ids 0×7、1×5。
- train 不同的中日韓字 5,804 個，不在詞表 842 個，出現 3,175 次（總 4,252,213 字的 0.075%）；最常見：鄴 68、麪 58、鈽 49、滎 47、犛 46、紇 42、煬 37、牀 34、覈 32、堊 30。
- dev 文章的 [UNK] token 1,346 個（來源 644 種）：拉丁字母 779、中日韓字 430、其他 137；最多：「—」97、NHL 32、GDP 27、韃 23、「…」22、閭 17、OVA 16。
- 轉小寫：dev [UNK] 1,346 → 564、train 8,278 → 4,051；但 decode 還原不了的答案 dev 仍 44（含 [UNK] 39 → 19，其餘變成大小寫不符）、train 311 → 309。
- **改用 offsets 從原文切答案**：還原不了的只剩 dev 1 題（id 3528：'1953年' → '11953年'，token 119 ##53 年，前面緊接一個 1）、train 73 題（都是數字併進大 token：'12' → '128'、'7' → '70'、'32' → '3200'）。
- 10,524 篇訓練文章斷詞 0.55 秒（Rust 寫的 fast tokenizer）。
- 五個中文模型（bert-base-chinese、ckiplab、hfl roberta-wwm-ext、hfl macbert-base、luhua large）的 vocab.txt md5 全是 3b5b76c4aef48ecf8cb3abaafe960f09；cls／sep 都是 101／102。
- **tokenizer 預設是否轉小寫**：bert-base-chinese、ckiplab（tokenizer_config `do_lower_case: false`）不轉；hfl roberta-wwm-ext、hfl macbert-base、luhua large 沒有設定 → 預設 True，'HTTP GDP' → ['http', 'gdp']。
- 'iPhone 13 於 2021 年發表' → ['[UNK]', '13', '於', '2021', '年', '發', '表']。

## ch02 實測（`docs/tools/hw07_ch02.py` → `hw07_ch02.txt`，CPU）
- max_seq_len 193。train[1]（百濟國在哪一年建國? → 公元前18年）：三個張量都是 (193,)；start/end 85/89（公 元 前 18 年）；問題部分含 CLS/SEP 12 個 token；token_type_ids 0 有 42 個（問題 12 + padding 30）、1 有 151 個（文章 150 + 結尾 SEP）；attention_mask 1 有 163 個、padding 30。
- **訓練視窗的答案位置**：答案中點正好在文章部分第 75 個 token 的 52.37%（70–80 之間 54.85%）；視窗被文章開頭擋住（答案在前 75 token）33.96%（10,762 題）、被文章結尾擋住 13.67%（4,331 題）；短於 150 的文章 1 篇。中點位置直方圖 [0,15,30,45,60,75,76,90,105,120,135,150)：3894、2102、1700、1647、1419、16597、896、932、901、906、696。
- dev 視窗：每題平均 3.32（2 個 310 題、3 個 2,389、4 個 1,312、5 個 81、6 個 21、7 個 15、8 個 3），共 13,695；padding 佔 21.51%；最後一個視窗的文章 token 平均 78.0，≤ 20 的 12.01%，≤ 10 的 258 題。
- dev 答案落在哪個視窗（stride 150，未被切斷的 4,057 題）：第 0 個 2,194（54%）、第 1 個 1,188、第 2 個 571、第 3 個 93、第 4 個 8、第 5 個 3。答案在所在視窗內的中點位置 [0,30,60,90,120,150)：1251、882、717、702、505（偏前）。
- 被切斷的答案 vs stride（dev 視窗總數）：150 → 74（13,695）；128 → 3（15,844）；100 → 0（19,619）；75 → 0（25,481）；50 → 0（37,151）。

## ch03 實測（`docs/tools/hw07_ch03.py` → `hw07_ch03.txt`；決定性 base 種子 0 的 checkpoint，eval 模式）
- **訓練的量法（一個以答案為中心的視窗、start 與 end 都對）**：dev 4,131 題 start 0.7981、end 0.8383、兩者都對 **0.7492**、loss 0.5598；train 抽 4,131 題（random.Random(0)）start 0.9199、end 0.9099、兩者 **0.8722**、loss 0.2868。→ 訓練最後幾百步印的 acc ≈ 0.74 和 dev 的同一量法相同；訓練題目高 12 點。
- **dev EM 的去向（評估視窗 stride 150，範例 evaluate）**：EM 1,718；答案被切斷 72（另 2 題被切斷卻答對：答案文字在別處也出現）；選錯視窗 1,215；視窗對、位置錯 1,116；位置對、文字還原不了 10；回答空字串 135（全部是選中視窗 end < start）；選中的 span 起點在問題區 17。
- 來源：`modeling_bert.py`（transformers 5.18.0）1291 起 `BertForQuestionAnswering`：`BertModel(config, add_pooling_layer=False)`、`qa_outputs = nn.Linear(hidden, num_labels=2)`；loss 在 1340–1347：位置 clamp 到 [0, 序列長]、`CrossEntropyLoss(ignore_index=序列長)`、`(start_loss + end_loss) / 2`。
- **非決定性不只 SDPA**（`docs/tools/hw07_eager_det.py` → `hw07_eager_det.txt`；訓練 100 步後的權重總和）：eager attention（2022 的寫法）兩次 −12787.717／−12787.707（不同）；eager＋`CUBLAS_WORKSPACE_CONFIG=:4096:8` 兩次 −12787.710／−12787.703（仍不同）；eager＋`use_deterministic_algorithms(True)` 兩次都是 −12787.694937936982（相同）。`warn_only=True` 對 eager 不發任何警告 → 有些運算在開關打開時被默默換成決定性的版本，PyTorch 不會提醒；本書沒有逐一找出是哪個運算。SDPA 一次 −12808.497。
- 未訓練的問答頭（`torch.manual_seed(0)` 後 from_pretrained，CPU、eval 模式，訓練集前 256 題的訓練視窗）：loss 5.2214，ln(193) = 5.2627；start_logits 標準差 0.356；qa_outputs 權重標準差 0.0206、偏差 [0, 0]（initializer_range 0.02）。
- 訓練迴圈的小毛病（讀程式）：`step` 從 1 開始、每批之後才加 1，`step % 100 == 0` 時印出 → 第一次印出只累積了 99 批、卻除以 100；之後每次 100 批；最後印在第 899 批之後，第 900–991 批（92 批）從來沒印。`train_loss += output.loss` 累加的是帶計算圖的張量（印的時候才 `.item()`）。
- `tokenizer.decode` 不略過特殊符號：decode([0,0,0]) = '[PAD] [PAD] [PAD]'、decode([1921,4868,0,0]) = '天 神 [PAD] [PAD]'、decode([101,1921]) = '[CLS] 天'。
- transformers 5.18 `TrainingArguments` 預設：lr_scheduler_type "linear"、weight_decay 0.0、adam_epsilon 1e-8、full_determinism False；建立時需要 accelerate>=1.1.0（本機沒裝，會 ImportError）。

## ch04 實測：後處理與評估 stride（`docs/tools/hw07_post.py` → `hw07_post.jsonl`；決定性 base 種子 0 的 checkpoint，不重新訓練）
規則：sample＝範例 evaluate；sample_off＝同樣的選擇、用 offsets 從原文切；valid＝每個視窗在文章範圍內取 i ≤ j 的最佳一對（start logit + end logit）、decode；valid_len＝再加答案 ≤ 30 token；valid_len_off＝再用 offsets；valid_len_off_lp＝視窗之間改比 log_softmax(start)+log_softmax(end)。工具一次推論整批 256 個視窗，sample 在 stride 150 得 0.41588，和 train.py 相同。

| stride | 視窗數 | sample | sample_off | valid | valid_len | valid_len_off | valid_len_off_lp |
|---|---|---|---|---|---|---|---|
| 150 | 13,695 | 0.41588 | 0.41854 | 0.42242 | 0.42411 | 0.42677 | 0.37570 |
| 100 | 19,619 | 0.46720 | 0.46962 | 0.47543 | 0.47785 | 0.48027 | 0.40644 |
| 75 | 25,481 | 0.49455 | 0.49746 | 0.49867 | 0.49964 | 0.50278 | 0.42290 |
| 50 | 37,151 | 0.52602 | 0.52893 | 0.53086 | 0.53280 | 0.53571 | 0.43210 |
| 32 | 56,958 | 0.56306 | 0.56596 | 0.56669 | 0.56863 | 0.57153 | 0.44275 |

- 速度：stride 150 全部 6 種規則一次 1 分 04 秒（推論＋規則）。

## 位置偏差（`docs/tools/hw07_pos.py` → `hw07_pos.jsonl`；同一個 checkpoint）
直方圖分組 [0,15,30,45,60,70,81,90,105,120,135,150)（文章部分的位置；[70,81) 寬 11 是中央）。
- 每個 dev 視窗的預測中點（(argmax start + argmax end)//2 − 問題長，只計落在文章部分的）：stride 150 → 2593、1535、1392、1219、1106、**4023**、684、428、237、85、24；stride 32 → 7519、5619、5194、5067、4895、**19759**、3319、2059、1112、457、135。中央每格密度約為左鄰的 3.3 倍（150）、3.7 倍（32）；90 以後很少。
- 範例規則選中的視窗含答案的題數：stride 150 → 2,734；32 → 3,213。選中視窗裡答案中點的分布：150 → 627、355、356、286、191、235、134、203、145、125、77；32 → 594、455、290、346、314、**654**、185、161、89、94、31。
- 視窗分數（max start + max end）平均：含答案的視窗 8.086（150）／6.260（32），不含答案的 0.757／0.278。
- 逐步加規則（stride 150，與前一列相比 修好／弄壞）：sample → sample_off 11／0；sample → valid 27／0；valid → valid_len 7／0；valid_len → valid_len_off 11／0；sample → valid_len_off 45／0。修好的 45 題：原本空字串 27、含 [UNK] 9、超過 40 字 6、其他 3。範例回答 stride 150：空字串 135（valid_len_off 0）、含 [UNK] 42、含 [SEP]/[CLS]/[PAD] 0、超過 30 字 79。stride 150 → 32（valid_len_off）：修好 842、弄壞 244。
- stride 8：視窗 221,409；sample 0.62672、sample_off 0.63036、valid 0.62794、valid_len 0.62842、valid_len_off 0.63205、valid_len_off_lp 0.45050。
- stride 16（同一個 checkpoint）：視窗 111,762；sample 0.60881、sample_off 0.61220、valid 0.61051、valid_len 0.61196、valid_len_off 0.61535、valid_len_off_lp 0.44977。
- 可讀版 `docs/tools/hw07_postprocess_v2.py`（evaluate_v2，逐題、DataLoader batch 1）stride 150：0.42677（1763/4131），與 hw07_post.py 的 valid_len_off 相同。
- 注意：M、O 組（nd3、lin*、optt*、low1*）訓練時 GPU 上同時跑過推論工具（hw07_post、hw07_pos、hw07_ch03 等），jsonl 裡的 train_s／dev_s 不是乾淨的計時；只用在決定性結果，不用來比速度。速度另外在乾淨的 GPU 上量。

## O 組：兩種 AdamW（1 epoch、其他照範例；`hw07_runs.jsonl` 的 optt1_s*）
- torch.optim.AdamW（預設 eps 1e-8、weight_decay 0.01）：種子 0／1／2 = 0.45243、0.47785、0.44372，平均 0.458。舊版（nd3_s* 的第 1 個 epoch）：0.41588、0.48221、0.43888，平均 0.446。

## M 組（`hw07_runs.jsonl`）
- 不衰減 3 epoch（nd3）dev_by_epoch：種子 0 [0.41588, 0.43282, 0.47301]；種子 1 [0.48221, 0.47906, 0.51053]；種子 2 [0.43888, 0.47107, 0.51440]。第 1 個 epoch 平均 0.446（範圍 0.416–0.482，差 6.6 點）。
- 線性衰減 1 epoch（lin1）：0.52820、0.56621、0.56161，平均 0.552。
- 後處理在其他 checkpoint（`hw07_post.jsonl`；valid_len_off）：lin1_s0 stride 150／100／32／16 = 0.54636／0.60809／0.67514／0.69838（sample 0.5282／0.59283／0.66449／0.68821）；lin1_s1 = 0.58533／0.65263／0.69886／0.70709（sample 0.56621／0.63568／0.68773／0.69547）。

## ch07 背景（模型卡，2026-10-09 讀取）
- hfl/chinese-roberta-wwm-ext：README 指向論文 Pre-Training with Whole Word Masking for Chinese BERT（arXiv 1906.08101）與 Revisiting Pre-Trained Models for Chinese NLP（arXiv 2004.13922，EMNLP 2020 Findings）；「Please use 'Bert' related functions to load this model」。config architectures = BertForMaskedLM（沒有問答頭）。
- hfl/chinese-macbert-base：MLM as correction（用 Synonyms 工具找相似詞取代 [MASK]，沒有相似詞時用隨機詞）＋ whole word masking、N-gram masking、Sentence-Order Prediction；「can be directly replaced with the original BERT as there is no differences in the main neural architecture」。BertForMaskedLM。
- ckiplab/bert-base-chinese-qa：config architectures = BertForQuestionAnswering（有問答頭）；README 要求 tokenizer 用 bert-base-chinese 的 BertTokenizerFast；ckip-transformers GitHub：語言模型以 ZhWiki（20200801，用 OpenCC 轉繁體）與 Chinese Gigaword 5th 的 CNA（中央社）訓練；README 沒有提 QA 模型用什麼資料訓練、也沒有 QA 分數。
- luhua/chinese_pretrain_mrc_roberta_wwm_ext_large：模型卡「使用大量中文MRC数据训练的roberta_wwm_ext_large模型」，GitHub basketballandlearn/MRC_Competition_Dureader：「网上收集的大量中文MRC数据（其中包括公开的MRC数据集以及自己爬取的网页数据等，囊括了医疗、教育、娱乐、百科、军事、法律、等领域。）」。沒有點名 DRCD；表格是 DuReader-2021 與 tencentmedical 的評估。config = BertForQuestionAnswering、24 層、hidden 1024、16 頭、intermediate 4096。→ 是否看過 DRCD 無法從文件確認，用 zero-shot dev EM 間接判斷。
- 位置偏差（`hw07_pos.jsonl`）中央格（70–80）每位置密度 ÷ 左鄰格（60–69）：base1_s0 3.31（150）／3.67（32）；lin1_s0 2.05／2.14；lin1_s1 2.00／2.03；lin1_s2 1.97／1.91。選中視窗含答案：base1_s0 2,734／3,213；lin1_s0 3,111／3,507；lin1_s1 3,235／3,574；lin1_s2 3,253／3,622。視窗分數平均（含答案／不含）：lin1_s0 11.64／2.26（150）。→ 線性衰減的模型中央偏好較弱（約 2 倍）但仍在。
- M 組：lin2_s0 dev_by_epoch [0.51513, 0.54345]。nd3_s0 後處理 valid_len_off：stride 100／32／16 = 0.53038／0.59477／0.62600（sample 0.51949／0.58557／0.61801）。
- M 組完成：lin2 dev_by_epoch 種子 0 [0.51513, 0.54345]、1 [0.53716, 0.54490]、2 [0.49915, 0.56233]（最終平均 0.550）；lin3 種子 0 [0.50666, 0.54733, 0.56403]、1 [0.50884, 0.54539, 0.53958]、2 [0.49116, 0.54684, 0.53522]（最終平均 0.546）。lin1 平均 0.552。不衰減：1 epoch 0.446、2 epoch 0.461（0.43282、0.47906、0.47107）、3 epoch 0.499（0.47301、0.51053、0.51440）。
- 乾淨時（GPU 只有訓練）的速度：lin3_s0 3 個 epoch 含每個 epoch 的 dev 評估 16 分 42 秒（00:03:14–00:19:56）。
- 固定 lr 5e-5（lr5，1 epoch、不衰減）：0.53062、0.52046、0.51997，平均 0.524。
- 衰減 lin2／lin3 checkpoint 的後處理 valid_len_off stride 150／16：lin2_s0 0.55967／0.71242、lin2_s1 0.56040／0.70031、lin2_s2 0.58243／0.71290；lin3_s0 0.58170／0.71339、lin3_s1 0.55265／0.70467、lin3_s2 0.54975／0.70007；lin1_s2 stride 16 = 0.70564。stride 16 平均：lin1 0.704、lin2 0.709、lin3 0.706。
- 轉小寫（low1，1 epoch、不衰減，範例評估）：0.47470、0.47035、0.41927，平均 0.455（基準 0.446）。

## ch06 實測：隨機視窗
- `hw07_exp.py --window random`（46–71 行 Exp_Dataset）：視窗起點在「答案完整落在視窗內、視窗不超出文章」的範圍裡均勻抽（`random.randint`，主行程、受 same_seeds 控制；每個 epoch 重抽）。
- 訓練答案中點位置（random.Random(0) 模擬一輪）：正好 75 的 0.48%（中心視窗 52.37%）；直方圖 [0,15,…,150)：7967、3797、2888、2528、2283、151、1893、2134、2307、2605、3137；前半（<75）61.42%、後半（>75）38.11%（中心視窗後半 13.67%）。0–14 偏多是因為答案在文章最前面時視窗只能從 0 開始。
- S 組（win1，隨機視窗＋線性衰減 1 epoch）：0.70201、0.70758、0.69741，平均 0.702。
- 後處理×stride（3 種子平均；sample／valid_len_off）：lin1 150 0.552／0.570、100 0.623／0.639、32 0.682／0.693、16 0.694／0.704；win1 150 0.702／0.719、100 0.713／0.728、32 0.718／0.729、16 0.717／0.727。lin1_s2 valid_len_off：150 0.57879、100 0.65626、32 0.70419、16 0.70564。win1 各種子 valid_len_off：s0 0.71992／0.72936／0.72694／0.72452；s1 0.72283／0.73324／0.73203／0.73275；s2 0.71557／0.72137／0.72936／0.72307（150／100／32／16）。
- 位置偏差 win1（中央÷左鄰密度，150／32）：s0 1.05／0.98、s1 0.93／0.94、s2 0.99／1.00。win1_s0 stride 150 預測中點直方圖：3919、1757、1308、1096、626、726、484、763、757、917、1126。選中視窗含答案（150）：s0 3,710、s1 3,714、s2 3,667。視窗分數平均（含／不含）：s0 15.52／3.82、s1 15.55／2.87、s2 14.42／3.65。
- 零樣本（範例評估 stride 150）：ckiplab/bert-base-chinese-qa 0.41467；luhua large 0.36868（tokenizer 轉小寫、decode）。
- D 組：rwe1_s0 0.58799、rwe1_s1 0.59477。
- 依答案位置分組的 EM（stride 150、範例規則，lin1_s0 中心 vs win1_s0 隨機；分組以 stride 150 的視窗、答案中點在視窗內的位置）：第 1 個視窗 0–49（1,156 題）0.783／0.775；50–99（579）0.511／0.665；100–149（459）0.240／0.691；之後的視窗 0–49（735）0.536／0.710；50–99（632）0.557／0.698；100–149（496）0.250／0.681；被切斷（74）0.014／0.014。中心 → 隨機：修好 901、弄壞 183。
