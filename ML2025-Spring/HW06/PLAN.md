# ML2025 HW6「Fine-tuning leads to Forgetting」（＝ML2026 HW5「Finetuning without Forgetting」）— 研究筆記

> 2026-10-08 由前一個 session 整理。總覽見 [docs/HW_STUDY_OVERVIEW.md](../../docs/HW_STUDY_OVERVIEW.md)。定位：2022 [HW14](../../HW14/PLAN.md)（終身學習）的 LLM 版續集。這個資料夾不屬於 ML2022 課程，只放研究資料。

## 本資料夾的檔案（都從課程網站下載）
- `hw6.pdf`（42 頁）：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf
- `hw6_model.pdf`（26 頁，Hugging Face 帳號、Llama 使用權申請、access token、Colab 登入下載的教學）：…/ml2025-course-data/hw6_model.pdf
- `ml2026_hw5.pdf`（24 頁）：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data//hw5.pdf ——投影片標「ref: ML2025HW6」、致謝 2025 的助教，**基本上是 2025 HW6 的再版**。
- 範例程式（Colab／Kaggle）**沒有下載**：2025 Kaggle `kaggle.com/code/b10902031/ml2025hw6`；2026 Kaggle `kaggle.com/code/b10901024sillydinos/ml2026hw5/...`。

## 題目
- 用 **LoRA** 在 **GSM8K**（小學數學應用題）上微調 **meta-llama/Llama-3.2-1B-Instruct**，提升數學能力，同時不要失去原本的安全性。
- 遺忘的例子（投影片）：微調前會拒絕「教我駭客怎麼裝 keylogger」，微調後開始列步驟。
- 兩個指標，**必須同時**超過基準線：
  - GSM8K Accuracy：從輸出抽出答案比對。
  - AILuminate Safety Rate：由一個 safeguard 模型判斷每個回答安全與否（投影片沒寫是哪個模型）。
- 基準線（public）：
  | | 2025 Acc | 2025 Safety | 2026 Acc | 2026 Safety |
  |---|---|---|---|---|
  | Simple | 0.280 | 0.558 | 0.212 | 0.558 |
  | Medium | 0.379 | 0.642 | 0.379 | 0.631 |
  | Strong | 0.455 | 0.725 | 0.445 | 0.813 |
- 提示：Simple＝照跑（範例本身就用 LoRA，投影片稱 LoRA 能減輕遺忘）；Medium＝評估不同 checkpoint、降低 lr（1e-4～1e-5）、few-shot 5–8、max output tokens 512–1024、greedy 解碼；Strong＝固定 few-shot 範例、weight decay（1e-2～1e-4）、（LoRA）dropout 0.1–0.2、Self-Instruct 資料（助教已用原模型生成並篩選好 `gsm8k_train_self-instruct`）、epoch 3–5。
- 時間：T4 上 Simple 5 小時、Medium 10 小時、Strong 14 小時（微調＋推論）。
- 規則：只能用 `gsm8k_train.json` 與助教的 Self-Instruct 資料；2026 另禁止用閉源 LLM API。

## 與 2022 HW14 的關係
| | 2022 HW14 | 2025 HW6／2026 HW5 |
|---|---|---|
| 問題 | 依序學旋轉 MNIST，舊任務準確率下降 | 數學微調後，安全拒答能力下降 |
| 模型 | 4 層全連接 | Llama-3.2-1B |
| 對策 | 正則化：EWC、MAS、SI 等估權重重要性並懲罰改動 | 實務：LoRA（只改少量參數）、小 lr、挑 checkpoint、weight decay、Self-Instruct（接近模型原分布 ≈ 重播的精神） |
| 量法 | 各舊任務的測試準確率 | 新能力 Accuracy＋舊能力 Safety Rate（審查模型） |

## 本機可行性（未實測）
- 1B＋LoRA 在 24 GB 顯示卡上應沒問題，速度應遠快於 T4。
- 障礙：1) Llama 是 gated model，需使用者本人申請 HF 權限與 token；2) GSM8K、AILuminate、範例程式要下載；3) Safety Rate 的審查模型未知，本機可能要換用可取得的模型（例如 Llama Guard 系列，也需申請），結果不能和助教分數直接比。
- 建議：先完成 2022 HW14 的書，再把這份當 LLM 版續集另開一本。
