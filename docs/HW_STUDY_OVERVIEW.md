# 下一本書要做哪一份作業：候選評估（2026-10-08）

HW01–HW04、HW06、HW07、HW09、HW10、HW13、HW14 的讀碼教材已完成（HW14、HW13 於 2026-10-09、HW07 與 HW10 於 2026-10-10 完成）。這份文件整理 2026-10-08 一個 session 裡看過的其餘作業，以及 ML2025／ML2026 的對照；每份作業的細節在各自資料夾的 `PLAN.md`。**新 session 從這裡開始**，然後讀選定作業的 `PLAN.md` 與 `docs/TEXTBOOK_WORKFLOW.md`（特別是「HW06 的調整」：大綱前先對照現行 skill、不寫附錄、每本書自己設計樣板）。

## 評估標準
1. **本機能不能量主指標**：這系列的原則是「每個數字都在本機實測」。測試集沒有標籤、又沒有驗證集的作業，結論會打折。
2. **觀念延續到 2026 的程度**：對照 ML2025／ML2026（課程已整個轉向 LLM，見下表）。
3. **環境風險**：能不能在共用 `.venv`（Python 3.12、torch 2.11+cu128、RTX PRO 4000 Blackwell）跑。
4. **範例程式在不在**：這系列是「讀既有程式」的教材。

## 候選總表

| 作業 | 主題 | 本機能量 | 延續到 2026 | 環境風險 | 範例程式 | 建議 |
|---|---|---|---|---|---|---|
| [HW14](../HW14/PLAN.md) | 終身學習（災難性遺忘） | ✓ | 高（ML2025 HW6、ML2026 HW5 同主題） | 很低 | 已拆成 .py | **已完成**（docs/HW14） |
| [HW13](../HW13/PLAN.md) | 模型壓縮（蒸餾、剪枝） | ✓（驗證集 3,430 張） | 很高（對應 ML2026 HW3 推論加速） | 低 | 已從投影片的 Colab 連結取得並拆成 .py | **已完成**（docs/HW13） |
| [HW07](../HW07/PLAN.md) | BERT 抽取式問答 | ✓（dev 4,131 題） | 中（ML2026 已無對應） | 低（已能跑） | 主線還原成官方範例 | **已完成**（docs/HW07） |
| [HW05](../HW05/PLAN.md) | 機器翻譯（seq2seq／Transformer） | ✓（驗證集有中文） | 很高（對應 ML2025 HW3/4、ML2026 HW4） | **高**（fairseq 裝不起來） | 已在 repo | 先試環境 |
| [HW10](../HW10/PLAN.md) | 對抗攻擊（轉移性） | ✓（用其他模型當本機黑箱） | 中高（對應 ML2026 HW1 越獄防禦） | 中（pytorchcv 已裝；imgaug 需 shim） | 已拆成 .py | **已完成**（docs/HW10） |
| [HW11](../HW11/PLAN.md) | 領域適應（DaNN） | ✗（目標域無標籤） | 中 | 低～中 | 官方 notebook 已複製 | 不建議 |
| [HW08](../HW08/PLAN.md) | 異常偵測（autoencoder） | ✗（測試集無標籤） | 中 | 低（已能跑） | 已在 repo | 不建議 |
| [HW12](../HW12/PLAN.md) | 強化學習（LunarLander） | ✓ | 高（RLHF） | 中（Gym→Gymnasium） | **2022 範例程式失傳** | 不建議 |
| HW15 | Meta Learning | 未看 | — | — | 官方 notebook 在 `~/poyi/GitHubPublic/ML2022-Spring/HW15/` | 未評估 |
| [ML2025 HW6](https://github.com/lordleeber/ML2025-Spring-pytorch/blob/main/HW06/PLAN.md)（＝ML2026 HW5） | 微調而不遺忘（Llama-3.2-1B + LoRA） | 部分（安全率要審查模型） | 現行課程 | 中（Llama 需申請權限） | Colab／Kaggle（未下載） | HW14 的續集 |

建議順序：HW14、HW13、HW07、HW10 已完成；下一本可接 ML2025 HW6 當 HW14 的 LLM 版續集（ML2025 HW5「Fine tune is powerful」則是 HW07 的續集）（HW13 的壓縮觀念則可接 ML2026 HW3 LLM Fast Inference）；有餘力再試 HW05 的環境。

## 三個年份的主題對照（課程網站，2026-10-08 讀取）

| 主題 | ML2022 | ML2025 | ML2026 |
|---|---|---|---|
| Transformer（訓練、理解、位置編碼） | HW04、HW05 | HW3 Understand Transformer、HW4 Training Transformer | HW4 Training Transformer |
| **微調後的遺忘** | **HW14 LifeLong Learning** | **HW6 Fine-tuning leads to Forgetting** | **HW5 Finetuning without Forgetting** |
| 模型編輯 | — | HW8 Model Editing | HW6 Model Editing |
| 模型合併 | — | HW9 Model Merging | HW7 Model Merging |
| 生成模型 | HW06 GAN | HW10 Diffusion | HW9 Flow Matching |
| AI Agent／RAG | — | HW1 AI Agent1 -- RAG、HW2 AI Agent2 | HW2 AI Agent as an AI Engineer |
| 攻擊與防禦 | HW10 Adversarial Attack | — | HW1 LLM Malicious Instruction Defense |
| 推論加速／壓縮 | HW13 Network Compression | （多 GPU 訓練助教課） | HW3 LLM Fast Inference |
| 對齊 | — | HW7 RLHF | — |
| 預訓練模型微調 | HW07 BERT QA | HW5 Fine tune is powerful | — |
| 其他 | HW08、HW11、HW12、HW15 | — | HW8 Test-Time Scaling、HW10 Spoken Language Model |

來源：https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php 、https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php （頁面註明內容暫定）。ML2022 官方 repo 在 `~/poyi/GitHubPublic/ML2022-Spring/`（上游 virginiakm1988/ML2022-Spring）：**沒有 HW12 資料夾**（README 連結是 404）、HW13 只有資料的 LFS 指標檔。

## 本機資料
- Kaggle 資料 zip 在 `/mnt/c/Users/valtec/Documents/poyi/ml_2022_data/`：hw1、2、3b、4、5、6、7、8、9、10、11、13、15。**沒有 hw12、hw14**（HW12 是 RL 環境，HW14 用 torchvision 下載 MNIST）。
- 依約定資料一律用本機 zip、不另外下載；要下載（模型權重、MNIST、HF 資料集）先問使用者。
