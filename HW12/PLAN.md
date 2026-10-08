# HW12 強化學習（LunarLander）— 研究筆記

> 2026-10-08 由前一個 session 整理。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。**目前不建議**：2022 年的範例程式失傳。

## 來源
- 官方 repo（virginiakm1988/ML2022-Spring）**沒有 HW12 資料夾**，README 的 Code／Slide 連結是 404（已用 GitHub API 確認目錄清單）。網路上也沒搜到 2022 的官方 notebook。
- 本資料夾的 `HW12_slides_en_2022.pdf`（18 頁）下載自課程網站：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/hw12_RL_slides_english_version.pdf
- 2021 年版的說明：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW12/HW12_EN.pdf（2021 範例程式或許還找得到，未查）。

## 題目（依 2022 投影片）
- OpenAI Gym 的 LunarLander；自己實作 Policy Gradient（REINFORCE，p.4 附演算法，「to get 3 points」）與 Actor-Critic（「to get 4 points」）。
- 提交：`.npy` 動作序列到 JudgeBoi（4 分，評分系統重播動作看總獎勵；長度不符會被拒；只有 public）、程式 2 分、報告 4 分。**基準線分數投影片文字裡沒有**（可能在範例程式）。
- 報告：1) 從 REINFORCE with baseline、Q Actor-Critic、A2C、A3C 選一個實作並說明與 PG 的差別；2) MuZero 論文的選擇題。
- 不需 GPU，應在 30 分鐘內訓練完；不准用額外資料或預訓練模型；助教只保證 Colab 的可重現性。p.6 的範例訓練曲線：總獎勵約從 −500 爬到 200–300。

## 評估
- 本機可量平均獎勵；觀念延續到 RLHF／推理模型的 RL 訓練。
- 但沒有範例程式可讀（不符本系列「讀既有程式」的定位），Gym 已由 Gymnasium 接手（版本名與 `reset`／`step` API 改過），RL 對種子很敏感。要做只能：自己寫一份，或改用 2021 版範例程式（要另外找與查證）。
