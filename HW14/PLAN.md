# HW14 終身學習（LifeLong Learning）— 研究筆記與計畫

> 2026-10-08 由前一個 session 整理，尚未開始做。總覽與排序見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。**建議下一本書做這份。**

## 題目（讀自 `HW14.ipynb`、`HW14.pdf`，兩者都從官方 repo 複製進本資料夾）
- 同一個模型**依序**學多個任務，學新任務時不要忘掉舊任務（catastrophic forgetting，災難性遺忘）。
- 資料：**旋轉 MNIST**，助教的程式產生；投影片：5 個任務、每個任務多轉 20°。**notebook 某處寫「utilize 5 different rotations to generate 10 different rotated MNISTs」，和投影片不一致，Phase 0 要以程式實際行為為準。**
- 模型固定：4 層全連接網路（為了公平比較，不准改架構）。
- 每個任務訓練 10 個 epoch；每種方法在 Tesla T4 約 20 分鐘、K80 約 60 分鐘。
- 範例程式**已經寫好六種方法**（同一套 train／evaluate 函式）：
  - Baseline（正則化項為 0）
  - EWC（Elastic Weight Consolidation，Fisher 矩陣估權重重要性）— arXiv:1612.00796
  - MAS（Memory Aware Synapses，輸出對權重的敏感度，不需標籤）— arXiv:1711.09601；**唯一要自己寫的 TODO：MAS 的 Omega 矩陣（只做 global 版：取最後一層的輸出）**
  - SI（Synaptic Intelligence，累積每次更新對 loss 下降的貢獻）— arXiv:1703.04200
  - RWalk（Riemannian Walk，結合 EWC 與 SI 的想法）— arXiv:1801.10112
  - SCP（Sliced Cramer Preservation）
- 評分**沒有排行榜**：
  - 20 題選擇題（8 分）：基本概念 3、EWC 2、MAS 2、SI 2、RWalk 2、SCP 3、其他方法與情境 6（iCaRL、LwF、GEM、DGR、三種持續學習情境）。
  - 報告（2 分）：畫每種方法的學習曲線（繪圖函式已提供）、說明使用的指標、貼出 MAS Omega 的 TODO 實作。

## 本機狀態
- 本資料夾只有官方 `HW14.ipynb`、`HW14.pdf`，**還沒拆成 .py**。
- 資料：notebook 用 torchvision **下載 MNIST**（本機的 Kaggle zip 裡沒有 hw14）。依約定，下載前要問使用者。
- 環境：純 PyTorch，應可直接在共用 `.venv` 跑；**未實測**。

## 為什麼值得做
- **「遺忘」是 ML2022、2025、2026 唯一連續三年都有的作業主題**（2025 HW6、2026 HW5 是 LLM 版，見 [ML2025-Spring/HW06/PLAN.md](../ML2025-Spring/HW06/PLAN.md)）。HW14 是這個主題的理論基礎：為什麼會忘、怎麼量、理論上怎麼保護。
- 本機可以完整實測：每個任務都有測試集，「學完第 k 個任務後前面每個任務的準確率」全部算得出來；MNIST 加小模型，多種方法、多個種子都跑得起。
- 六種方法並列在同一套程式裡，正好逐一讀懂「權重重要性」各自怎麼算。

## 弱點
- 任務很玩具（旋轉 MNIST），離實務較遠。
- 作業是選擇題，沒有「往上爬分數」的主線；書的結構偏「方法比較」，不是改良競賽。

## Phase 0 要做的事（照 docs/TEXTBOOK_WORKFLOW.md）
1. **先對照現行 skill**（`completed-repo-to-html-textbook`）：不寫附錄、最後一章是總結章（題庫＋名詞對照＋速查）、樣板自己設計寫在 index.html（HW06 的 `docs/tools/hw06_book/` 可參考做法，但配色要為本書重新設計）。
2. 問使用者能否下載 MNIST；把 notebook 拆成 .py（參考 HW06 的拆法：config、dataset、model、各方法、train、plot），和原版逐格比對。
3. 跑 Baseline 與六種方法、計時；確認逐位元可重現。
4. 確認任務數（5 還是 10）、旋轉角度、每個任務的資料量、評估指標的定義。
5. 寫大綱（outline.html）與事實清單（docs/HW14/FACTS.md），等使用者核可。

## 已想到的章節方向（未核可）
- ch00 任務與模型總覽（4 層 FC、參數量；災難性遺忘的示範：Baseline 的學習曲線）。
- 資料：旋轉 MNIST 怎麼產生。
- 訓練／評估框架：共用的 train、evaluate，正則化項怎麼插進 loss。
- EWC、MAS（含 TODO 實作）、SI、RWalk、SCP 各一節或一章：重要性怎麼算、要不要標籤、成本。
- 方法比較：同一規格下的學習曲線、平均準確率、遺忘量；多種子。
- 總結章：三種持續學習情境、其他方法（iCaRL、LwF、GEM、DGR）、選擇題題庫、名詞對照、速查。
- 「現在的做法」框：LoRA、Self-Instruct（重播的精神）如何延續這些想法，引用 ML2025 HW6／ML2026 HW5。
