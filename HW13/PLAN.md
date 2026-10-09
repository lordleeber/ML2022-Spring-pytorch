# HW13 模型壓縮（Network Compression）— 研究筆記與計畫

> 2026-10-08 由前一個 session 整理，尚未開始做。總覽與排序見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。**建議排第二。**

## 題目（讀自本資料夾的 `Machine Learning HW13.pdf`，23 頁；從課程網站下載：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Machine%20Learning%20HW13.pdf）
- 把 HW03 的 Food-11 分類模型縮小：**學生模型參數 ≤ 100,000**（用 `torchsummary` 算，含不可訓練參數；ensemble 要把所有模型加總）。違反就整份 0 分。不准用預訓練模型與外部資料；測試資料只能拿來推論（不能用老師模型對測試集做 pseudo-label）。
- 資料：Food-11，11 類；訓練 9,866、**驗證 3,430（有標籤）**、測試 3,347。
- 提供老師模型：ResNet-18，`resnet18_teacher.ckpt`（44,806,605 bytes），測試準確率約 0.899。
- 主題：knowledge distillation、architecture design（depthwise + pointwise 卷積）、network pruning（報告 Q3）；另提 parameter quantization、dynamic computation。
- 基準線（Kaggle 準確率）：
  | 基準線 | 準確率 | 做法 | 時間（投影片） |
  |---|---|---|---|
  | Simple | > 0.44820 | 照跑 | < 1 小時 |
  | Medium | > 0.64840 | 完成 KD 的 KL 散度 loss、調 α 與 T、訓練更久 | < 3 小時 |
  | Strong | > 0.82370 | depthwise／pointwise 卷積（MobileNet、ShuffleNet、DenseNet、SqueezeNet、GhostNet…） | 8–12 小時 |
  | Boss | > 0.85159 | 進階 KD（FitNet、Relational KD、DML）、更強的老師、TAKD | 未量 |
- 報告：
  1. 貼學生模型架構與 `torchsummary` 結果（≤ 100,000）。
  2. 貼 KD loss（`loss_fn_kd`，α = 0.5、T = 1.0）；溫度 T 的選擇題（T 越高分布越軟）。
  3. 對老師模型做剪枝，畫剪枝比例對驗證準確率的曲線；回答 PyTorch 教學的剪枝能否加速推論（教學的剪枝只是把權重設 0，計算量不變 → 很適合實測）。

## 本機狀態
- 本資料夾只有投影片。**官方 GitHub 的 HW13 只有資料的 LFS 指標檔，沒有範例程式**；投影片說範例程式在 Kaggle 比賽頁（「Kaggle Competition (with sample code)」），Colab 版「not recommended」。**取得範例程式是 Phase 0 的第一件事**（Kaggle 可能要登入／API token，先問使用者）。
- 資料：`/mnt/c/Users/valtec/Documents/poyi/ml_2022_data/ml2022spring-hw13.zip`（1,205,018,903 bytes），內含 `food11-hw13/{training,validation,evaluation}/` 與 `resnet18_teacher.ckpt`，共 16,644 個檔。
- 需要 `torchsummary`（.venv 裡未確認）。

## 為什麼值得做
- 驗證集有標籤，本機能量準確率。
- 是 HW03（同一份 Food-11）的續集，可沿用 docs/HW03 的事實。
- 蒸餾、剪枝、量化、輕量架構是今天部署大模型的核心；對應 ML2026 HW3「LLM Fast Inference」。
- 「剪枝不一定變快」這類觀念能實測。

## 風險
- 範例程式要從 Kaggle 取得。
- Strong 在 Kaggle 上 8–12 小時（本機應快很多，但實驗量大）。

## Phase 0
1. 對照現行 skill（見 HW14/PLAN.md 第 1 點）。
2. 取得範例程式、建立 .py、和原版比對；解壓資料（只在本機）。
3. 跑 Simple、計時；確認驗證指標沒有偏差（HW01、HW04 的教訓）。
4. 老師模型在驗證集上的準確率、參數量；學生模型的參數量用 torchsummary 驗證。
5. 寫大綱，等使用者核可。

## 狀態（2026-10-09 09:50）
- 範例程式：投影片的 Colab 連結匿名下載成功（使用者同意），`HW13/HW13.ipynb`、`HW13/HW13_pruning_example.ipynb`。torchsummary 1.5.1 裝進共用 .venv（使用者決定）。
- 現行 skill 與 HW14 時相同（mySkills 27d96f9）：不寫附錄、最後一章是總結章、樣板自己設計寫在 index.html、不冷讀（照前幾本）。
- notebook 拆成 `config.py`、`dataset.py`、`model.py`、`kd.py`、`train.py`、`test.py`；參照版（`docs/tools/hw13_make_ref.py`）8 分 19 秒，train.py 與它逐位元一致。
- 事實在 `docs/HW13/FACTS.md`：學生 87,907 參數、老師 11,182,155；老師驗證 0.86093；Simple 驗證 0.50408；指標無偏差；資料與 HW03 相同（只有測試集重新編號）。
- 下一步：大綱與實驗規格給使用者核可。

## 使用者決定（2026-10-09 09:55）
- 大綱核可：index、ch00 全貌／ch01 程式與訓練迴圈／ch02 參數怎麼數／ch03 老師／ch04 知識蒸餾／ch05 架構設計／ch06 剪枝／ch07 量化與其他／ch08 總結；不寫附錄、不冷讀、樣板自己設計。
- 實驗 A+B+C+D：A KD（10 epoch CE vs KD；50 epoch T∈{1,2,4,8}×α∈{0.5,0.9} + CE）、B 架構（兩種 ≤100k 學生 × CE／最佳 KD，200 epoch + HW03 增強，加同參數對照）、C 剪枝、D 量化。實驗工具先在 nw=0 驗證與 train.py 逐位元一致，正式實驗用 16 workers；一組一組依序跑。
- 節奏照 HW14：每章寫完驗證就推、接著寫下一章；需要 GPU 實驗就直接做。長任務每 30 分鐘回報。
