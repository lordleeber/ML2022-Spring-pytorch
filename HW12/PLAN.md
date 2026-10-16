# HW12 強化學習（LunarLander）— 計畫

> **2026-10-17 全書完成**：docs/HW12（index、outline、ch00–ch08，本機 commit，未推）。實驗 A–E 共 303 次訓練＋試跑 38＋Phase 0 44；事實在 docs/HW12/FACTS.md；教訓在 docs/TEXTBOOK_WORKFLOW.md「HW12 的調整」。不冷讀。

> 2026-10-08 由前一個 session 整理成研究筆記，結論是「不建議」（2022 範例程式失傳）。**2026-10-16 使用者要求重讀並重新規劃**，本檔改寫成計畫（Phase 0 前）。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)，流程見 [docs/TEXTBOOK_WORKFLOW.md](../docs/TEXTBOOK_WORKFLOW.md)。

## 來源（2026-10-16 重查）
- 本資料夾的 `HW12_slides_en_2022.pdf`（18 頁）下載自課程網站：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/hw12_RL_slides_english_version.pdf
- **2022 範例程式**：官方 repo（virginiakm1988/ML2022-Spring）沒有 HW12 資料夾，投影片 p.2 只寫「Other details can be found in the sample code」、沒有連結。2026-10-16 再搜一次網路也沒找到，視為失傳。
- **2021 範例程式還在**：ML2021 官方 repo `ga642381/ML2021-Spring` 的 `HW12/` 有 `HW12_EN.ipynb`（242 KB）、`HW12_ZH.ipynb`（255 KB）和兩份說明 PDF（2026-10-16 用 GitHub API 列過目錄，**還沒下載**）。2022 投影片的題目、兩個演算法框和「Sample Result」曲線形式都跟 2021 相同，所以 2022 的範例程式很可能就是 2021 版，或只改了一點。這點下載後要對照兩年的投影片確認，書裡只能寫「推測」，不能寫成事實。

## 2021 範例（2026-10-16 已下載，使用者同意）
- 本資料夾：`HW12_EN.ipynb`、`HW12_ZH.ipynb`、`HW12_slides_en_2021.pdf`（原檔名 `HW12_EN.pdf`），來自 `ga642381/ML2021-Spring/HW12`。
- **2022 沿用 2021 的證據**：兩年的「Sample Result」頁是同一張圖（兩條曲線逐像素看起來相同），演算法頁和環境也相同。2022 只改了配分（PG 3 分、AC 4 分；2021 是 8／10 分），新增了報告題。
- **2021 的配分門檻**（2021 p.7，圖片）：JudgeBoi 平均獎勵 <0 → 2 分、0–99 → 3、100–199 → 4、200–240 → 5、≥241 → 6。2022 沒有公布門檻，本書用這組當參考線，並標明出自 2021。
- EN 和 ZH 版的程式碼差異：ZH 多了計時（`time`）、`!pip freeze`、`agent.save()/load()`，測試時關掉繪圖、多印動作長度；兩版的訓練與測試邏輯相同。主線用 EN 版。
- 範例內容（`HW12_EN.ipynb` 程式格）：
  - 環境：`gym[box2d]==0.18.3`、`numpy==1.19.5`、`torch==1.8.1`，`seed = 543`（「Do not change this」）。`fix()` 呼叫 `env.seed()`、`torch.set_deterministic(True)`。
  - 網路：8→16→16→4，tanh，softmax。Agent 用 SGD lr 0.001，`loss = (-log_probs * rewards).sum()`。
  - 訓練：`NUM_BATCH=400` × `EPISODE_PER_BATCH=5`，共 2,000 局。**reward 是每一步的瞬時獎勵 r_t**，不是 p.4 的 R_t。註解寫明 medium 要改成折扣累積（γ=0.99），boss 要實作 DQN。整批 reward 做標準化。
  - 測試：`fix(env, seed)` 後跑 5 局，**用 `agent.sample()` 隨機取樣而不是 argmax**，存成 `Action_List.npy`。「Server」格模擬 JudgeBoi：重播動作、檢查有 5 局，遇到 done 就 break，沒檢查長度。
- **在現在的環境會壞的地方**（2026-10-16 實測）：
  - `torch.set_deterministic` 在 torch 2.11 不存在 → 改用 `torch.use_deterministic_algorithms`。
  - `np.array(action_list)` 遇到長短不一的局，在 NumPy 2 會丟 `ValueError`（inhomogeneous shape）→ 改用 `dtype=object`。
  - Gym 0.18 的 `env.seed()`、四值 `step()`、`render(mode=...)`，以及 `LunarLander-v2` → 用一層 shim 處理。Gymnasium 1.4 只註冊了 `LunarLander-v3`。

## 可行性探測（2026-10-16，scratchpad 的 `sample_probe.py`，用 shim 把範例原樣跑在 Gymnasium 1.4 上，CPU、單執行緒、5 個行程並行）
| 訓練種子 | 訓練時間 | 環境步數 | 最後 50 批的訓練平均 | 測試 5 局 | 測試平均 |
|---|---|---|---|---|---|
| 543 | 128.6 s | 471,776 | −81.8 | 254.1, −9.2, 213.7, −145.0, 229.4 | 108.6 |
| 543（再跑一次） | 128.9 s | 471,776 | −81.8 | 同上，完全相同 | 108.6 |
| 1 | 151.0 s | 537,733 | −39.1 | −224.8, −84.7, 258.1, 255.1, −210.9 | −1.4 |
| 2 | 145.2 s | 525,441 | −61.8 | −87.2, 240.0, −222.0, −9.5, 94.4 | 3.2 |
| 3 | 145.9 s | 513,284 | −90.4 | 78.9, −41.2, −57.3, −58.6, −32.6 | −22.1 |
- **可以重現**：同種子跑兩次，結果逐位元相同（這只是初步檢查，正式版要比 state_dict）。
- **很便宜**：一次完整訓練約 2.5 分鐘，比投影片說的 30 分鐘快很多，每格跑 10 個以上的種子也負擔得起。
- **單局差異大**：同一個策略的 5 局測試，分數從 −145 到 254 都有。範例的評分用 5 局平均，而且測試時是隨機取樣，所以運氣占了很大一部分。種子 543 的測試平均是 108.6，但它最後 50 批的訓練平均只有 −81.8，兩者差很多。ch07 的主題就是這個。
- 以上用一次性的 probe 量得，**不寫進書**；正式的數字要等參照版做好後重量。

## 題目（2022 投影片，p.3–p.6 是圖，已轉成圖片看過）
- 環境：OpenAI Gym 的 `LunarLander-v2`。狀態是 8 維，動作有 4 個離散選項。p.3 是兩張著陸畫面，地形每局隨機，兩支旗子之間是降落區。
- p.4「Policy Gradient（to get 3 points）」：REINFORCE。每局算折扣報酬 $R_t=\sum_{i=t}^{T}\gamma^{i-t}r_i$，更新 $\theta \leftarrow \theta+\alpha\nabla_\theta\log\pi_\theta(a_t|s_t)R_t$。
- p.5「Actor-Critic（to get 4 points）」：框裡的演算法其實是 **REINFORCE with Baseline**。$A_t=R_t-b_\phi(s_t)$，用 $\|b_\phi(s_t)-R_t\|^2$ 訓練 baseline。也就是說 baseline 用蒙地卡羅報酬來擬合，不是 TD 的 critic。書裡要講清楚標題和內容不一致的地方。
- p.6 Sample Result：兩條訓練曲線。左邊約 1,500 局，總獎勵從 −500 爬到 0～250，抖得很厲害；右邊約 1,800 局，穩定在 250 上下。
- 繳交：程式 2 分；JudgeBoi 上傳動作序列 `.npy` 4 分（小於 2 MB，評分時重播動作，**最後一個動作必須剛好讓環境結束**，長度不對就退件；沒有 private set）；報告 4 分。**投影片沒寫基準線**。
- 報告：(1) 從 REINFORCE with baseline、Q Actor-Critic、A2C、A3C 擇一實作，說明它和 PG 的差別；(2) MuZero 論文的選擇題（arXiv 1911.08265）。
- 規則：預設不用 GPU；訓練應在 30 分鐘內跑完；不准用額外資料或預訓練模型；助教只保證 Colab 上能重現。

## 重新規劃：為什麼現在可以做
前一版不建議的三個理由，現在都有對策：
1. **範例程式失傳** → 用 2021 官方範例當「助教的程式」來讀，這是最接近的現存版本。另外照投影片自己實作進階演算法；這份作業本來就要學生自己寫 Actor-Critic，所以自己實作也符合題目。
2. **Gym → Gymnasium** → 照 HW07／HW10 的做法，把範例原樣抄出來，只用一層薄 shim 處理 API 差異：`reset()` 改成回傳 `(obs, info)`，`step()` 改成回傳五個值（`terminated`／`truncated`），環境名稱 `LunarLander-v2` → `v3`（`v3` 是哪個 Gymnasium 版本改的、物理有沒有改，Phase 0 要查證）。shim 的每一處改動都寫進書裡。
3. **RL 對種子很敏感** → 本書以多種子為原則：每個設定至少跑 5 個訓練種子，報告平均和範圍，不報單次結果。只用 CPU、`torch.set_num_threads(1)`，24 核可以同時跑約 20 個行程，30 分鐘規格的網格成本也還可以接受。

另外，**本機量得到主指標**：環境本身就是評分器。JudgeBoi 只是用固定種子重播動作序列，本機可以做一模一樣的重播器。

## 本機現況（2026-10-16）
- 共用 `.venv`：Python 3.12.3、torch 2.11.0+cu128、numpy 2.5.3。
- **已安裝**（使用者同意裝進共用 .venv）：gymnasium 1.4.0、box2d 2.3.10、pygame-ce 2.5.8、farama-notifications 0.0.6，`requirements.txt` 加了 `gymnasium[box2d]==1.4.0`。
- CPU 24 核；GPU 不需要。

## 實作設計
程式放在 `HW12/`，比照其他 HW 拆成 .py：
- `HW12/HW12_EN.ipynb`：2021 原版 notebook，不修改。
- `HW12/sample.py`：notebook 照原樣串成腳本，只套 Gymnasium shim。參照工具 `docs/tools/hw12_make_ref.py` 比照 HW13／HW14 的做法，驗證拆檔後的程式與它逐位元相同（同種子下每局獎勵序列和最後的 state_dict 都 `torch.equal`）。
- `HW12/rl.py`（自己寫，名稱 Phase 0 再定）：一支腳本，用 `--algo` 切換演算法，共用同一個策略網路和評估流程：
  - `pg`：範例本身的做法，用瞬時獎勵 r_t 加整批標準化。這其實不是 p.4 的演算法，因為每個動作只看到它當下那一步的獎勵。
  - `pg_rtg`：投影片 p.4，逐步折扣的 reward-to-go。
  - `reinforce_baseline`：投影片 p.5，用蒙地卡羅報酬擬合 baseline。
  - `a2c`：TD 版 critic（n-step／GAE），報告第 1 題的「A2C」。
  - `dqn`（選做）：範例註解裡的「boss」等級。它是 value-based 方法，跟主線的 policy gradient 系列不同，要不要放進書裡，大綱階段再決定。
  - `ppo`：「現在的做法」，只跑少量對照。它和 RLHF 與 GRPO 有直接關係：GRPO 就是用同一組回答的平均分當 baseline 的 REINFORCE-like 方法。
- `HW12/judge.py`：本機版 JudgeBoi。給環境種子和 `.npy`，重播動作，檢查最後一步剛好結束，回傳總獎勵。
- 評估分兩種，書裡要分開講：
  - **開迴路（JudgeBoi 的量法）**：固定種子重播一串動作。分數只反映那幾個種子，學生甚至可以不學策略、直接對那幾個種子搜尋動作序列。這是本書要點出的評分漏洞，只用來分析，不當成做法推薦。
  - **閉迴路（真正的泛化）**：用訓練時沒看過的 100 個種子讓策略即時互動，報告平均獎勵和成功率（LunarLander 通常以 ≥200 分視為解決，出處 Phase 0 查 Gymnasium 文件）。

## 實驗主線（草案，Phase 0 量完範例的速度後再定規格）
| 組 | 內容 | 規模 |
|---|---|---|
| A | 範例原樣（Simple 的起點）＋隨機策略的下限 | 5 種子 |
| B | 回報的形式：整局總獎勵 vs reward-to-go vs 折扣；γ；報酬標準化有無 | 每格 5 種子 |
| C | baseline：p.5 的蒙地卡羅 baseline vs A2C 的 TD critic；n-step／GAE λ | 每格 5 種子 |
| D | 常見技巧：熵正則、梯度裁切、學習率、每批局數 | 挑重點，每格 5 種子 |
| E | PPO 對照；同樣 30 分鐘的 CPU 預算下比樣本效率 | 5 種子 |
| F | 開迴路 vs 閉迴路：同一個策略的 JudgeBoi 分數 vs 100 個新種子的分數；種子搜尋的漏洞 | 分析 |

所有結果寫進 `docs/tools/hw12_runs.jsonl`，事實寫進 `docs/HW12/FACTS.md`。

## 章節草案（ch00–ch08，大綱階段再對照現行 skill 調整）
- ch00 作業與模型總覽：2022 vs 現在（RLHF、GRPO、推理模型的 RL 訓練）；範例程式失傳和本書的取材方式。
- ch01 環境：LunarLander 的狀態、動作、獎勵設計；Gym → Gymnasium shim；隨機策略的分數。
- ch02 策略網路與取樣：Categorical、log_prob；範例的網路架構。
- ch03 範例的訓練迴圈：讀 2021 範例，記下它的怪處和問題。
- ch04 回報怎麼算：reward-to-go、折扣、標準化，以及它們跟變異數的關係（B 組）。
- ch05 Baseline 與 Actor-Critic：p.5 標題與內容的落差；MC baseline vs TD critic（C 組）。
- ch06 讓它穩定：常見技巧與 PPO（D、E 組），接到 RLHF／GRPO 的「現在的做法」。
- ch07 評分與重現：動作序列、開迴路的漏洞、決定性；報告第 2 題 MuZero（只依論文作答）。
- ch08 總結：全部實驗一張表、問題清單、題庫、名詞表、速查表。

## 主要風險
1. **2021 ≠ 2022**：範例只能當「推測的前身」。書裡所有「助教的程式」都要標明是 2021 版。
2. **Gymnasium 的 LunarLander 版本差異**：v2 → v3 若改了物理或獎勵，跟投影片曲線的數值比較就要打折。這要在 ch01 交代。
3. **決定性**：Box2D 加上 CPU torch，在固定種子下應該可以逐位元重現，但要先驗證兩次結果完全相同（HW07 的教訓）。
4. **時間**：實測範例一次約 2.5 分鐘（見上方探測），所以時間不是問題；進階演算法若跑到 30 分鐘的預算上限，每格 10 種子並行仍約 30 分鐘。GPU 完全不用，可以跟 ML2025 repo 的 GPU 工作同時跑。

## Phase 0 要做的事（要下載的先問使用者）
1. ~~下載 2021 範例~~：完成（2026-10-16）。
2. ~~安裝環境~~：完成（共用 .venv）。還要查證 `LunarLander-v3` 和 v2 的差異。
3. ~~做參照版~~：完成（2026-10-16）。
   - `docs/tools/hw12_make_ref.py` 把 EN notebook 的程式格原樣串成腳本。改動的每一行都保留原文，加註 `# [ref]`：`import gym` → `gym_compat`、`set_deterministic` → `use_deterministic_algorithms`、NumPy 2 的三處、Colab／繪圖相關的行。腳本最後把要比對的東西存成一個檔。
   - 拆成 `HW12/gym_compat.py`（自己寫的 shim）、`env_setup.py`、`model.py`、`train.py`（含探索格 [15]–[24]，因為它們會推進環境的亂數）、`test.py`、`judge.py`。
   - `docs/tools/hw12_check_split.py` 比對：訓練曲線兩條、`network` state_dict（`torch.equal`）、動作序列、judge 分數**全部相同**，stdout 逐字相同。參照版單次 2 分 20 秒（`OMP_NUM_THREADS=1`）。
   - 種子 543 的範例結果（主線的起點，正式數字之後寫進 FACTS）：測試 5 局 −34.25、−161.82、−170.98、257.59、−22.56，judge 平均 **−26.40**，動作分布 {2: 749, 3: 341, 1: 316, 0: 128}，每局 312／238／286／479／219 步。依 2021 的門檻，這落在「<0」那一級。
   - notebook 存檔裡的 2021 Colab 輸出也一樣是負的：EN −147.26、ZH −30.32。也就是**範例本身三次都沒過最低的那一級**。
   - 前面的 probe 沒有跑探索格，同樣用種子 543，judge 平均卻是 108.6。主要的差別應該就是這幾格，但還沒單獨驗證；驗證後可以當種子敏感度的例子寫進 ch03／ch07。
4. ~~量範例~~：完成（2026-10-16）。實驗工具 `docs/tools/hw12_exp.py`，預設路徑和參照版逐位元相同（同一套比對）。原始紀錄在 `docs/tools/hw12_p0_runs.jsonl`（Phase 0 版的工具、舊的紀錄格式），執行腳本是 `hw12_phase0.sh`、`hw12_probe.sh`。Phase 1 改寫工具後，正式實驗另寫 `hw12_runs.jsonl`，預設路徑重驗仍逐位元相同。
   - 量法分三種：
     - **judge**：test.py 加 judge.py 的作法，`fix(env, 543)` 後跑 5 局，隨機取樣。
     - **閉迴路**：環境種子 100000–100099 共 100 局，每局重設 torch 種子，分兩種：隨機取樣（stochastic），以及取機率最大的動作（greedy、argmax）。
     - 結局分類：最後一步 ≥ +50 算 landed，≤ −50 算 crashed，被截斷算 timeout。
   - **範例跑 21 個訓練種子**（543 加上 0–19）：

     | | 平均 | 標準差 | 最小 | 最大 |
     |---|---|---|---|---|
     | judge 平均 | −47.7 | 65.6 | −164.5 | 102.6 |
     | 閉迴路 stochastic | −42.4 | 33.2 | −125.3 | 40.2 |
     | 閉迴路 greedy | −6.3 | 34.6 | −80.5 | 48.5 |
     | 最後 50 批的訓練平均 | −60.1 | 10.8 | −82.5 | −39.8 |

     - judge ≥ 0 的只有 **4／21** 個種子。
     - 閉迴路的結局大多是墜毀：stochastic 約 70–92 局 crashed，greedy 約 57–83 局。
     - 種子 4 的策略 greedy 100 局全部墜毀（平均 154 步，學會的是直接往下掉）。
     - greedy 平均比 stochastic 高約 36 分。
   - **隨機策略**（均勻取樣）：judge −162.3，閉迴路 −184.5，100 局全部墜毀，平均 92 步。
   - **探索格的影響已證實**：同樣用種子 543 但不跑 [15]–[24]，judge 從 −26.4 變成 **108.6**，閉迴路 stochastic 從 −34.4 變成 −11.1、greedy 從 2.3 變成 17.9。judge 跳了 135 分，閉迴路只差二十多分，所以 judge 的變化大多是運氣。
   - **計時**：單獨一個行程 130 s，20 個行程同時跑時每個約 250 s（CPU 記憶體頻寬互搶）。正式計時要另外單獨跑。
   - **改報酬的探測**（`hw12_probe.sh`，各 10 個種子 0–9，其他設定同範例；預設路徑再驗一次，仍逐位元相同）：

     | | judge | 閉迴路 stochastic | 閉迴路 greedy | 最後 50 批 | 訓練步數 |
     |---|---|---|---|---|---|
     | 範例（瞬時 r_t，種子 0–9） | −44.0 ± 58.6 | −37.0 ± 40.0 | 2.3 ± 38.8 | −61.5 | 52.6 萬 |
     | 折扣 reward-to-go，γ = 0.99（助教的 medium 提示） | 40.5 ± 66.9 | **40.5 ± 52.3** | −33.6 ± 120.7 | 22.2 | 41.6 萬 |
     | 整局總報酬（每一步都乘同一個數） | −87.2 ± 40.1 | −101.4 ± 29.5 | −222.9 ± 209.3 | −102.5 | 18.2 萬 |

     - reward-to-go 讓 stochastic 平均高了約 78 分，但**沒有一組到 200**：小網路、SGD lr 0.001、2,000 局都太保守。所以主線要靠 baseline、A2C、PPO 和其他技巧才可能到 200。預算要怎麼定（照範例的 2,000 局，或照投影片的「30 分鐘內」）留到 Phase 1 決定。
     - **greedy 會懸停**：rtg 的 10 個種子裡有 5 個的 greedy 策略幾乎 100 局都撐到 1,000 步被截斷（懸在空中不降落）。同一個策略用 stochastic 反而會降落。這是「局部最佳解：不墜毀就好」的直接例子，放 ch04／ch07。
     - 整局總報酬比範例還差：每一步都乘同一個 R，訊號被整局的雜訊蓋掉，策略學會的是快點墜毀（平均每局步數少）。
   - **v2 → v3**（Gymnasium 1.4 原始碼的 Version History）：v3 修了「world 沒有完全銷毀造成的非決定性」，reset 時會重設風與亂流（預設關閉），x、y、速度、角度的 observation 宣告範圍放寬（只影響 `print(env.observation_space)`：2021 印 `Box(-inf, inf, (8,))`）。物理與獎勵沒改。`max_episode_steps=1000`、`reward_threshold=200`。地形與初始推力的亂數流和 gym 0.18 不同，所以 2021 notebook 印出的初始狀態對不上（2021 是 `[0.00396109 1.4083536 …]`，本機是 `[-0.0012619973 1.3984586 …]`）。
5. **定實驗規格與大綱**：`docs/HW12/outline.html` 草案完成（2026-10-16），樣板在 `docs/tools/hw12_book/`（由 HW10 複製，改成月面青；`outline_body.html` 是大綱的內文）。**使用者已核可（2026-10-16）**：
   1. 訓練預算兩種都報：同一次訓練在「2,000 局」和「100 萬環境步」各存一個快照、各評估一次；主表用環境步數，附一欄局數。
   2. DQN 做一組當對照，放第 6 章。
   3. 樣式沿用前幾本的版面，改成月面青。
   4. 不冷讀。
   5. 規格內的實驗和章節可以一路做下去，不必每章停下來。推送仍只在使用者說「推」時才做。
