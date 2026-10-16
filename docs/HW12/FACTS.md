# HW12 教材事實清單（不進最終教材）

每章動筆前先讀這份。章節裡的數字一律出自這裡或 `docs/tools/hw12_*` 的原始紀錄；新查證推翻舊事實時，回頭修正舊章並在這裡註明。

## 來源與環境
- 2022 年範例程式失傳；本書讀 2021 年官方範例 `HW12/HW12_EN.ipynb`（`ga642381/ML2021-Spring/HW12`，2026-10-16 下載，使用者同意）。2022 與 2021 投影片的「Sample Result」是同一張圖 → 2022 很可能沿用（推測，書裡要這樣寫）。
- 2021 投影片 p.7 的配分（JudgeBoi 平均獎勵）：<0 → 2 分、0–99 → 3、100–199 → 4、200–240 → 5、≥241 → 6。2022 投影片沒有門檻。
- 2022 投影片：PG（p.4）3 分、Actor-Critic（p.5）4 分；程式 2 分、動作序列 4 分、報告 4 分；報告題 1（實作進階演算法並說明與 PG 差別，2 分）、題 2（MuZero 選擇題，2 分）；HW12 預設不用 GPU；訓練應在 30 分鐘內完成；`.npy` < 2 MB；長度不符會退件；沒有 private set。
- 2021 notebook pin：`gym[box2d]==0.18.3`、`numpy==1.19.5`、`torch==1.8.1`（Colab，Python 3.7）。
- 本機：Python 3.12.3、torch 2.11.0+cu128（只用 CPU）、NumPy 2.5.3、gymnasium 1.4.0、box2d 2.3.10、pygame-ce 2.5.8；CPU 24 核。
- Gymnasium 1.4 只註冊 `LunarLander-v3`：`max_episode_steps=1000`、`reward_threshold=200`。v3 相對 v2：修「world 沒完全銷毀」的非決定性、reset 時重設風與亂流（預設關）、observation 宣告範圍放寬（x、y ±2.5，速度 ±10，角度 ±2π）。物理與獎勵沒改。

## 拆檔與驗證
- 參照腳本：`docs/tools/hw12_make_ref.py`（EN notebook 程式格原樣串接，改動行標 `# [ref]`）。
- 主線：`HW12/gym_compat.py`（自寫 shim）、`env_setup.py`、`model.py`、`train.py`、`test.py`、`judge.py`。
- `docs/tools/hw12_check_split.py`：訓練曲線兩條、`network` state_dict（`torch.equal`）、動作序列、judge 分數全部相同，stdout 逐字相同（`OMP_NUM_THREADS=1`）。參照版單次 2 分 20 秒。
- 實驗工具 `docs/tools/hw12_exp.py`：預設（`--step_budget 0`）與參照版逐位元相同；Phase 1 改寫後重驗一次（verify3），仍相同。
- 在 NumPy 2／torch 2.11 會壞的地方：`torch.set_deterministic`（已移除）、`np.array(action_list)`（ragged → ValueError）、`np.shape(log_probs)`（requires_grad 的 tensor → RuntimeError）。2021 Colab 輸出顯示當年 `np.shape(log_probs)` 印出 `(448,)`，`np.array(action_list)` 只有 VisibleDeprecationWarning，`torch.set_deterministic` 只有 deprecation 警告。

## 2021 notebook 存檔裡的輸出（當年的 Colab）
- `print(env.observation_space)` → `Box(-inf, inf, (8,), float32)`；初始狀態 `[0.00396109 1.4083536 0.40119505 -0.11407257 -0.00458307 -0.09087662 0. 0.]`；隨機動作 0；一步的獎勵 −0.8588900517154912。
- 訓練前三批的步數：448、515、392（EN 與 ZH 相同）。
- EN：測試 5 局平均 −147.26（server 格重播也是 −147.26），每局 −29.53、−36.44、−194.16、−268.27、−207.91；動作分布 {2: 1144, 1: 516, 0: 30, 3: 501}。
- ZH：測試 260.62、−212.89、11.86、8.02、−219.22，平均 −30.32；動作長度 299、319、241、231、256；動作分布 {1: 278, 2: 698, 3: 297, 0: 73}；訓練時間 674.24 秒（Colab，訓練時不畫圖）。

## 模型與環境（`docs/tools/hw12_facts.py` → `hw12_facts.txt`）
- PolicyGradientNetwork：fc1 (16, 8) 128 + 16、fc2 (16, 16) 256 + 16、fc3 (4, 16) 64 + 4 = **484** 參數；輸入 (N, 8) → (N, 16) → (N, 16) → (N, 4) softmax。
- 本機 `print(env.observation_space)`：`Box([-2.5 -2.5 -10. -10. -6.2832 -10. -0. -0.], [2.5 2.5 10. 10. 6.2832 10. 1. 1.], (8,), float32)`。
- `fix(env, 543)` 後第一次 reset：`[-0.0013 1.3985 -0.1278 -0.5538 0.0015 0.029 0. 0.]`；探索格的 `action_space.sample()` = 3；那一步獎勵 −1.0511407416545058；探索格 [24] 的隨機局 120 步。
- 獎勵（Gymnasium `lunar_lander.py` `step`）：`shaping = -100·√(x²+y²) - 100·√(vx²+vy²) - 100·|θ| + 10·腳1 + 10·腳2`；每步獎勵 = shaping − 上一步 shaping − 0.30（主引擎，離散動作 2）− 0.03（側引擎，動作 1、3）。**結束時直接 `reward = -100`（墜毀、`abs(x) >= 1`）或 `+100`（`not lander.awake`，靜止）——是「設成」不是「加上」**，說明文件卻寫 additional。每一步都會用 `np_random` 抽 2 個 dispersion 亂數（引擎噴射的擾動）。
- 範例（種子 543）的 5 局測試（逐步重算 shaping 差與耗油，和環境的獎勵相符到 1e-3）：

  | 局 | 步數 | 總分 | 結局 | 動作 [0,1,2,3] | shaping 合計 | 主引擎 | 側引擎 | 結束 |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 312 | −34.25 | 墜毀 | [1, 57, 180, 74] | 123.38 | −53.7 | −3.93 | −100 |
  | 1 | 238 | −161.82 | 墜毀 | [1, 61, 129, 47] | −19.91 | −38.7 | −3.21 | −100 |
  | 2 | 286 | −170.98 | 墜毀 | [5, 68, 148, 65] | −22.62 | −44.4 | −3.96 | −100 |
  | 3 | 479 | 257.59 | 降落 | [121, 83, 179, 96] | 216.66 | −53.7 | −5.37 | +100 |
  | 4 | 219 | −22.56 | 墜毀 | [0, 47, 113, 59] | 114.22 | −33.6 | −3.18 | −100 |

  第 3 局逐步（`hw12_facts.txt`）：t=0 起動作 3（機率 0.997），t=280 時只有第二隻腳觸地、t=320 時只有第一隻（彈跳），t=360 兩腳都觸地，之後幾乎都選動作 0（機率 0.919），靜止到 t=478 得 +100。
- 隨機策略（均勻）在閉迴路 100 個種子：平均 −184.5、標準差 115.1、最小 −588.2、最大 24.0、平均 92 步、100 局全墜毀。
- **Gymnasium 內建的手寫控制器 `heuristic`** 在同樣 100 個種子：平均 **225.5**、標準差 118.0、最小 −221.5、最大 313.8、≥200 的比例 0.86、平均 274 步，降落 89、墜毀 8、截斷 3。當成「不用學習也能做到」的參考線。

## Phase 0 實測（舊格式紀錄 `docs/tools/hw12_p0_runs.jsonl`）
- 範例 21 個種子（543、0–19），2,000 局：judge −47.7 ± 65.6（−164.5 到 102.6），只有 4／21 ≥ 0；閉迴路 stochastic −42.4 ± 33.2；greedy −6.3 ± 34.6；最後 50 批訓練平均 −60.1 ± 10.8。
- 種子 543：judge −26.40（−34.25、−161.82、−170.98、257.59、−22.56），閉迴路 stochastic −34.4、greedy 2.3。
- 不跑探索格（種子 543）：judge 108.6、stochastic −11.1、greedy 17.9。
- 隨機策略：judge −162.3。
- 探測（種子 0–9）：reward-to-go γ = 0.99 judge 40.5 ± 66.9、stochastic 40.5 ± 52.3、greedy −33.6 ± 120.7（5 個種子 greedy 幾乎全部被截斷，懸停）；整局總報酬 judge −87.2、stochastic −101.4、greedy −222.9。
- 計時：單一行程約 130 s；20 個並行時每個約 250 s。

## Phase 1 之後的正式實驗（`docs/tools/hw12_runs.jsonl`）
- 工具：`hw12_exp.py`（Phase 1 版，`--algo pg|ppo|dqn|random`；每次訓練兩個快照：`ep` = 完成 2,000 局的那次更新之後，`steps` = 環境步數達 1,000,000 的那次更新之後；評估用自己的環境並還原所有亂數）、`hw12_grid.sh`（20 個行程並行）、`hw12_queue.sh`、`hw12_specs/*.txt`、`hw12_summarize.py`。權重存在 `docs/tools/hw12_models/`（不進 git）。
- 預設路徑重驗（verify3）：與參照版逐位元相同；種子 543 的 ep 快照 judge −26.4、stochastic −34.4、greedy 2.3（= Phase 0）。

### 試跑（pilot、pilot2；種子 0–2；只用來選規格，不進主表）
- pilot（critic 每次更新 Adam 1e-3 跑 5 步）：GAE λ 0.95 + Adam 3e-3 三個種子 ep 快照 judge −87.8、stochastic −78.5、greedy −113.1，**greedy 100 局有 98–99 局被截斷（懸停）**；hidden 64 也一樣（−95.0）。MC baseline + Adam 3e-3：judge 133.0、stochastic 133.2；reward-to-go + Adam 3e-3：stochastic 121.6、greedy 21.8（83% 截斷）。PPO（hidden 64、lr 3e-4）1M 步 stochastic 129.3（−5.4 到 241.0）；DQN（hidden 64、lr 5e-4）1M 步 166.9（103.1 到 214.4）。
- pilot2（critic 50 步）：GAE λ 0.95 → ep stochastic 30.4、1M 步 66.2；**GAE λ 0.99 → ep judge 182.0、stochastic 178.9**；n-step 20 → stochastic 104.3；MC → 154.8；MC + SGD 0.001（p.5 照字面、範例的優化器）→ 88.6；GAE 0.95 + 熵 0.01 → 71.8。
- 解讀（假設，待 C 組的 ci5 對照驗證；第 5 章）：critic 擬合不足時 V ≈ 0，GAE 的優勢 ≈ 以 γλ ≈ 0.94 折扣的短視報酬（有效視野約 16 步），看不到 +100 的降落獎勵 → 懸停。critic 練夠、或 λ 拉高到 0.99，就恢復。

### A 組（範例與它的旋鈕；`hw12_specs/A.txt`，2026-10-16）
- A_sample ep（n=21，平均 2000 局、522568 步、train 320 s）：judge -47.7 ± 65.6（-164.5 到 102.6）；stochastic -42.4 ± 33.2（-125.3 到 40.2）；greedy -6.3 ± 34.6（-80.5 到 48.5）；greedy 結局 降落/墜毀/截斷 26.6/70.7/2.7；stochastic 降落 17.2；≥200 比例 greedy 0.21
- A_sample steps（n=21，平均 3343 局、1000961 步、train 625 s）：judge -42.4 ± 61.8（-114.6 到 97.2）；stochastic -42.1 ± 52.8（-128.6 到 97.9）；greedy -33.2 ± 53.7（-125.0 到 96.8）；greedy 結局 降落/墜毀/截斷 25.0/70.6/4.3；stochastic 降落 21.0；≥200 比例 greedy 0.18
- A_nonorm ep（n=10，平均 2000 局、413259 步、train 250 s）：judge -21.1 ± 92.9（-230.8 到 85.8）；stochastic -71.7 ± 81.6（-302.6 到 12.8）；greedy -55.6 ± 86.3（-297.0 到 32.6）；greedy 結局 降落/墜毀/截斷 17.0/79.9/3.1；stochastic 降落 12.9；≥200 比例 greedy 0.15
- A_nonorm steps（n=10，平均 3908 局、1000715 步、train 612 s）：judge -53.2 ± 86.6（-188.5 到 79.6）；stochastic -47.2 ± 39.6（-120.3 到 15.7）；greedy -57.4 ± 49.0（-142.6 到 29.5）；greedy 結局 降落/墜毀/截斷 13.7/78.0/8.3；stochastic 降落 14.7；≥200 比例 greedy 0.10
- A_lr1e-4 ep（n=10，平均 2000 局、214557 步、train 125 s）：judge -205.0 ± 53.8（-278.5 到 -99.9）；stochastic -229.3 ± 17.0（-255.3 到 -203.8）；greedy -817.6 ± 111.1（-883.0 到 -503.4）；greedy 結局 降落/墜毀/截斷 0.0/100.0/0.0；stochastic 降落 0.0；≥200 比例 greedy 0.00
- A_lr1e-4 steps（n=10，平均 6346 局、1000909 步、train 585 s）：judge -142.5 ± 29.7（-196.3 到 -98.2）；stochastic -138.4 ± 19.8（-174.4 到 -105.8）；greedy -958.1 ± 196.1（-1361.1 到 -594.0）；greedy 結局 降落/墜毀/截斷 0.1/62.4/37.5；stochastic 降落 0.1；≥200 比例 greedy 0.00
- A_lr1e-2 ep（n=10，平均 2000 局、204538 步、train 119 s）：judge -428.6 ± 389.4（-1068.0 到 23.3）；stochastic -395.3 ± 326.6（-883.0 到 0.6）；greedy -394.7 ± 327.5（-883.0 到 7.2）；greedy 結局 降落/墜毀/截斷 2.6/97.4/0.0；stochastic 降落 2.7；≥200 比例 greedy 0.02
- A_lr1e-2 steps（n=10，平均 9691 局、1000460 步、train 580 s）：judge -345.1 ± 394.9（-1068.0 到 15.9）；stochastic -314.3 ± 321.3（-883.0 到 -19.2）；greedy -317.5 ± 319.3（-883.0 到 -3.6）；greedy 結局 降落/墜毀/截斷 5.0/94.0/1.0；stochastic 降落 5.4；≥200 比例 greedy 0.04
- A_random ep（n=1，平均 2000 局、0 步、train 0 s）：judge -162.3 ± 0.0（-162.3 到 -162.3）；stochastic -184.5 ± 0.0（-184.5 到 -184.5）；greedy -184.5 ± 0.0（-184.5 到 -184.5）；greedy 結局 降落/墜毀/截斷 0.0/100.0/0.0；stochastic 降落 0.0；≥200 比例 greedy 0.00
- A_noexplore ep（n=1，平均 2000 局、471776 步、train 240 s）：judge 108.6 ± 0.0（108.6 到 108.6）；stochastic -11.1 ± 0.0（-11.1 到 -11.1）；greedy 17.9 ± 0.0（17.9 到 17.9）；greedy 結局 降落/墜毀/截斷 28.0/70.0/2.0；stochastic 降落 16.0；≥200 比例 greedy 0.26
- A_noexplore steps（n=1，平均 3535 局、1000430 步、train 495 s）：judge 102.8 ± 0.0（102.8 到 102.8）；stochastic -0.3 ± 0.0（-0.3 到 -0.3）；greedy 46.4 ± 0.0（46.4 到 46.4）；greedy 結局 降落/墜毀/截斷 38.0/54.0/8.0；stochastic 降落 24.0；≥200 比例 greedy 0.26
- A_sample ep 快照：judge 與閉迴路 stochastic 的相關係數 0.53（21 個種子）；judge ≥ 0 的種子 ep 4／21、steps 4／21。
- A_sample 的 ep 快照與 Phase 0 的數字完全相同（同一條程式路徑）。種子 543 的 1M 快照：judge 97.2、stochastic −6.5、greedy 2.7（3,330 局）。
- 範例訓練到 100 萬步（約 3,343 局，多 1.6 倍）沒有變好：stochastic −42.4 → −42.1。
