# HW01 教材事實清單（維護筆記，不進教材）

教材對應 commit：`717528b`（HW01 程式碼最後變動於 `81d1817`）。
原始碼根目錄：`HW01/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW01/HW01.ipynb`。

## 全書約定
- 檔名：index, outline, ch00–ch07, appendix。
- 圖例「色彩 = 角色」：
  - 藍 `#58a6ff` = CPU 端資料（pandas / numpy / Dataset）
  - 綠 `#3fb950` = GPU 上的運算（模型、tensor 計算）
  - 金 `#e3b341` = 磁碟檔案（csv、model.ckpt、pred.csv、runs/）
  - 紫 `#a371f7` = 控制與超參數（config、迴圈控制）
- Python 程式碼用 `<pre class="py">`。
- listing 的 `data-hot` 寫**原始碼行號**（與 figcaption 的 `檔名:起–迄` 同一套）；enhance.js 依 figcaption 起始行換算（2026-10-03 修正，之前會標錯行）。

## 環境（實測 2026-10-02）
- Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.3、pandas 3.0.6。
- GPU：NVIDIA RTX PRO 4000 Blackwell（sm_120），WSL2。原 requirements 的 torch 1.13.1+cu116 不支援。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW01/` 內用 `../.venv/bin/python train.py`。

## 資料
- covid.train.csv：2699 列 × 118 欄；covid.test.csv：1078 列 × 117 欄（少最後一欄 tested_positive）。
- 欄位：0 = id；1–37 = 37 州 one-hot（AL…WA，每列恰好一個 1）；
  38–53 = 第 1 天 16 欄；54–69 第 2 天；70–85 第 3 天；86–101 第 4 天；102–117 第 5 天。
- 每天 16 欄 = cli, ili, hh_cmnty_cli, nohh_cmnty_cli（4 類症狀）+ 8 行為 + 3 心理 + tested_positive。
- pandas 讀進來重複欄名會改成 `cli.1`、`cli.2`…
- train id：最小 0、最大 2699（2699 列 → 中間缺一號）；test id 0–1077。
- tested_positive（第 1 天欄）平均 9.60、範圍 0.345–30.30。

## 程式事實
- `select_all=True` → feat_idx = range(117)，**含 id 欄**。
- `train_valid_split`：valid = int(0.2 × 2699) = 539；train = 2160。註解寫 `train_size * valid_ratio` 是錯的。
- `random_split` 用 `torch.Generator().manual_seed(seed)`，與全域 seed 無關。
- 回傳 `np.array(Subset)` → 逐筆取出組成 (2160, 118) 陣列。
- batch_size 256 → train 每 epoch 9 個 batch（8 滿 + 1 個 112）；valid 3 個 batch（256,256,27）；test 5 個 batch。
- 模型參數量：117·16+16 = 1888；16·8+8 = 136；8·1+1 = 9；共 2033。model.ckpt 11005 bytes。
- 優化器 SGD lr 1e-5 momentum 0.9；loss MSELoss(mean)。
- early_stop 400：連續 400 個 epoch valid loss 沒有創新低就停。
- `config.py` device = "cuda" 是**刻意設計**（fail fast and loud，不要 CPU fallback）；教材以中性描述，不列為問題或練習。import 時印 `True` 與 `0`。
- `valid_loader` shuffle=True（不影響 loss 平均值？每 batch 平均再平均，batch 大小不同時結果會隨分組微變）。
- `train.py` 重複 import tqdm；`random_split` 在 train.py import 但沒用。
- predict.py 為了拿 input_dim 重讀並切分 train。

## 實測執行（2026-10-02 21:34，seed 5201314）
- train.py：real 34.2 s；Epoch 1 train 134.2442 / valid 107.2155；Epoch 2 69.8929 / 50.8182。
- 最佳 valid loss 1.6611 於 epoch 1483；epoch 1883 early stop；共存檔 49 次。
- 兩次執行結果相同（17:39 與 21:34 都是 1.661 / 1883）。
- predict.py：5 個 batch，pred.csv 1079 行（header + 1078）。
- runs/ 下產生 `Oct02_21-34-17_ValtecBlackwell` 這類目錄（SummaryWriter 預設命名：月日_時-分-秒_主機名）。
- 驗證 loss 軌跡：ep20 12.4151、ep50 10.0611、ep100 4.2813、ep300 2.6302、ep1000 3.8595。
- 沒 GPU（CUDA_VISIBLE_DEVICES=""）時 `import config`：先印 False，再 `RuntimeError: No CUDA GPUs are available`。
- covid.train.csv / covid.test.csv 雖被 HW01/.gitignore 的 `*.csv` 涵蓋，但早已 commit，git 仍追蹤。

## ch01 查證（2026-10-02）
- 各州列數 53–91（平均 ~73）；第 5 天陽性率州平均最低 CT 3.23、最高 MS 22.15。
- 無缺值；.values → numpy float64 (2699,118)。pandas 後綴：無=第1天、.1=第2天…、.4=第5天。
- **滑動視窗**：同州按 id 排序，相鄰列錯開一天（29/37 州完全連續；其餘 8 州各 1–2 處斷）。FL：id 0,32,63,67…
- 缺號 id 2239。test id 0–1077 另一套編號。test 第 1 天問卷出現在 train 的只有 36/1078 列。
- 與答案（tested_positive.4）相關：tp.3 0.985、tp.2 0.970、tp.1 0.953、tp 0.935、hh_cmnty_cli.4 0.898；id 0.264；wearing_mask.4 −0.037。
- 圖例另加：紅 `#f85149` = 預測目標（答案欄）。

## 已在前面章節定義過的名詞（後續章節不必重講，可簡短回指）
- ch00：迴歸、MSE、Kaggle 與四條基準線、PyTorch、CUDA、compute capability/sm、kernel、venv、uv、
  超參數、seed（亂數種子）、驗證集/訓練集/測試集、epoch、batch、early stopping（概念）、
  weights/參數、checkpoint、stdout/stderr、`2>`、`VAR=x cmd`、`from X import *`、Jupyter 筆記本。
- ch01：pandas 基本語法（read_csv、d[col]、iloc、布林篩選、groupby、corr）、python -c 多行、np.isclose、one-hot、滑動視窗、相關係數（Pearson）、標準化（概念）、pandas DataFrame、`.values`、重複欄名後綴、`if __name__ == '__main__'`、0 起算 vs 1 起算欄號。
- ch02：偽亂數（pseudo-random）與「每個函式庫各有獨立的亂數產生器」、cuDNN（deterministic／benchmark 旗標）、cuBLAS（一句帶過）、確定性演算法、docstring 與 `__doc__`、`torch.Generator`、`random_split`／`Subset`（indices）、duck typing、`int()` 截斷 vs 四捨五入、x（特徵）／y（目標）慣例、NumPy 整數陣列索引（fancy indexing）、`feat_idx`、全域產生器（沒指定 generator 時共用的那一個）、Dataset（只先粗略定義為「能 len() 與 [i] 的容器」，正式在 ch03）。
  ch02 圖號：圖 2.1 本章資料流、圖 2.2 random_split 打亂再切、圖 2.3 select_feat 欄位選擇。
- ch03：tensor（PyTorch 多維陣列，可上 GPU、可算梯度）、float32 vs float64（4 vs 8 bytes、約 7 位有效數字）、`torch.FloatTensor` 會複製 vs `torch.from_numpy` 共用記憶體、繼承（class X(Dataset)）、`__init__`／`self`、特殊方法（dunder：`__getitem__`、`__len__`）、tuple、0 維 tensor、`.item()`、sampler（RandomSampler）、collate（batch 是 list）、`len(loader)` 無條件進位、`drop_last`、`num_workers`、行程（process）、DataLoader 的 shuffle 用 PyTorch 全域產生器（每 epoch 新順序、每次執行相同）、pinned（page-locked）memory、`non_blocking`、GPU 有自己的記憶體（搬上 GPU = 複製）、`enumerate` 與 test loader 必須不打亂、`reduction='mean'/'sum'`、「份量」（平均時的加權比例，刻意不叫權重以免和模型權重混淆）、系統性偏差 vs 雜訊、第 5 百分位／中位數、選擇偏差（winner's curse）、TensorDataset。
  ch03 圖號：圖 3.1 Dataset／DataLoader 資料流、圖 3.2 三個 loader 的 batch 切法、圖 3.3 同一 checkpoint 的驗證 loss 量測分布（1.661 vs 2.0685）。
  ch03 已完整解釋 1.661 vs 2.069（§3.5「三層」：份量放大 → valid 打亂造成雜訊 → 挑最小值）；後面章節回指 3.5 節即可。
- ch04：`nn.Module`（所有模型與層的父類別）、「名冊」＝指定屬性時自動登記子模組（`__setattr__`、`_modules`）、`super(My_Model, self).__init__()` 與 `super().__init__()`（同義，後者是 Python 3 簡寫）、`nn.Sequential`（容器模組，子模組名為 0、1、2…）、`nn.Linear` 的 weight 形狀 (輸出, 輸入) 與計算式 x·Wᵀ+b、轉置、`named_parameters()`、參數名稱（`layers.0.weight`）＝名冊路徑＝state_dict key、`requires_grad`（一句帶過，梯度在 ch05）、預設初始化 U(−1/√in, 1/√in) 與「按輸入個數縮放」的理由、`__call__`（`model(x)` → `nn.Module.__call__` → `forward`）、hook（一句）、`Module.to` 原地搬移 vs `tensor.to` 回傳新 tensor、`squeeze(1)` vs `squeeze()`、`unsqueeze`、廣播（broadcasting）、UserWarning、dying ReLU（永遠輸出 0 的單元）、「參數量是容量上限，不是實際用上的量」、`torch.load` + `load_state_dict`（依名字複製）、mat1/mat2 錯誤訊息的讀法、LeakyReLU／GELU（一句）、`map_location`、`weights_only`（回指目錄頁）。
  ch04 圖號：圖 4.1 layers.0.weight (16,117) 與 x·Wᵀ+b、圖 4.2 model(x) 的呼叫鏈與逐層形狀、圖 4.3 少了 squeeze 的廣播 (256,256)、圖 4.4 id 欄的 |w| 與 |w·x| 對照。
  ch04 已講完 dead ReLU 與 id 約 400 倍（§4.7），後面章節回指即可；predict.py 為了 input_dim 重讀資料只在 §4.8 帶過，細節留給 ch06。
- ch05：criterion（nn.MSELoss 物件，`criterion(pred, y)`）、優化器（optimizer，只改交給它的參數）、SGD（stochastic gradient descent，「隨機」= 每步只看一個 batch）、學習率（lr）、momentum 與 momentum buffer（buf ← 0.9·buf + grad、w ← w − lr·buf；第一步 buf = grad；記在 optimizer 身上）、`print(optimizer)` 各欄位、Parameter Group、weight_decay（= SGD 內建的 L2，細節 ch07）、梯度（gradient：loss 對每個參數的斜率，存在 `.grad`，形狀與參數相同）、偏導數（一句）、梯度下降（w ← w − lr·g）、自動微分（autograd）、計算圖（computational graph）、`grad_fn`（SqueezeBackward1、MseLossBackward0）、連鎖律（chain rule，一句）、反向傳播（backpropagation）、梯度累加與 `zero_grad`（`set_to_none=True` 預設，清成 None）、計算圖在 backward 後釋放、`.detach()`（在 train.py:61 是多餘的）、`model.train()`／`eval()` 只切 `training` 旗標、`torch.no_grad()`（不建計算圖）、`torch.inference_mode()`（一句）、`math.inf`、`torch.save(state_dict)` 覆寫同一檔、early stopping 的計數器（耐心）與 n_epochs 硬上限、SummaryWriter、`add_scalar(tag, value, step)`、tag 依斜線分組、橫軸 step = 更新次數（每 epoch 9）、tqdm `set_description`／`set_postfix`／`leave`／`position`、AdamW（一句）、梯度裁剪 `clip_grad_norm_`（一句）、`writer.close()`、`os.makedirs(exist_ok=True)`。
  ch05 圖號：圖 5.1 trainer() 的結構、圖 5.2 五步驟各讀寫什麼（.grad／計算圖／權重／momentum）、圖 5.3 49 次存檔的時間軸、圖 5.4 15 個 epoch 的 train／valid loss（兩軸對數）。
  ch05 已講完五步驟、梯度、momentum、train loss「邊更新邊記錄」、第 1483 個 epoch 的幸運分組（§5.7）、lr／momentum 對照組與 lr 1e-4 輸出常數 9.731（§5.10），後面章節回指即可。
  ch05 early_stop 200（**已實測**，2026-10-03）：最後一次存檔是 `Epoch [1260/3000]: Train loss: 1.7980, Valid loss: 1.7165`（印出 `Saving model with loss 1.716...`），在第 1460 個 epoch 停止，共存檔 48 次，跟從存檔清單推算的結果一致（第 1260 個 epoch 之前最長的存檔間隔是 686→860 的 174）。
  注意：上方「ch05 實測」寫「loss=509 就是 ch00 進度條上的 loss=509」，但 ch00 只引用了每個 epoch 結束時的進度條（loss=60.4），沒有 509／136；ch05 §5.5 改寫成「跑完第 1 個 batch 時短暫顯示 loss=509」。
- ch06：推論（inference）、函式簽名（一句）、pickle／UnpicklingError（weights_only 只放行純資料）、OrderedDict（記住插入順序的 dict）、checkpoint 存成字典（input_dim＋feat_idx＋state_dict）、從 `layers.0.weight.shape[1]` 反推 input_dim、`.cpu()`（GPU→主記憶體）、`.numpy()` 只能用在 CPU tensor、host memory、`torch.cat(dim=0)`（回指 ch04 動手做）、md5（檔案指紋）、`with open(file, 'w')` 清空重寫、`csv.writer`／`writerow`／`lineterminator`／`csv.excel`、float32 的最短往返字串表示（位數不固定、不代表精度）、bytes 與 `b'...'`、二進位模式 `'rb'`、行尾 LF／CRLF、RFC 4180、文字模式的行尾轉換與 Windows 上的 `\r\r\n`、`newline=''`、變異數（＝每筆都猜平均的 MSE）、標準差（變異數開根號，一句）、「MSE(預測, 第 4 天)」這把替代尺（測試集沒答案時用）、獨立的兩數相減 → 約 2 倍變異數、外推／內插、ReLU 網路在範圍外線性延伸、`wc -l`、`md5sum`、`tail -N`、`repr`、`np.arange`、`np.array_equal`、`np.random.default_rng(seed).permutation`。
  ch06 圖號：圖 6.1 predict.py 的資料流（上排「只為了 117」的繞路）、圖 6.2 predict() 的形狀變化（5 個 batch → cat → (1078,)）、圖 6.3 pred.csv 行尾位元組（Linux 實測 CRLF／Windows \r\r\n／newline=''）、圖 6.4 訓練答案／測試第 4 天／預測的平均與最大值。
  ch06 已講完：predict.py 重讀訓練資料的兩個問題（依賴訓練資料、117 是當下重算）與存字典的解法、weights_only 預設實際為 True、pred.csv 格式（CRLF、最短表示、id 對齊）、打亂 id 的代價 1.0045 → 120.0056、測試集是高陽性率時期與外推（§6.9）。後面章節回指即可；改進模型留給 ch07。
  ch06 的 4 個 `TODO(本機實測)` 都在 §6.10 動手做：第 1 項 wc／md5sum 逐字輸出、第 2 項 `repr(csv.excel.lineterminator)` 逐字輸出、第 3 項整段輸出（並確認 permutation 方式是否重現 120.0056）、第 4 項輸出（預期 `OrderedDict 6 117`）。
- ch07：Kaggle public／private 排行榜（一句）、基準線分數是**測試集** MSE（不可與驗證集 MSE 比，本書沒上傳 Kaggle）、特徵組合短名 noid／survey／corr／tp4（corr 只在訓練部分算相關係數）、`np.corrcoef`（一句）、標準化插在 train.py 第 114 行之後（std 為 0 改成 1；ch01 版本是 +1e-8）、Adam（每個參數依梯度平方的移動平均自適應步幅）、AdamW（decoupled weight decay；PyTorch 的 AdamW 不寫 weight_decay 時預設 0.01）、L2 正則化（loss ＋ λ/2·Σw²，梯度多 λw）、SGD／Adam 的 `weight_decay`（加到梯度上）、Dropout（訓練時以機率 p 丟、乘 1/(1−p)；由 train()／eval() 切換）、GELU 的「永遠為 0」判準與 ReLU 不同（float32 下溢）、過擬合（overfitting）、「驗證方式決定你選到什麼模型」、判讀規則「差距小於 DNN 換 seed 的標準差（0.01～0.06）不算數」、checkpoint 存 feat_idx＋mean／std（`.tolist()`）＋state_dict（map_location 到 GPU 時 list 不受 `.numpy()` 限制）、題庫在 repo 根目錄 `cp -r HW01 HW01-lab` 的複本裡做（`../.venv` 仍可用）、Pipeline／TimeSeriesSplit（一句，現在的做法框）、梯度提升樹（回指目錄頁）。
  ch07 圖號：圖 7.1 作業提示對應的程式位置、圖 7.2 隨機切分上的完整比較（橫條圖，虛線 1.166）、圖 7.3 兩種切分下 8 個設定的名次（紅＝時間切分 train MSE < 0.75，藍＝> 0.95）、圖 7.4 4 個 seed 的真實 valid MSE 點圖。
  ch07 已講完：作業提示對應位置、所有「第 7 章再談」的承諾（§7.1 表）、1.240（ch02，原版量法挑選）vs 1.1848（量法修正後）是兩次不同訓練、ch04 的 10 個 vs ch07 的 12 個沒在工作的單元同理。後續（appendix）回指即可。
  ch07 的 8 個 `TODO(本機實測)` 都已在審稿時實測填入，數字見下方「ch07 審稿補測」。

## Baseline 實測（2026-10-03，與範例同一驗證集：random_split seed 5201314）
- 抄第 4 天 tested_positive（第 101 欄）：MSE 1.313
- 線性迴歸 4 個 tp 欄（53,69,85,101）：1.303
- 線性迴歸 116 欄（拿掉 id）：1.166；117 欄含 id：1.172
- 範例 DNN：印出的最佳 1.661，在 539 筆上一次算的真實 MSE 是 2.069（見「ch03 實測」）。跟 baseline 比時要用 2.069。
- 全書開頭（index.html#now）已寫「過時三層次」；各章遇到過時寫法要加「現在的做法」框；ch07 要延伸 baseline 比較。

## 時間切分實測（每州依 id 排序，前 80% 訓練／後 20% 驗證）
- 2143／556 筆；線性迴歸 116 欄 1.21；抄第 4 天 1.241。
- 標準化（只用訓練集統計量）：訓練集 mean 0.0 / std 1.0；驗證集 0.011 / 1.002。
- uv 專案流程實測：uv init --bare + uv add torch==2.11.0 --index pytorch-cu128=… → 解析 30 個套件，lock 內 torch 2.11.0+cu128。

## 全書結構約定（使用者要求，2026-10-03）
- **每個 HW 的 ch00 都要有「模型總覽」一節**（架構圖、各層 shape、參數量、所在檔案）；模型的逐行細講仍在後面的模型章。HW01：ch00 §0.2，圖 0.1；原檔案地圖改為圖 0.2、§0.8。

## ch02 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
執行方式：在 `HW01/` 內 `PYTHONPATH=. ../.venv/bin/python <腳本>`，import `utils.py` 的函式。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

`train.py` 開頭的 stdout，逐字照抄，含行尾空白與縮排（後面接 tqdm 進度條和 Epoch 1 的 loss，見上方「實測執行」）：
```
True
0
train_data size: (2160, 118) 
    valid_data size: (539, 118) 
    test_data size: (1078, 117)
number of features: 117
```
- 第 1、2 行來自 `config.py` import 時的 print。`train_data size` 那行行尾有一個空格，下一行開頭有 4 格縮排，因為 f-string 用三引號跨行。

same_seed：
- docstring 寫成 `""""`（4 個引號），所以 `__doc__` 實際是 `'" Fixes random number generator seeds for reproducibility. '`，開頭多一個 `"`。
- cudnn 旗標：呼叫前 deterministic=False、benchmark=False；呼叫後 deterministic=True、benchmark=False。
- 呼叫 `same_seed(5201314)` 後，`torch.rand(3)` = [0.5436, 0.9728, 0.8315]，`np.random.rand(3)` = [0.6076, 0.9134, 0.281]。重設 seed 再取一次，結果完全相同。
- **不會**設定 Python 內建的 `random` 模組，因為 same_seed 沒有呼叫 random.seed。
- 有 `torch.cuda.is_available()` 判斷，在沒 GPU 的機器上也不會報錯。但 config.py 的 device="cuda" 在別處會失敗，見上方。

train_valid_split（seed 5201314）：
- `0.2*2699` = 539.8000000000001，經 `int()` 截斷成 539，不是四捨五入。train 2160。
- `random_split` 的回傳型別是 `Subset`，長度 2160 和 539。
- train indices 前 10 個：[696, 2073, 909, 1475, 2165, 2487, 479, 1795, 2150, 2588]（未排序）。
- valid indices 前 10 個：[1995, 1042, 86, 1667, 464, 481, 1040, 85, 423, 788]。因為 id 欄 = 列號，valid 的 id 前 10 個也是這組數字。
- 同一個 seed 重切一次，結果相同。改用 seed 1 時，train 前 10 個是 [1659, 792, 971, 1755, 777, 408, 450, 248, 1528, 1631]。
- 先改全域 `torch.manual_seed(0)` 再切，結果仍與原本相同，因為切分用的是獨立的 Generator。
- 不傳 generator 時，切分依賴全域 seed：全域 seed 相同，切分就相同；全域 seed 不同，切分就不同。
- `random_split(data, [0.8, 0.2], ...)`（比例寫法，torch ≥1.13 支援）切出來也是 [2160, 539]。
- `np.array(Subset)` 得到 (2160,118) float64，耗時約 0.0015 s。`data[subset.indices]` 結果完全相同（array_equal True），耗時約 0.0012 s。這組資料量太小，兩種寫法的速度看不出差別。
- valid 裡 37 州都有出現，每州 7–25 列。
- y_valid 平均 10.1355、範圍 0.3448–29.8157；y_train 平均 9.7404。

select_feat：
- select_all=True：x_train (2160,117)、x_valid (539,117)、x_test (1078,117)、y_train (2160,)、y_valid (539,)。
- select_all=False：feat_idx [0,1,2,3,4] = 欄名 ['id','AL','AK','AZ','AR']，也就是 id 加 4 個州的 one-hot，全部不是有用的特徵。shape 分別是 (2160,5) (539,5) (1078,5)。
- test 沒有答案欄，所以 `raw_x_test = test_data` 直接使用全部 117 欄。train[:, :-1] 也是 117 欄，兩邊對齊；第 116 欄是 worried_finances.4。

## ch02 審稿補測（2026-10-03 本機）
- y_train：平均 9.7404、範圍 0.4545–30.3046。
- 動手做第 1 題逐字輸出：`(2160, 118) (539, 118)`，接著 `[1995. 1042.   86. 1667.  464.  481. 1040.   85.  423.  788.]`。
- 先執行 `torch.manual_seed(0)` 再切，valid 前 10 個 id 不變。seed 1 時 valid 前 10 個 id：`[1265. 2071.  158. 1588. 1822. 1955. 1547.  969.  476.  856.]`。
- **拿掉 id**（utils.py:32 改成 `feat_idx = list(range(1, raw_x_train.shape[1]))`）：116 個特徵；最佳 valid loss 0.982，出現在 `Epoch [1369/3000]: Train loss: 1.0857, Valid loss: 0.9820`；第 1769 個 epoch early stop；共存檔 55 次。
- **保留 id 但除以 2699**（117 欄）：最佳 valid loss 1.003，在第 1367 個 epoch。結論：id 拖累的主因是數值尺度太大（id 最大 2699，次大的欄位是 wearing_mask 的 89.8），程式又沒有做標準化。
- 這兩組實驗都在暫存複本裡跑，repo 的 models/model.ckpt 沒有被動到。

## ch03 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
執行方式：在 `HW01/` 內 `PYTHONPATH=. ../.venv/bin/python <腳本>`，照 train.py 的順序呼叫 same_seed → split → select_feat → COVID19Dataset → DataLoader。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

COVID19Dataset（data_loader.py）：
- `torch.FloatTensor(x)` 把 float64 轉成 float32，而且會**複製**：之後改 numpy 陣列，tensor 不會跟著變。對照組 `torch.from_numpy` 會共用記憶體，dtype 維持 float64。
- x_train 佔用的記憶體：numpy float64 是 2,021,760 bytes，tensor float32 是 1,010,880 bytes，剛好一半。
- float32 的誤差：x_train 全部數值的最大絕對誤差 3.81e-06，最大相對誤差 5.94e-08。例：y_train[0] 在 float64 是 3.7109291，在 float32 是 3.7109291553497314。
- `isinstance(train_dataset, Dataset)` 為 True。self.x 是 (2160,117) float32，self.y 是 (2160,) float32；test 的 dataset 的 self.y 是 None。
- len：train 2160、valid 539、test 1078。
- `train_dataset[0]` 回傳 tuple，長度 2：x 的 shape (117,)，y 是 0 維 tensor（shape ()），值 3.7109。x 前 3 個值是 [696.0, 0.0, 0.0]，第一個是 id 696，對應 ch02 train indices 的第一個。
- `test_dataset[0]` 只回傳一個 Tensor，shape (117,)。
- 切片也能用：`train_dataset[0:3]` 回傳 x (3,117) 和 y (3,)。`train_dataset[-1]` 的 y 是 2.1545。

DataLoader（batch_size 256）：
- len(loader)：train 9、valid 3、test 5。2160/256 = 8.4375、539/256 = 2.105…、1078/256 = 4.21…，都是無條件進位。
- 每個 batch 的大小：train 是 256×8 + 112；valid 是 256, 256, 27；test 是 256×4 + 54。
- 一個 batch 的型別是 **list**（不是 tuple），長度 2：x (256,117) float32、y (256,)。`pin_memory=True` 時 `is_pinned()` 為 True，device 仍是 cpu；`pin_memory=False` 時 is_pinned 為 False。
- 預設值：num_workers 0、drop_last False；shuffle=True 時 sampler 是 RandomSampler。設 `drop_last=True` 時 train 的 len 是 8。
- shuffle 的順序依照 train.py 的實際流程：same_seed → split → `My_Model(117).to('cuda')`（建立模型會先消耗全域亂數）→ 開始迭代。第 1 個 epoch 第一個 batch 的前 8 個 id 是 [2285, 2508, 1402, 559, 1361, 2114, 469, 1572]；第 2 個 epoch 是 [385, 383, 712, 792, 821, 1382, 971, 665]，每個 epoch 順序都不同。
- pin_memory 的速度：迭代 10 個 epoch 並 `.cuda()`，pin 0.038 s、不 pin 0.039 s。這份資料太小，量不出差別。

**valid loss 的量法有偏差（重要）**：
- trainer 的 valid loss = 每個 batch 的 MSE 加總後除以 batch 數（sum(loss_record)/len(loss_record)）。valid 的 batch 是 256, 256, 27，所以只有 27 筆的最後一個 batch 跟 256 筆的 batch 權重一樣。valid_loader 又設了 shuffle=True，哪 27 筆落在最後一個 batch，每個 epoch 都不同。
- 用 repo 的 model.ckpt（訓練時印出 1.661 的那個模型）實測：
  - 全部 539 筆一次算的真實 MSE：**2.0685**
  - shuffle=False 的 batch 平均：2.0690
  - shuffle=True 換 20 種順序：1.7670–2.4477，平均 2.0699。前 5 次是 [1.9214, 1.973, 2.1159, 2.0788, 1.8774]。
  - 換 200 種順序：min 1.7069、第 5 百分位 1.8095、中位數 2.0528、max 2.8467。
- 結論：1.661 是 1883 個 epoch 裡最小的那個「雜訊量測值」，挑最小值本身就會偏低。checkpoint 的真實 valid MSE 是 2.07。
- 拿掉 id 的 checkpoint（印出 0.982）：真實 MSE **1.2403**；200 種順序 min 0.9970、中位數 1.2326、max 1.7167。
- id 除以 2699 的 checkpoint（印出 1.003）：真實 MSE **1.2588**；200 種順序 min 1.0154、中位數 1.2465。
- 線性迴歸的 1.166、1.172、1.313、1.303 都是在 539 筆上一次算的真實 MSE，**可以跟 2.07 / 1.24 直接比，但不能跟 1.661 / 0.982 比**。真實數字下的排名：線性迴歸 116 欄 1.166 < 117 欄 1.172 < DNN 拿掉 id 1.240 < DNN id 縮放 1.259 < 抄第 4 天 1.313 < DNN 原版 2.069。
- train loss 也有同樣的問題：train 最後一個 batch 是 112 筆，而且是一邊更新權重一邊記錄的，所以 train loss 也不是某一個固定模型的 MSE。

## ch03 審稿補測（2026-10-03 本機）
- id 除以 2699 的 checkpoint 換 200 種順序：最大值 1.8034。
- 動手做第 1 題逐字輸出：`2160 torch.Size([117]) torch.float32 torch.Size([]) 3.7109291553497314`，接著 `tensor([696.,   0.,   0.])`。動手做第 2 題輸出 `[256, 256, 27]`。
- 動手做第 3 題：`True`、`0`、`one shot 2.0685`、`shuffle False 2.0690`；三行 shuffle True 這次是 1.9393、2.1544、2.0068，每次執行都會不同。
- **修好量法後重新訓練**（train.py 第 123 行改 shuffle=False，第 79 行改 `loss.item() * len(y)`，第 81 行除以 `len(valid_loader.dataset)`）：最佳 `Epoch [2968/3000]: Train loss: 1.5739, Valid loss: 1.7174`，checkpoint 一次算完的 MSE 也是 1.7174，跟印出值相同。**沒有 early stop**，跑滿 3000 個 epoch，最後一行 `Epoch [3000/3000]: Train loss: 1.7492, Valid loss: 2.0180`，共存檔 187 次。
  - 注意：valid 不打亂後，valid_loader 不再消耗全域亂數，train 的打亂順序也會改變，所以 2.069 → 1.717 的進步不能全歸功於量法。
- 實驗都在暫存複本裡跑，repo 的 models/model.ckpt 沒有被動到。

## ch04 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
執行方式：在 `HW01/` 內 `PYTHONPATH=. ../.venv/bin/python <腳本>`。「訓練好的模型」指 repo 的 models/model.ckpt，也就是印出 1.661、真實 MSE 2.069 的那份。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

`print(My_Model(117))` 逐字輸出：
```
My_Model(
  (layers): Sequential(
    (0): Linear(in_features=117, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=8, bias=True)
    (3): ReLU()
    (4): Linear(in_features=8, out_features=1, bias=True)
  )
)
```

參數（`named_parameters()`）：
- layers.0.weight 的形狀是 (16, 117)，**注意是 (輸出, 輸入)**，共 1872 個；layers.0.bias (16,) 16 個。
- layers.2.weight (8, 16) 128 個；layers.2.bias (8,) 8 個。
- layers.4.weight (1, 8) 8 個；layers.4.bias (1,) 1 個。
- 全部都是 float32，requires_grad=True。總數 2033，第一層佔 92.9%。
- 名稱裡沒有 1、3，因為 ReLU 沒有參數（layers[1] 的參數數量是 0）。`len(model.layers)` 是 5。
- state_dict 的 key 依序是 'layers.0.weight', 'layers.0.bias', 'layers.2.weight', 'layers.2.bias', 'layers.4.weight', 'layers.4.bias'。
- 輸入改成 116 欄（拿掉 id）時總數是 2017。
- model.ckpt 的檔案大小是 11005 bytes，其中參數本身佔 2033 × 4 = 8132 bytes，其餘是 key 名稱、形狀等格式資訊。

初始權重（跟 train.py 一樣先 `same_seed(5201314)` 再建立模型，所以就是 train.py 開始訓練時的權重）：
- nn.Linear 預設從均勻分布 U(−1/√in, 1/√in) 抽初始權重和 bias。第一層的範圍是 1/√117 = 0.0925，實測 weight 最小 −0.0924、最大 0.0924，bias −0.0891～0.065。
- 第二層範圍 1/√16 = 0.25，實測 weight −0.2459～0.2499。
- 第三層範圍 1/√8 = 0.3536，8 個 weight 是 [0.1932, -0.168, 0.1907, 0.1301, -0.2984, -0.0179, -0.3317, 0.1277]，bias 0.0609。
- layers.0.weight[0, :5] = [0.0081, 0.0874, 0.0613, -0.0031, 0.0789]。

forward 時各層的形狀（valid 第一個 batch，shuffle=False，256 筆，在 cuda 上）：
- 輸入 (256,117) → layers[0] Linear (256,16) → layers[1] ReLU (256,16) → layers[2] Linear (256,8) → layers[3] ReLU (256,8) → layers[4] Linear (256,1) → squeeze(1) 變成 (256,)。
- `model(x)` 和 `model.forward(x)` 結果完全相同（torch.equal 為 True）。model(x) 會經過 `nn.Module.__call__`，再由它呼叫 forward。
- `.to('cuda')` 之後參數的 device 是 cuda:0。
- 前 5 筆預測是 [15.9939, 5.0906, 9.6055, 3.0774, 8.2978]，真值是 [13.4923, 3.7415, 11.25, 1.875, 8.8235]。

**拿掉 squeeze(1) 會怎樣**（同一個 batch）：
- `MSELoss((256,1), (256,))` 會被廣播成 (256,256)，loss 是 **89.0963**；正確的值是 1.7758。
- 會跳出 UserWarning，逐字是：`Using a target size (torch.Size([256])) that is different to the input size (torch.Size([256, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.`
- 只有 1 筆時，原始輸出是 (1,1)，`squeeze(1)` 得到 (1,)，`squeeze()` 得到 ()（0 維）。所以指定維度 1 比較安全。HW01 每個 batch 最少也有 27 筆，實際上不會碰到這個情況。

訓練好的模型：
- 驗證集的預測範圍 1.8327～28.2623，真值範圍 0.3448～29.8157。
- **ReLU 沒在工作的單元**：在整個訓練集和驗證集上都永遠輸出 0 的，第一層有 **10/16** 個，第二層 1/8 個。第一層所有輸出裡 81.5%（train）、81.7%（valid）是 0。
  - 對照：剛初始化時第一層只有 2/16 個這樣的單元（train，0 的比例 46.6%）。所以大部分是訓練過程中才變成這樣的。
  - 對照：拿掉 id 的模型第一層是 7/16，0 的比例 56.2%；第二層 2/8。
- **id 欄壓過其他欄**：第一層 weight 的平均絕對值，id 欄是 0.22643，其他欄平均 0.05002。乘上驗證集各欄的平均值之後，id 欄的平均 |w·x| 是 **309.13**，其他欄平均只有 0.7558，差了約 400 倍。可以接 ch02 「數值尺度」的結論。

故意寫錯時的錯誤訊息（逐字取第一行）：
- 輸入 float64：`RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float`
- 輸入 116 欄給 117 欄的模型：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x116 and 117x16)`
- 自訂 nn.Module 時沒呼叫 `super().__init__()` 就指定子模組：`AttributeError: cannot assign module before Module.__init__() call`

## ch04 審稿補測（2026-10-03 本機）
- 沒在工作的單元，用 train＋valid 合併的 2,699 筆判定（分開判定的單元數也一樣）：
  - 初始化：第一層 2/16，0 的比例 46.6%；第二層 0/8。
  - model.ckpt：第一層 10/16，0 的比例 81.5%；第二層 1/8。
  - 拿掉 id：第一層 7/16，0 的比例 56.2%（train 和 valid 分開算也都是 56.2%）；第二層 2/8。
- `My_Model(116)` 的 state_dict 載入 `My_Model(117)`，錯誤訊息最後兩行：`RuntimeError: Error(s) in loading state_dict for My_Model:`，接著 `size mismatch for layers.0.weight: copying a param with shape torch.Size([16, 116]) from checkpoint, the shape in current model is torch.Size([16, 117]).`（開頭是 tab）。
- 沒呼叫 super().__init__() 時，traceback 最後三行指向 `torch/nn/modules/module.py` 第 2005 行的 `__setattr__`。
- 動手做第 3 題在真正的終端機裡跑（tty），UserWarning 的兩行（`.../torch/nn/modules/loss.py:626: UserWarning: ...`，下一行是 `  return F.mse_loss(input, target, reduction=self.reduction)`）出現在 `with squeeze 1.7758` 和 `without      89.0963` 之間。輸出接到管線時，stdout 會被緩衝，警告反而出現在最前面。
- 動手做第 4 題：`dead units 10 / 16`、`zero ratio 0.815`。

## ch05 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
範圍：train.py:27–97 的 trainer()。「原版」指 config 不改直接跑 train.py。對照組都在暫存複本裡跑，repo 的 models/model.ckpt 沒被動到。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

### 一次更新拆開看（照 train.py 的順序：same_seed → 切分 → DataLoader → 建模型，然後拿第 1 個 epoch 的第 1 個 batch）
- `print(optimizer)` 逐字輸出：
```
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    fused: None
    lr: 1e-05
    maximize: False
    momentum: 0.9
    nesterov: False
    weight_decay: 0
)
```
- 還沒做過任何 backward 時，`layers.0.weight.grad` 是 None；`optimizer.zero_grad()` 之後也是 None（現在的 PyTorch 預設 set_to_none=True，是把 grad 設成 None，不是填 0）。
- 建立之後 `model.training` 是 True（預設就是訓練模式）。
- `pred.requires_grad` 是 True，`grad_fn` 是 SqueezeBackward1，因為最後一個運算是 squeeze。
- loss 逐字是 `tensor(508.8512, device='cuda:0', grad_fn=<MseLossBackward0>)`。`loss.item()` = 508.8511657714844，跟自己算的 `((pred-y)**2).mean()` 完全相同。**這就是 ch00 進度條上的 loss=509**。第 2 個 batch 的 loss 是 136.3234，也就是 ch00 的 loss=136。
- backward 之後各參數的梯度：

| 參數 | grad 形狀 | 範數 norm | 最大的 \|g\| |
|---|---|---|---|
| layers.0.weight | (16,117) | 15824.6025 | 8688.2920 |
| layers.0.bias | (16,) | 8.7824 | 4.9981 |
| layers.2.weight | (8,16) | 6073.4746 | 2242.7007 |
| layers.2.bias | (8,) | 21.4800 | 12.7494 |
| layers.4.weight | (1,8) | 2681.5046 | 1831.3115 |
| layers.4.bias | (1,) | 38.5490 | 38.5490 |

  - 梯度的形狀一定跟參數的形狀相同。
- **id 欄的梯度**：第一層 weight 梯度的平均 \|g\|，id 欄是 2602.08，其他欄平均 24.13，大約是 **108 倍**。這個 batch 裡 id 欄的平均 \|x\| 是 1348.7，其他欄平均 15.75。這直接印證 ch01 §1.4「梯度和那欄的數值大小成正比」。
- **第一步更新**：Δw 跟 −lr × grad 的差距最大只有 7.45e-09（浮點誤差），所以第一步就是 w ← w − 1e-5 × grad。第一層最大的 \|Δw\| 是 **0.0869，就在 id 欄**，幾乎等於整個初始化範圍 ±0.0925：只走一步，id 欄的權重就移動了一整個初始範圍。
- 第一步之後，optimizer 的 momentum buffer 等於這一步的 grad（torch.allclose 為 True）。
- **第二步驗證了 momentum 公式**：buf ← 0.9 × buf + grad、w ← w − lr × buf，算出來的跟實際結果吻合（allclose 為 True）。
- **忘了 zero_grad**：對同一個 batch 連續 backward 兩次，最後一層 bias 的 grad 從 −57.9991 變成 −115.9982，剛好 2 倍，梯度是累加的。
- **在 no_grad 裡**：pred.requires_grad 是 False，grad_fn 是 None。對它算 loss 再 backward 會報錯：`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`。
- 同一張圖做第二次 backward 會報錯，開頭是：`RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).`
- `model.train()` 和 `model.eval()` 對同一個 batch 的輸出完全相同（torch.equal 為 True），因為 HW01 沒有 Dropout 或 BatchNorm。

### 原版完整訓練（stdout 存檔分析，結果跟 FACTS 開頭的「實測執行」相同）
- 1883 個 epoch，存檔 49 次。存檔的 epoch 依序是：1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 34, 36, 46, 58, 59, 65, 70, 85, 126, 127, 155, 162, 222, 227, 237, 256, 270, 272, 278, 320, 322, 491, 543, 686, 860, 879, 1028, 1030, 1122, 1260, 1483。
  - 前 100 個 epoch 就存了 26 次，之後越來越稀疏。最長的間隔是 223 個 epoch（1260 → 1483）。1483 + 400 = 1883 時觸發 early stop。最後 400 個 epoch 裡最低的 valid loss 是 1.6727，仍然比 1.6611 高。
- 各 epoch 的 train／valid loss：

| epoch | train | valid |
|---|---|---|
| 1 | 134.2442 | 107.2155 |
| 2 | 69.8929 | 50.8182 |
| 3 | 48.5299 | 39.1161 |
| 5 | 34.3680 | 34.5855 |
| 10 | 29.1486 | 29.9683 |
| 20 | 14.4297 | 12.4151 |
| 50 | 9.3503 | 10.0611 |
| 100 | 5.0110 | 4.2813 |
| 200 | 4.2483 | 4.6337 |
| 300 | 3.5812 | 2.6302 |
| 500 | 3.8538 | 3.2954 |
| 1000 | 2.0760 | 3.8595 |
| 1483 | 1.8014 | 1.6611 |
| 1500 | 2.1995 | 1.9105 |
| 1883 | 1.7689 | 2.5030 |

- 1883 個 epoch 中，有 1275 個 epoch 的 train loss 比 valid loss 低。最後 400 個 epoch 的平均：valid 2.3761、train 1.9957。
- checkpoint（第 1483 個 epoch）在整個訓練集上的真實 MSE 是 **1.7938**，印出的 train loss 是 1.8014。對照驗證集的真實 MSE 2.0685（ch03）。
- stdout 共 1940 行。
- **TensorBoard**：每次執行在 `runs/` 下產生一個目錄（例如 `Oct03_16-39-09_ValtecBlackwell`），裡面只有一個 `events.out.tfevents.…` 檔，約 184,720 bytes。scalar tag 只有 `Loss/train` 和 `Loss/valid`，各 1883 個點。橫軸是 `step`（累計的 batch 數），**不是 epoch**：第一個點在 step 9（第 1 個 epoch 跑了 9 個 batch），最後一個點在 step 16947 = 1883 × 9。

### 「最佳 epoch」是被幸運的分組決定的（ch03 第三層偏差的直接證據）
- 重建 train.py 每個 epoch 的 valid 打亂順序，拿第 1483 個 epoch 的 checkpoint 去量，用第 1483 個 epoch 的分組**正好重現 1.6611**，證明重建正確。
  - 重建時要注意：先載入要評估的 checkpoint，再呼叫 same_seed。在 same_seed 之後多建立模型，會消耗全域亂數，打亂順序就對不上了（ch03 §3.4 的現象）。
- 同一個模型配上 1883 種分組：第 1483 個 epoch 那一組排第 **7** 名（最低的是 1.6103，中位數 2.0471）。
- **學習率改成 1e-6 的那次訓練，最佳也是第 1483 個 epoch（印出 1.818），也在第 1883 個 epoch 停止**。它的 checkpoint 在第 1483 個 epoch 的分組上排第 **5** 名（1.8176；中位數 2.2278）。學習率差 10 倍，「最佳」卻落在同一個 epoch：每個 epoch 的分組只由全域亂數決定，跟學習率無關，兩次訓練都在第 1483 個 epoch 碰到同一組特別幸運的分組。

### 學習率與 momentum 的對照組（各改一處，其餘照原版）
| 設定 | 最佳（印出） | 最佳 epoch | 停止 | 存檔次數 | 真實 valid MSE | 第一層沒在工作的單元 |
|---|---|---|---|---|---|---|
| 原版 lr 1e-5、momentum 0.9 | 1.661 | 1483 | 1883（early stop） | 49 | 2.0685 | 10/16 |
| lr 1e-4 | 36.157 | 398 | 798（early stop） | 13 | **44.5095** | **16/16** |
| lr 1e-6 | 1.818 | 1483 | 1883（early stop） | 76 | 2.2601 | 8/16 |
| momentum 0 | 2.463 | 2816 | 3000（跑滿上限） | 59 | 2.9992 | 6/16 |

- **lr 1e-4 整個壞掉**：第 1 個 epoch 的 train loss 是 **2005349.0116**（兩百萬），valid 112.8123；第 2 個 epoch 是 67.2213 / 64.5274，之後停在 40 左右。第一層 16 個單元全部沒在工作，所以模型對每一筆都輸出同一個數字 **9.731**（預測的標準差是 0），約等於訓練集答案的平均 9.7404。真實 MSE 44.5095，跟「永遠猜訓練集平均」的 44.5021、驗證集答案的變異數 44.346 幾乎一樣。這就是 ch04 的 dying ReLU 推到極端：學習率太大，第一步就把所有單元推死了。
- 有一點要注意：lr 1e-4 的 early stop 機制照常運作，印出的 36.157 看起來只是「比較差」，從 log 看不出模型其實已經只會輸出常數。
- momentum 0：同樣的學習率下，進步慢很多，3000 個 epoch 都沒觸發 early stop，最後的真實 MSE 2.9992 比原版差。
- 四組同時在一張 GPU 上跑，各花了 base 38.6 s、lr1e-4 19.3 s、lr1e-6 38.1 s、mom0 56.8 s（單獨跑原版是 34.2 s，見上方「實測執行」）。

## ch05 審稿補測（2026-10-03 本機）
- 動手做第 1 題：`print(SGD)` 共 12 行。不給 momentum 時印 `momentum: 0`；加 `weight_decay=1e-4` 時印 `weight_decay: 0.0001`。
- 動手做第 2 題逐字輸出（用 tty 跑）：`True`、`0`、`True None`、`True SqueezeBackward1`、`tensor(508.8512, device='cuda:0', grad_fn=<MseLossBackward0>)`，接著六行參數 `layers.0.weight (16, 117) 15824.6025` … `layers.4.bias (1,) 38.5490`，然後 `2602.08 24.13`、`0.0869 0`。
- 動手做第 3 題：模型 step 過一次之後，連續 backward 兩次，最後一層 bias 的 grad 是 `15.719593048095703` 和 `31.439186096191406`，剛好 2 倍。最後一行 traceback 是 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`。
- 原版第一次存檔印出 `Saving model with loss 107.215...`（valid 是 107.2155）。

## ch06 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
範圍：predict.py 全檔、utils.py:39–45 save_pred()。「模型」指 repo 的 models/model.ckpt（印出 1.661、真實 valid MSE 2.0685）。
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

### 執行 predict.py（在 HW01/ 內，`../.venv/bin/python predict.py`）
- stdout 逐字（`train_data size` 那行行尾有一個空格，跟 train.py 一樣）：
```
True
0
train_data size: (2160, 118) 
    valid_data size: (539, 118) 
    test_data size: (1078, 117)
```
  - predict.py **不印** `number of features`，這行只有 train.py 有。
- stderr 是 tqdm 進度條，跑完停在 `100%|██████████| 5/5 [00:00<00:00, 37.49it/s]`（速度每次不同），沒有 `Epoch` 描述，因為 predict() 沒呼叫 set_description。
- 含 import 和讀檔，整支約 2.5 秒。其中讀 CSV、切分、選特徵約 0.025 秒。
- 執行兩次，pred.csv 的 md5 完全相同（`46a0484b5d4eff1d72c9dd06923e65bb`）：預測是確定性的。predict.py 沒呼叫 same_seed 也沒關係，因為預測過程不用到亂數，切分則有自己的 Generator（ch02）。

### predict() 內部
- `in no_grad` 時 `pred.requires_grad` 是 False，所以第 17 行的 `.detach()` 是多餘的，但無害（ch05 §5.5 的 train loss 也有同樣的情況）。
- `preds` 是 5 個 tensor 組成的 list，形狀 [(256,), (256,), (256,), (256,), (54,)]，都在 cpu、float32。`torch.cat(preds, dim=0).numpy()` 得到 ndarray (1078,) float32。
- 對 cuda tensor 直接呼叫 `.numpy()` 會報錯：`TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`，所以要先 `.cpu()`。

### pred.csv 格式（save_pred）
- 1079 行（header + 1078），15,799 bytes。header 是 `id,tested_positive`。
- 前三筆：`0,8.850631`、`1,7.368863`、`2,4.1526003`；最後兩筆：`1076,36.028175`、`1077,38.698322`。
- 數字寫成 `str(np.float32)`，也就是最短的 float32 表示法（8.850631、4.1526003），不是 float64 的 8.850630760192871。
- **每一行結尾是 `\r\n`（CRLF），即使是在 Linux 上產生的**。`csv.writer` 預設的 lineterminator 就是 `'\r\n'`，檔案開頭的 bytes 是 `b'id,tested_positive\r\n0,8.850631'`。save_pred 用 `open(file, 'w')` 開檔，沒有加 `newline=''`（Python csv 文件建議要加）。在 Windows 上，文字模式會再把 `\n` 換成 `\r\n`，結果變成 `\r\r\n`，用一般工具看會多出空行。
- `pd.read_csv('pred.csv')` 讀回來的欄位是 ['id','tested_positive']，tested_positive 是 float64，1078 列，數值跟模型在 x_test 上的輸出一致（allclose）。
- 測試集的 id 欄正好是 0..1077 依序排列（array_equal True），所以 save_pred 用 `enumerate` 的 i 當 id，跟測試集的 id 對得上。前提是 test_loader 的 shuffle=False（ch03）。

### 測試集預測的分布：測試集是陽性率更高的時期
- 測試集的預測：最小 3.7642、最大 **44.7035**、平均 15.1763、標準差 7.4867。有 **61 筆超過訓練集最大的答案 30.3046**，沒有負值。
- 原因**不是 id**，而是測試集本身陽性率就比較高：

| 欄位 | 訓練集平均 | 訓練集最大 | 測試集平均 | 測試集最大 | 測試集超過 30.3046 的筆數 |
|---|---|---|---|---|---|
| tested_positive（第 1 天） | 9.5997 | 30.3046 | 14.7545 | 46.4835 | 50 |
| tested_positive.3（第 4 天） | 9.7698 | 30.3046 | **15.1686** | **46.9521** | 67 |

  - 訓練集答案 tested_positive.4 的平均是 9.8193、最大 30.3046。測試集 37 州都有出現，每州 7–38 列。
- 預測超過 30 的那 61 筆，第 4 天陽性率本身就超過 30，例如 id 96 是 32.1、id 835 是 32.08。模型是跟著第 4 天的值往上預測，也就是在訓練資料沒涵蓋的範圍做外推。
- 拿「抄第 4 天」當參照（測試集沒有答案，算不出真實 MSE）：

| 模型 | MSE(預測, 第 4 天)：valid | MSE(預測, 第 4 天)：test | 平均(預測 − 第 4 天)：valid | 平均(預測 − 第 4 天)：test |
|---|---|---|---|---|
| 原版（含 id） | 0.9536 | 1.0045 | +0.1567 | +0.0078 |
| 拿掉 id | 0.1711 | 0.3029 | +0.0689 | +0.1256 |

  - 對照真實 valid MSE：原版 2.0685、拿掉 id 1.2403、抄第 4 天 1.3134。
  - 把原版模型在測試集上的 id 全部改成 1349（約訓練 id 的中位數）再預測：預測平均改變 +0.1305，MSE(預測, 第 4 天) 從 1.0045 變成 0.8888。id 在測試集上的影響存在，但不大。
  - 拿掉 id 的模型在測試集上有 65 筆超過 30.3046，最大 44.8734。
- **如果 test_loader 被打亂**：把 1078 個預測隨機重排（`np.random.default_rng(0)`），MSE(預測, 第 4 天) 從 **1.0045 變成 120.0056**；測試集第 4 天陽性率的變異數是 59.6773。順序錯了，預測就等於配錯答案，大約是變異數的 2 倍。這量化了 ch03 說的「像亂猜」。

### 載入與錯誤
- 這版 torch 的 `torch.load` 簽名中，weights_only 的預設值是 `None`，執行時的效果是 **True**：載入含自訂類別的檔案會報 `UnpicklingError: Weights only load failed. ...`，加 `weights_only=False` 才能載入。`collections.Counter` 這類在白名單裡的型別照樣能載入。model.ckpt（OrderedDict，6 個 key）和 `{'input_dim': 117, 'state_dict': ...}` 這種字典，用預設值都能載入。
- 把 models/model.ckpt 移走再執行：`FileNotFoundError: [Errno 2] No such file or directory: './models/model.ckpt'`
- 在 repo 根目錄執行 `.venv/bin/python HW01/predict.py`：`FileNotFoundError: [Errno 2] No such file or directory: './covid.train.csv'`，在讀 CSV 時就失敗，不會留下 pred.csv。

## ch06 審稿補測（2026-10-03 本機）
- 動手做第 1 題：`wc -l pred.csv` 輸出 `1079 pred.csv`，`md5sum pred.csv` 輸出 `46a0484b5d4eff1d72c9dd06923e65bb  pred.csv`。
- 動手做第 2 題：`repr(csv.excel.lineterminator)` 輸出 `'\r\n'`。
- 動手做第 3 題：輸出 `61 44.7035 67`、`1.0045 59.6773`、`120.0056`。`np.random.default_rng(0).permutation(p)`（直接打亂陣列）跟原本實測用的 `arr[rng.permutation(1078)]`（打亂索引）結果相同。
- 動手做第 4 題：輸出 `OrderedDict 6 117`。
- 自我測驗第 1 題（刪掉 predict.py 第 26 行，並改成 `My_Model(input_dim=train_data.shape[1])`）**已實測**：走不到第 39 行。第 26 行是 valid_data 唯一被定義的地方，所以在 f-string 的 `valid_data size:` 那一行（原檔第 30 行）就報 `NameError: name 'valid_data' is not defined`。
- predict.py 第 23–34 行與 train.py 第 103–114 行逐字相同（diff 沒有差異）。

## ch07 實測（2026-10-03 本機 GPU 環境；雲端不重跑，直接引用這裡）
**這節沒有的輸出不要寫進教材**；需要新數字就標 `TODO(本機實測)` 留給本機補。

### Kaggle 基準線與作業提示（出自 HW01/HW01.pdf）
- 第 22 頁 Hints，逐字：`simple : sample code`、`medium : Feature selection`、`strong : Different model architectures and optimizers`、`boss : L2 regularization and try more parameters`。
- 第 14 頁「Grading -- Kaggle」是一張排行榜截圖，public 分數（MSE，越小越好）：**simple 2.28371、medium 1.49430、strong 1.05728、boss 0.86161**；截圖上助教隊伍 TA_RT 是 0.85800。
- 第 13 頁計分：simple、medium、strong、boss 各分 public 與 private，每項 1 分，共 8 分；code submission 2 分；總分 10 分。
- **注意：這些是測試集（Kaggle）上的 MSE。本書在驗證集上量的 MSE 不能直接拿來比**：測試集是陽性率更高的時期（ch06），本書也沒有上傳 Kaggle。

### 實驗方法
- 用一支腳本照 train.py 的流程重現訓練（same_seed → 切分 → 選特徵 →〔標準化〕→ DataLoader → 建模型 → trainer 的五步驟、存最佳、early stop），可以切換設定。腳本是 `docs/tools/hw01_exp.py`，全部 71 組設定列在 `docs/tools/hw01_ch07_runs.txt`，用 `docs/tools/hw01_run_grid.sh docs/tools/hw01_ch07_runs.txt <輸出.jsonl> 6` 可以整批重跑（單張 GPU 約 10 分鐘）。這些工具都需要 GPU，雲端不要執行。
- **驗證過腳本的正確性**：原版設定逐位元重現印出的 1.6611、最佳第 1483 個 epoch、第 1883 個 epoch 停止、存檔 49 次、真實 MSE 2.0685、第一層 10 個單元沒在工作。修正量法的版本也重現了 ch03 的 1.7174、第 2968 個 epoch、跑到 3000、存檔 187 次。
- **以下所有改良實驗都用修正後的量法**（valid shuffle=False＋依筆數加權，ch03 §3.5 的「現在的做法」），所以印出的最佳值就等於該 checkpoint 的真實 valid MSE。其他照原版：batch 256、n_epochs 3000、early_stop 400、模型 117→16→8→1（除非另外註明）。
- 標準化：只用訓練部分的平均與標準差（std 為 0 的欄改成 1），同一組數字套用到 valid 與 test。
- 「corr」特徵：在訓練部分上，跟答案的 |相關係數| > 0.5 的欄（不含 id）。隨機切分選到 **34 欄**，是 cli、ili、hh_cmnty_cli、nohh_cmnty_cli、work_outside_home、anxious 這 6 個指標各 5 天（30 欄），加上 tested_positive 前 4 天（4 欄）。按時間切分時選到 38 欄，多了 depressed.1～depressed.4。
- 「tp4」= 前 4 天的 tested_positive（第 53、69、85、101 欄）；「survey」= 第 38–116 欄（拿掉 id 和 37 州，79 欄）；「noid」= 拿掉 id 的 116 欄。
- 「按時間切分」照 ch01 的做法：每一州依 id 排序，前 80% 當訓練、後 20% 當驗證，得到 2143／556 筆。
- 「測試集：MSE(預測, 第 4 天)」跟 ch06 一樣，是拿第 4 天當參照的替代指標，不是成績。原版是 1.0045（ch06）。
- 預設 seed 是 5201314（只影響初始權重和 batch 順序；隨機切分的 seed 固定是 5201314）。

### 對應到程式的哪裡（要改哪一行）
- 特徵選擇：utils.py:34 的 `feat_idx = [0, 1, 2, 3, 4]`，搭配 config 的 `'select_all': False`；或直接改 utils.py:32。
- 學習率：config.py:15 `'learning_rate'`。優化器、momentum、weight_decay：train.py:34。
- 架構和激活函數：model.py:14–20 的 nn.Sequential。
- 標準化：原程式**沒有**對應的位置，要在 train.py 的 select_feat 之後自己加。predict.py 也必須套用**訓練時算出的同一組**平均與標準差，所以要像 ch06 §6.3 那樣把它們存進 checkpoint。
- 修正量法：train.py:123（shuffle=False）、79、81（見 ch03 動手做第 4 題）。

### 第一組：單一變因（隨機切分，seed 5201314）
| 設定 | 特徵數 | 參數量 | 真實 valid MSE | 真實 train MSE | 最佳 epoch | 停止 | 第一層沒在工作的單元 | 測試集：MSE(預測, 第 4 天) | 測試集預測最大值 |
|---|---|---|---|---|---|---|---|---|---|
| 原版 117 欄（量法修正後） | 117 | 2033 | 1.7174 | 1.4799 | 2968 | 3000 | 12 | 0.7175 | 45.2725 |
| 拿掉 id（116 欄） | 116 | 2017 | 1.1848 | 1.0592 | 2977 | 3000 | 7 | 0.2518 | 44.8421 |
| survey 79 欄 | 79 | 1425 | 1.2159 | 1.0743 | 2982 | 3000 | 6 | 0.2249 | 45.1104 |
| corr 34 欄 | 34 | 705 | 1.1886 | 1.1012 | 2995 | 3000 | 4 | 0.2393 | 45.1379 |
| tp4 4 欄 | 4 | 225 | 1.3561 | 1.1785 | 2996 | 3000 | 1 | 0.0253 | 46.2615 |
| 116 欄＋標準化，SGD lr 1e-5 | 116 | 2017 | 1.2915 | 0.9656 | 2994 | 3000 | 0 | 2.6444 | 46.5314 |
| 116 欄＋標準化，SGD lr 1e-4 | 116 | 2017 | 1.0722 | 0.5874 | 2446 | 2846 | 0 | 5.0139 | 53.2061 |
| 116 欄＋標準化，SGD lr 1e-3 | 116 | 2017 | 1.0715 | 0.5636 | 543 | 943 | 0 | 4.1104 | 55.4179 |
| 116 欄＋標準化，SGD lr 1e-2 | 116 | 2017 | 6.669 | 5.1295 | 516 | 916 | 0 | 34.0022 | 25.981 |
| 116 欄、不標準化，Adam lr 1e-3 | 116 | 2017 | 1.145 | 0.8982 | 2408 | 2808 | 7 | 0.7348 | 42.4037 |
| 116 欄＋標準化，Adam lr 1e-3 | 116 | 2017 | 1.1704 | 0.6956 | 632 | 1032 | 0 | 9.2868 | 54.4809 |
| 116 欄＋標準化，Adam lr 1e-4 | 116 | 2017 | 1.1944 | 0.7847 | 2957 | 3000 | 0 | 8.9152 | 50.3724 |
| 116 欄＋標準化，AdamW lr 1e-3（wd 0） | 116 | 2017 | 1.1704 | 0.6956 | 632 | 1032 | 0 | 9.2868 | 54.4809 |
| tp4＋標準化，Adam 1e-3 | 4 | 225 | 1.3028 | 1.1635 | 977 | 1377 | 1 | 0.0152 | 46.31 |
| corr＋標準化，Adam 1e-3 | 34 | 705 | 1.1063 | 0.921 | 1578 | 1978 | 0 | 1.8546 | 49.3553 |
| survey＋標準化，Adam 1e-3 | 79 | 1425 | 1.223 | 0.846 | 800 | 1200 | 0 | 1.0355 | 45.938 |
- 光是拿掉 id（不標準化、照原版 SGD 1e-5）就從 1.7174 降到 1.1848。
- 標準化之後，lr 1e-5 太小（1.2915，跑滿 3000 個 epoch）；1e-3、1e-4 最好（1.0715、1.0722）；1e-2 就壞了（6.669）。ch05 說「lr 1e-5 是被 id 欄逼出來的」，這裡得到證實。
- **wd=0 時，Adam 和 AdamW 的結果完全相同**（兩組數字一模一樣）。AdamW 跟 Adam 只差在 weight decay 的做法。
- 標準化之後，第一層沒在工作的單元變成 0（原版 12、拿掉 id 7）。

### 第二組：L2（weight decay）、架構、激活函數（隨機切分，seed 5201314，116 欄＋標準化）
| 設定 | 特徵數 | 參數量 | 真實 valid MSE | 真實 train MSE | 最佳 epoch | 停止 | 第一層沒在工作的單元 | 測試集：MSE(預測, 第 4 天) | 測試集預測最大值 |
|---|---|---|---|---|---|---|---|---|---|
| SGD lr 1e-3（對照） | 116 | 2017 | 1.0715 | 0.5636 | 543 | 943 | 0 | 4.1104 | 55.4179 |
| SGD lr 1e-3，wd 1e-3 | 116 | 2017 | 1.0584 | 0.5704 | 543 | 943 | 0 | 4.4729 | 55.4244 |
| SGD lr 1e-3，wd 1e-2 | 116 | 2017 | 1.041 | 0.5461 | 764 | 1164 | 0 | 4.7655 | 54.7951 |
| Adam lr 1e-3，wd 1e-2 | 116 | 2017 | 1.1181 | 0.6658 | 805 | 1205 | 0 | 7.519 | 54.9073 |
| AdamW lr 1e-3，wd 1e-2 | 116 | 2017 | 1.1909 | 0.6873 | 620 | 1020 | 0 | 9.144 | 53.9716 |
| AdamW lr 1e-3，wd 1e-1 | 116 | 2017 | 1.1754 | 0.6656 | 846 | 1246 | 0 | 9.7567 | 53.8229 |
| corr＋標準化，Adam 1e-3，wd 1e-2 | 34 | 705 | 1.0943 | 0.9864 | 797 | 1197 | 0 | 0.9537 | 47.2874 |
| corr＋標準化，AdamW 1e-3，wd 1e-1 | 34 | 705 | 1.105 | 0.9583 | 2879 | 3000 | 0 | 3.6159 | 54.0425 |
| 架構 116→64→32→1，SGD 1e-3 | 116 | 9601 | 0.9909 | 0.3769 | 390 | 790 | 0 | 5.7501 | 55.684 |
| 架構 116→8→1，SGD 1e-3 | 116 | 945 | 1.1349 | 0.7092 | 792 | 1192 | 0 | 3.2123 | 48.2503 |
| LeakyReLU，SGD 1e-3 | 116 | 2017 | 1.0834 | 0.626 | 383 | 783 | 0 | 3.0105 | 51.3183 |
| GELU，SGD 1e-3 | 116 | 2017 | 1.0346 | 0.4111 | 810 | 1210 | 0 | 9.6035 | 53.0476 |
| 原版 117 欄不標準化，LeakyReLU | 117 | 2033 | 1.7336 | 1.4529 | 2954 | 3000 | 0 | 0.7795 | 44.8432 |
| 原版 117 欄不標準化，GELU | 117 | 2033 | 1.7304 | 1.4947 | 2968 | 3000 | 6 | 0.5792 | 44.9632 |
- L2 讓 SGD 從 1.0715 進步到 1.041（wd 1e-2）。Adam 的 wd 1e-2 是 1.1181，AdamW 的 wd 1e-2 是 1.1909：同樣的 wd 數值，在 Adam 和 AdamW 裡的意義不同。
- 隨機切分下最好的是 64-32 架構（0.9909，9,601 個參數），其次是 GELU（1.0346）。
- 在原版（不標準化）設定下換激活函數：LeakyReLU 讓第一層沒在工作的單元從 12 變成 0，但 MSE 沒變好（1.7336 對 1.7174）；GELU 有 6 個。GELU 只有在輸入剛好是 0、或輸入非常負而在 float32 下溢成 0 時，才會輸出剛好 0，所以對 GELU 來說，「永遠為 0」代表那個單元的輸入一直極度偏負，判定的意義跟 ReLU 不完全相同，引用時要註明。結論：激活函數治不了 id 的尺度問題，跟 ch04 的推論一致。

### 第三組：線性模型（同一套訓練流程，沒有隱藏層：Linear(n, 1)，Adam 1e-3，標準化）
| 設定 | 特徵數 | 參數量 | 真實 valid MSE | 真實 train MSE | 最佳 epoch | 停止 | 第一層沒在工作的單元 | 測試集：MSE(預測, 第 4 天) | 測試集預測最大值 |
|---|---|---|---|---|---|---|---|---|---|
| 線性，tp4（隨機切分） | 4 | 5 | 1.3169 | 1.1689 | 2985 | 3000 | — | 0.0157 | 46.3632 |
| 線性，corr（隨機切分） | 34 | 35 | 1.1422 | 1.0638 | 2991 | 3000 | — | 0.4154 | 46.6117 |
| 線性，116 欄（隨機切分） | 116 | 117 | 1.1464 | 0.9775 | 2900 | 3000 | — | 0.6292 | 45.6827 |
| 線性，tp4（時間切分） | 4 | 5 | 1.2348 | 1.1871 | 2998 | 3000 | — | 0.0189 | 46.3957 |
| 線性，corr（時間切分） | 38 | 39 | 1.1333 | 1.0563 | 2993 | 3000 | — | 0.5534 | 47.1858 |
| 線性，116 欄（時間切分） | 116 | 117 | 1.197 | 0.9706 | 2818 | 3000 | — | 0.8263 | 46.2512 |
- 對照目錄頁用最小平方法一次解出的線性迴歸：隨機切分 116 欄是 1.166、4 欄是 1.303；時間切分 116 欄是 1.21（ch01）。用梯度下降訓練的線性模型（1.1464、1.3169、1.197）跟它們在同一個量級，差異來自 early stop 和標準化。

### 第四組：按時間切分（驗證集是每州最後 20% 的日子，比較接近「預測未來」）
| 設定 | 特徵數 | 參數量 | 真實 valid MSE | 真實 train MSE | 最佳 epoch | 停止 | 第一層沒在工作的單元 | 測試集：MSE(預測, 第 4 天) | 測試集預測最大值 |
|---|---|---|---|---|---|---|---|---|---|
| 原版 117 欄 | 117 | 2033 | 1.3025 | 1.3405 | 2960 | 3000 | 9 | 0.3477 | 44.7125 |
| 拿掉 id 116 欄 | 116 | 2017 | 1.1784 | 1.0652 | 2950 | 3000 | 7 | 0.3415 | 45.0473 |
| corr 38 欄 | 38 | 769 | 1.165 | 1.0888 | 2996 | 3000 | 5 | 0.2498 | 46.1877 |
| tp4 | 4 | 225 | 1.2641 | 1.1958 | 2996 | 3000 | 1 | 0.023 | 46.4334 |
| 116 欄＋標準化，SGD 1e-3 | 116 | 2017 | 1.655 | 0.7607 | 134 | 534 | 0 | 2.6635 | 47.6076 |
| 116 欄＋標準化，Adam 1e-3 | 116 | 2017 | 2.2158 | 0.6257 | 574 | 974 | 0 | 9.8059 | 52.936 |
| 116 欄＋標準化，SGD 1e-3，wd 1e-2 | 116 | 2017 | 1.6203 | 0.7391 | 158 | 558 | 0 | 3.3913 | 49.8619 |
| 116 欄＋標準化，64-32 | 116 | 9601 | 1.7157 | 0.7048 | 81 | 481 | 0 | 3.2783 | 51.2621 |
| 116 欄＋標準化，GELU | 116 | 2017 | 1.8337 | 0.7145 | 178 | 578 | 0 | 2.454 | 45.8497 |
| corr＋標準化，Adam 1e-3 | 38 | 769 | 1.1334 | 0.9569 | 822 | 1222 | 0 | 2.166 | 50.7843 |
| corr＋標準化，Adam 1e-3，wd 1e-2 | 38 | 769 | 1.1219 | 0.9528 | 1426 | 1826 | 0 | 2.5776 | 51.3118 |
| corr＋標準化，AdamW 1e-3，wd 1e-1 | 38 | 769 | 1.1512 | 0.9912 | 442 | 842 | 0 | 2.2787 | 50.8734 |
| tp4＋標準化，Adam 1e-3 | 4 | 225 | 1.2313 | 1.1825 | 746 | 1146 | 1 | 0.0244 | 46.6106 |
- **在隨機切分下最好的那幾組，換成時間切分就大幅變差**：64-32 從 0.9909 變成 1.7157，GELU 從 1.0346 變成 1.8337，SGD＋wd 1e-2 從 1.041 變成 1.6203，116 欄＋標準化＋Adam 從 1.1704 變成 2.2158。它們的 train MSE 很低（0.38～0.74），把同一段時期的資料背了下來，換到沒見過的日子就失靈。
- 在時間切分下最好的是特徵少、有選過的組合：corr＋標準化＋Adam＋wd 1e-2 是 1.1219，線性 corr 是 1.1333，tp4 系列是 1.23～1.26。
- 時間切分的驗證集跟隨機切分的不是同一批資料，兩邊的數字不能直接比大小，只能比**排名怎麼變**。

### 第五組：換 seed 的變異（每組 4 個 seed：5201314、1、2、3）
| 設定 | 切分 | 4 個 seed 的真實 valid MSE | 平均 | 標準差 |
|---|---|---|---|---|
| 原版 117 欄（量法修正後） | 隨機 | seed 1: 1.7243, seed 2: 12.7959, seed 3: 1.7467, seed 5201314: 1.7174 | 4.4961 | 5.5332 |
| 拿掉 id | 隨機 | seed 1: 1.1999, seed 2: 1.1955, seed 3: 1.1714, seed 5201314: 1.1848 | 1.1879 | 0.0127 |
| corr＋標準化＋Adam＋wd 1e-2 | 隨機 | seed 1: 1.1155, seed 2: 1.0882, seed 3: 1.0952, seed 5201314: 1.0943 | 1.0983 | 0.0119 |
| 116 欄＋標準化，64-32 | 隨機 | seed 1: 1.0446, seed 2: 0.9789, seed 3: 0.9077, seed 5201314: 0.9909 | 0.9805 | 0.0563 |
| corr＋標準化＋Adam＋wd 1e-2 | 時間 | seed 1: 1.1525, seed 2: 1.1605, seed 3: 1.137, seed 5201314: 1.1219 | 1.1430 | 0.0171 |
| 116 欄＋標準化，64-32 | 時間 | seed 1: 1.7282, seed 2: 1.7258, seed 3: 1.7648, seed 5201314: 1.7157 | 1.7336 | 0.0215 |
| 線性 corr | 時間 | seed 1: 1.1337, seed 2: 1.1333, seed 3: 1.1332, seed 5201314: 1.1333 | 1.1334 | 0.0002 |
- **原版設定換 seed 會整個壞掉**：seed 2 的真實 MSE 是 12.7959。它沒有像 ch05 lr 1e-4 那樣全死（第一層 16 個單元死了 6 個），而是一開始就卡在幾乎只輸出常數的狀態：前 3 個 epoch 各 batch 的 loss 一直在 120～160 之間（第 1 個 batch 是 354.8），第 3 個 epoch 結束時訓練集預測的標準差只有 0.0058、平均 −0.27。之後非常緩慢地往下爬，跑滿 3000 個 epoch 都還在創新低（存檔 1552 次），最後停在 12.7959。測試集預測的最大值只有 14.47。
  - 這個 seed 2 的原版，第一個 epoch 印出 train 164.9592、valid 153.3218（量法修正後）。
  - 注意：第五組全部用修正後的量法（valid 不打亂）。直接跑原版 train.py 時，valid 打亂會多消耗全域亂數，所以同一個 seed 的訓練軌跡會不同，**不能說「原版 train.py 用 seed 2 一定會壞掉」**，只能說「這套設定對 seed 很敏感」。
- 拿掉 id 之後，4 個 seed 都穩定在 1.17～1.20（標準差 0.0127）。
- **時間切分下，corr DNN 的平均是 1.1430（標準差 0.0171），線性 corr 是 1.1334（標準差 0.0002）**：平均下來線性模型反而略好，而且幾乎不受 seed 影響。單看 seed 5201314 時 DNN 的 1.1219 略勝，那只是運氣。
- 隨機切分下 64-32 架構平均 0.9805（0.9077～1.0446），是隨機切分的第一名；同一個架構在時間切分下平均 1.7336。

### 結論（給 ch07 的主線）
- 作業提示四步（特徵選擇 → 架構與優化器 → L2 → 多試參數）在**隨機切分**的驗證集上全部有效：1.7174 → 1.1848（拿掉 id）→ 1.0715（標準化＋lr 1e-3）→ 1.041（＋wd）→ 0.9805（64-32，4 個 seed 平均），**DNN 確實大幅進步，也贏過線性迴歸的 1.166**。這兌現了目錄頁「DNN 可以大幅進步」的說法。
- 但**按時間切分**之後，高容量的贏家全部退步，最穩的是「少而精的特徵＋簡單模型」，DNN 跟線性模型打平。測試集正是「未來的日子」（ch06：陽性率更高的時期），所以時間切分比較能預告 Kaggle 上的表現。這不是本書實測得出的結論，因為本書沒有上傳 Kaggle。
- 測試集上的替代指標也支持這一點：隨機切分的贏家在測試集上離第 4 天很遠（64-32 是 5.75、116 欄＋標準化＋Adam 是 9.29），tp4 和 corr 的線性模型只有 0.02～0.55。


## ch07 審稿補測（2026-10-03 本機）
**題庫判準已用真的 train.py 驗證**：在 HW01 的複本裡照題目字面改 utils.py／train.py／config.py／model.py 後執行 train.py，印出的最佳與最佳 epoch 都跟 hw01_exp.py 逐位相同。驗證過的組合：
- 入門 1（noid）：`Epoch [2977/3000]: Train loss: 1.0820, Valid loss: 1.1848`、`Saving model with loss 1.185...`；最後一行 `Epoch [3000/3000]: Train loss: 1.0893, Valid loss: 1.1888`；存檔 336 次。
- 入門 2（tp4）：第 2996 個 epoch 1.3561，存檔 950 次。
- 進階 1（noid＋標準化＋lr 1e-3）：第 543 個 epoch 1.0715（`Train loss: 0.5903`），第 943 個 epoch early stop，第一層沒在工作的單元 0/16。
- 進階 3（時間切分 noid）：`train_data size: (2143, 118)`；第 2950 個 epoch 1.1784。
- 進階 4（線性＋corr〔np.corrcoef 寫法，選到 34 欄〕＋標準化＋Adam 1e-3）：第 2991 個 epoch 1.1422，存檔 1400 次。
- 挑戰 2 第一部分（量法修正、117 欄、config seed 2、第 106 行切分固定 5201314）：12.7959，第一個 epoch `Train loss: 164.9592, Valid loss: 153.3218`。

新的實測：
- **入門 4**（117 欄、不標準化、SGD lr 1e-3、量法修正）：第 1 個 epoch `Train loss: 673668052737.8724, Valid loss: 411272072.7273`；最佳是第 52 個 epoch 的 `Valid loss: 44.3461`；第 452 個 epoch early stop，存檔 47 次；真實 valid MSE 44.3461、train 40.7400；第一層 16/16 沒在工作；驗證集預測的標準差 0，平均 10.1485。
- **Dropout**（noid＋標準化＋SGD lr 1e-3，116→64→ReLU→Dropout(0.2)→32→ReLU→Dropout(0.2)→1，量法修正，seed 5201314，直接改 model.py 實跑）：
  - 隨機切分：最佳第 845 個 epoch 1.0094（印出的 train loss 1.6428，因為 Dropout 開著），第 1245 個 epoch early stop；真實 train MSE（eval 模式）0.5353；第一層 0/64 沒在工作。
  - 時間切分：第 645 個 epoch 1.7219，真實 train 0.5415。
  - 隨機切分但刪掉 train.py 第 47 行 `model.train()`：第 494 個 epoch 0.9969，真實 train 0.3130（印出的 train loss 0.3332）。第 1 個 epoch 之後 Dropout 就不再起作用。
  - 對照沒有 Dropout 的 64-32：隨機 0.9909（train 0.3769）、時間 1.7157（train 0.7048）。
- **進階 2**（7.12 節的 checkpoint 字典＋改寫 predict.py）：checkpoint 用 `weights_only=True` 載入正常，key 依序是 ['feat_idx', 'mean', 'std', 'state_dict']，mean 是長度 116 的 list。改寫後的 predict.py 不讀 covid.train.csv，pred.csv 是 15,762 bytes，前兩筆 `0,7.8344135`、`1,8.61439`。§6.10 第 3 項指令輸出 `85 55.4179 67`、`4.1104 59.6773`、`140.3394`。
- **NumPy 陣列存進 checkpoint**：`torch.save({'mean': np.zeros(3)}, ...)` 再用 `weights_only=True` 載入，會報 `UnpicklingError: Weights only load failed. ...`。所以 7.12 節要先 `.tolist()` 的說法正確。
- **挑戰 2 第二部分**（原版 train.py、沒有量法修正、config seed 2、切分固定 5201314）：第一個 epoch `Train loss: 164.9592, Valid loss: 152.6497`；印出的最佳是第 2954 個 epoch 的 `Valid loss: 10.3637`；跑滿 3000 個 epoch，存檔 83 次；checkpoint 的真實 valid MSE **13.2407**、train 11.3880；第一層 6/16 沒在工作；驗證集預測的標準差 4.2206。量法修正版結束時標準差是 4.1824。
- **挑戰 3 permutation importance**（時間切分、corr 38 欄＋標準化＋Adam 1e-3＋wd 1e-2、seed 5201314，真實 valid 1.1219；`np.random.default_rng(0)` 依欄位順序逐欄打亂標準化後的驗證集）：前 8 名的 MSE 增加量依序是 tested_positive.3 81.314、cli.4 2.4311、cli.3 2.1536、hh_cmnty_cli.4 1.9466、hh_cmnty_cli.3 1.2411、cli.2 0.7924、cli.1 0.7137、tested_positive 0.7045；最低 3 名是 ili.1 0.0227、anxious.4 0.0272、depressed.4 0.0301。重現方式：`hw01_exp.py ... --perm 1`。
- **挑戰 4 LightGBM 4.7.0**（noid 116 欄、不標準化；重現方式：`docs/tools/hw01_lgbm.py`。LightGBM 裝在專案 venv 之外，requirements.txt 沒有改）：

| 設定 | 真實 valid MSE | 真實 train MSE | 樹的數量 | 測試集：MSE(預測, 第 4 天) | 測試集預測最大值 |
|---|---|---|---|---|---|
| 隨機切分，預設（100 棵） | 1.2808 | 0.1684 | 100 | 4.8417 | 28.0957 |
| 隨機切分，early stop 100 | 1.2802 | 0.2481 | 75 | 4.8705 | 27.9937 |
| 時間切分，預設 | 1.3759 | 0.1613 | 100 | 5.1995 | 28.1781 |
| 時間切分，early stop 100 | 1.3602 | 0.5149 | 37 | 5.5735 | 27.1906 |

  - 樹的預測不會超出訓練時見過的答案範圍（訓練集最大 30.3046），測試集第 4 天最大 46.95，所以樹在測試集上無法外推。
