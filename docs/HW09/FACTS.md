# HW09 教材事實清單（維護筆記，不進教材）

> **這是什麼**：docs/HW09/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 裡不會連到它。
>
> **寫作分工**：本機（有 GPU）已在 2026-10-03 把全書需要的數字一次量完，寫在這裡；雲端 session 沒有 GPU、沒有環境、不跑程式，只引用這裡的數字寫章。這裡找不到的，標 `<!-- TODO(本機實測): 要量什麼 -->`，PR 回來後由本機補。
>
> **重現方法**（需要 GPU、checkpoint.pth、food/ 與 HF 模型）：在 `HW09/` 裡跑 `../.venv/bin/python ../docs/tools/hw09_facts.py [env model predict lime saliency smooth filter ig bert embed]`。下面「CNN 實測」「BERT 實測」兩節就是它的輸出整理。

教材對應 commit：`b2dbe4c`（HW09 程式碼在 `e951b97` 加入，之後沒改過）。
原始碼根目錄：`HW09/`。官方原版：`~/poyi/GitHubPublic/ML2022-Spring/HW09/HW09.ipynb`（Colab notebook，不在 repo 裡，下面「原版 vs 本 repo」列了差異）。作業投影片：`HW09/HW09.pdf`（20 頁，已在 repo）。

## 全書約定
- 樣式：用 mySkills **新版共用資產**（`completed-repo-to-html-textbook/assets/style.css`、`enhance.js`；有 Python 上色、`pre.shell.cmd`、hero/pit/hl-*/tldr 元件）。使用者 2026-10-03 選的，**不要**複製 docs/HW01 的舊版。
- 指令塊：讀者要貼上執行的用 `<pre class="shell cmd">`，輸出用 `<pre class="shell">`。Python 用 `<pre class="py">`。
- listing 的 `data-hot` 寫**原始碼行號**（與 figcaption `檔名:起–迄` 同一套）。
- 檢查：`python3 docs/tools/verify_book.py . docs/HW09/chNN.html`（2026-10-03 已改成依 HTML 所在資料夾找原始碼，`docs/HW09/x.html` 引用 `HW09/<file>`）。
- 實際輸出圖在 `docs/HW09/img/`（44 張 PNG，從 `HW09/output/` 複製，見「圖檔清單」）。機制圖照 skill 規定手繪 inline SVG；**結果圖**（LIME 疊色、熱圖等）是這份作業的主角，直接用 `<img src="img/xxx.png">` 引用真的輸出，不要手畫假的。
- 每本書的規則（使用者指定）：ch00 要有「模型總覽」（架構 SVG、每層 tensor 形狀、參數量：手算 + PyTorch 印出、哪個檔案定義模型），放在任務說明之後；全書開頭有「2022 vs 2026」段落；2022 寫法過時的地方加「現在的做法」框。參考 docs/HW01 的 ch00 §0.2、圖 0.1。

## 環境（實測 2026-10-03）
- Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.3、matplotlib 3.11.2、lime 0.2.0.1（`lime.__version__` 不存在，要用 `importlib.metadata.version('lime')`）、scikit-image 0.26.0、scikit-learn 1.9.1、transformers 5.18.0。
- GPU：NVIDIA RTX PRO 4000 Blackwell（sm_120，24 GB），WSL2。
- 共用 venv 在 repo 根目錄 `.venv`；在 `HW09/` 裡用 `../.venv/bin/python <script>.py`。`explain_cnn.py` 寫死 `.cuda()`，沒有 CPU fallback（兩支 BERT 腳本沒有 `.cuda()`，在 CPU 上算；2026-10-03 ch00 審稿更正），**這是刻意的設計**，教材以中性描述，不列為問題或練習。
- 原版 Colab 指定 `lime==0.1.1.37`（投影片 p.7）、`transformers==4.5.0`；本 repo 改用上面的新版，requirements.txt 有 pin lime/scikit-image/scikit-learn。

## 資料與檔案
- 來源：本機 Kaggle zip `ml2022spring-hw9.zip`（Windows Documents/poyi/ml_2022_data），內含 `checkpoint.pth`（170,002,879 bytes）與 `food.zip`（621,495 bytes）。解到 `HW09/checkpoint.pth`、`HW09/food/`。
- `HW09/.gitignore`：`food/`、`checkpoint.pth`、`output/`、`hw9_bert/`，所以 repo 裡**看不到**資料與輸出；全書的資料描述都只能依這份 FACTS。
- `food/` 只有 10 張圖，檔名 `<類別>_<編號>.jpg`：
  | 編號 | 檔名 | 原始尺寸 (W×H) | 類別 |
  |---|---|---|---|
  | 0 | 0_0.jpg | 512×512 | 0 Bread |
  | 1 | 1_1.jpg | 849×565 | 1 Dairy product |
  | 2 | 1_2.jpg | 1294×1300 | 1 Dairy product |
  | 3 | 2_3.jpg | 512×341 | 2 Dessert |
  | 4 | 2_4.jpg | 512×512 | 2 Dessert |
  | 5 | 3_5.jpg | 512×512 | 3 Egg |
  | 6 | 5_6.jpg | 512×512 | 5 Meat |
  | 7 | 6_7.jpg | 512×512 | 6 Noodles/Pasta |
  | 8 | 8_8.jpg | 512×512 | 8 Seafood |
  | 9 | 9_9.jpg | 512×384 | 9 Soup |
  全部 RGB。類別 4 Fried food、7 Rice、10 Vegetable/Fruit 沒有圖。11 類名稱順序（notebook 與投影片 p.4）：Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit。
- 從圖片內容看（img/images.png）：0 披薩切片、1 奶油塊、2 牛奶壺加草莓、3 巧克力蛋糕、4 鬆餅、5 荷包蛋吐司、6 牛排、7 炒麵加荷包蛋、8 生魚片、9 味噌湯。
- `get_paths_labels` 用 `類別*1000000 + 編號` 排序，結果就是上表順序；`labels` 印出 `[0, 1, 1, 2, 2, 3, 5, 6, 8, 9]`。
- 評估轉換：Resize (128,128)（不保持長寬比，非正方形的圖會被壓扁）→ ToTensor；tensor (10, 3, 128, 128) float32，值域 0.0–1.0。`mode='train'` 的隨機翻轉／旋轉在本作業從沒用到（只建 `mode='eval'`）。
- checkpoint 是 dict，keys：`epoch`（= 208）、`model_state_dict`（81 個 entry）、`optimizer_state_dict`。170 MB ≈ 權重 56.7 MB（14.16M × 4 bytes）× 3（權重 + Adam 的兩份動量），所以 optimizer 狀態佔了約 2/3。
- 投影片 p.4 說模型是 HW3 食物分類（同一個 food-11 資料集）。

## 程式結構（HW09/，e951b97）
- `model.py`：`Classifier`（model.py:8–45）。
- `dataset.py`：`FoodDataset`（dataset.py:10–44，含 `getbatch`）、`get_paths_labels`（dataset.py:49–59）。
- `explain_cnn.py`：Part 1，Q1–20。`normalize` :24、`save_fig` :28、`lime_explain` :39、`compute_saliency_maps` :79、`saliency` :99、`smooth_grad` :120、`smoothgrad` :144、`layer_activations` 全域變數 :164、`filter_explanation` :165、`filter_explain` :208、`IntegratedGradients` :225、`integrated_gradients` :268、main :284。
- `bert_hidden_states.py`：Q25–27。`qa_model_name` :36、`visualize` :75、main :125。
- `bert_embedding.py`：Q28–30。`FONT_PATH`、`select_word_index` :59、`euclidean_distance` :63、`cosine_similarity` :67、`get_select_embedding` :74、main :85。
- 行號會因為編輯而變；寫 listing 前用原始碼再核一次，verify_book.py 會逐行比對。

## 原版 notebook vs 本 repo（「2022 vs 2026」與「現在的做法」的素材）
- Colab 專屬的東西拿掉：`drive.mount`、`os.chdir` 到 Drive、`!gdown` 下載資料與模型、`!pip install`、`!unzip`。
- 一個 notebook 拆成 5 個檔：model.py、dataset.py、explain_cnn.py、bert_hidden_states.py、bert_embedding.py。
- 圖從 `plt.show()` 改成 `fig.savefig('output/xxx.png')`，並 `matplotlib.use('Agg')`（沒有螢幕也能畫）。
- `from torch.autograd import Variable` 與 `Variable(...)` 拿掉：PyTorch 0.4（2018）起 tensor 本身就能追蹤梯度，`Variable` 只是相容殼。smooth_grad 的雜訊改成 `x.new_empty(x.size()).normal_(mean, sigma**2)`。
- `x.grad.data`、`.data.numpy()` 的 `.data` 拿掉（現在用 `.detach()`）。
- IG 的 `torch.FloatTensor(1, n).zero_().cuda()` 改成 `torch.zeros(1, n).cuda()`。
- `pdb.set_trace`、`argparse.Namespace` 這些沒用到的 import 拿掉；路徑改成模組層常數（`ckptpath`、`dataset_dir`、`output_dir`）。
- `print(model.load_state_dict(...))`：原版在 notebook 裡靠最後一行自動顯示 `<All keys matched successfully>`，腳本要明確 print。
- Part 2a：助教的 `hw9_bert.zip`（Google Drive id `1h3akaNdouiIGItOqEs6kUZE-hAF0QeDk`，含 `hw9_bert/Tokenizer` 與 `hw9_bert/output/model_q{1,2,3}` 預存 hidden states）**已經下架**：2026-10-03 用兩個下載網址都回 HTTP 404。本 repo 的 `bert_hidden_states.py`：有 `hw9_bert/` 就照原版用；沒有就用 HF 上公開的 `deepset/bert-base-cased-squad2`（BERT-base cased，SQuAD 2.0 微調）自己算 hidden states。**所以圖與投影片／原作業不同**；助教用的是哪個模型無從得知。
- Part 2a 原版一次只畫一題（改 `QUESTION = 1`），本 repo 三題都畫。
- Part 2b：原版 `euclidean_distance`、`cosine_similarity` 是留給學生寫的 `return 0`（不寫的話整張矩陣都是 0）；本 repo 已實作（`np.linalg.norm(a - b)`、`np.dot(a,b)/(‖a‖‖b‖)`）。
- Part 2b 中文字型：原版 `!gdown` 下載「台北思源黑體」`taipei_sans_tc_beta.ttf` 再用 `FontProperties`；本 repo 用系統的 `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf`，並把 `font.family` 設成 `['DejaVu Sans', 'Droid Sans Fallback']`。原因：Droid Sans Fallback 缺英文字母，單用它會出現 `UserWarning: Glyph 73 (I) missing from font(s) Droid Sans Fallback`（"Face ID" 的 I、D），matplotlib 3.6 起支援逐字 fallback。
- Q21–24 exBERT：原版用 `IPython.display.IFrame` 嵌 https://exbert.net/exBERT.html；腳本沒有對應程式（只在 docstring 提一句）。2026-10-03 實測 exbert.net DNS 有解析（52.116.48.130）但連線逾時（curl 回 000）；投影片 p.14 的替代網址 https://huggingface.co/exbert/ 會 302 轉到 https://huggingface.co/spaces/exbert-project/exbert（HTTP 200；Space 本身是否在跑沒有檢查）。
- 套件版本見「環境」。

## 執行實測（2026-10-03，在 HW09/ 內）
- `../.venv/bin/python explain_cnn.py`：real 50.9 s（第一次在新環境跑是 1 分 52 秒，含 import 與 cuDNN 暖機；user time 17 分 31 秒，LIME/skimage 用了多核）。stdout 逐字：
  ```
  <All keys matched successfully>
  labels: [0, 1, 1, 2, 2, 3, 5, 6, 8, 9]
  saved ./output/images.png
  saved ./output/lime.png
  saved ./output/saliency.png
  saved ./output/smoothgrad.png
  saved ./output/filter_cnn6.png
  saved ./output/filter_cnn23.png
  saved ./output/integrated_gradients.png
  ```
  stderr 有 10 條 tqdm 進度條（LIME 每張圖一條 `1000/1000`，約 450–700 it/s）。
- `../.venv/bin/python bert_hidden_states.py`：real 12.3 s。stdout：
  ```
  saved ./output/bert_q1_layer{1..12}.png
  saved ./output/bert_q2_layer{1..12}.png
  saved ./output/bert_q3_layer{1..12}.png
  ```
  stderr 有 transformers 5 的 LOAD REPORT：`bert.pooler.dense.weight`、`bert.pooler.dense.bias` 標 UNEXPECTED（QA 模型不用 pooler，可忽略）。
- `../.venv/bin/python bert_embedding.py`：real 6.7 s。stdout：`saved ./output/bert_embedding.png`。stderr LOAD REPORT：7 個 `cls.predictions.*`、`cls.seq_relationship.*` 標 UNEXPECTED（預訓練用的 MLM/NSP 頭，`BertModel` 不用）。
- 第一次跑 BERT 腳本會從 HF 下載模型（bert-base-chinese 約 400 MB）；之後在 `~/.cache/huggingface`。
- 所有 transformers 的 LOAD REPORT 都以 `Notes: - UNEXPECTED: can be ignored when loading from different task/architecture; not ok if you expect identical arch.` 結尾。

## CNN 實測

### 模型（ch00 模型總覽用）
- `model.cnn` 共 38 層（index 0–37），5 個 stage：
  - stage 1：cnn[0–8] = 3×(Conv 3×3 + BN + ReLU)，3→128→128→128；cnn[9] MaxPool → (128, 64, 64)
  - stage 2：cnn[10–18] = 3×(Conv+BN+ReLU) 128→128；cnn[19] MaxPool → (128, 32, 32)
  - stage 3：cnn[20–28] = 3×(Conv+BN+ReLU) 128→256→256→256；cnn[29] MaxPool → (256, 16, 16)
  - stage 4：cnn[30–32] Conv 256→512 + BN + ReLU；cnn[33] MaxPool → (512, 8, 8)
  - stage 5：cnn[34–36] Conv 512→512 + BN + ReLU；cnn[37] MaxPool → (512, 4, 4)
  - 全部 Conv 是 `kernel_size=3, stride=1, padding=1`（不改尺寸），只有 MaxPool(2,2) 讓長寬減半：128→64→32→16→8→4。
- `model.fc`：Linear(8192, 1024) → ReLU → Dropout(0.3) → Linear(1024, 11)。8192 = 512×4×4。
- 參數量（PyTorch 數的）：總計 **14,162,827**；cnn 5,761,920；fc 8,400,907（Linear 8192·1024+1024 = 8,389,632；1024·11+11 = 11,275）。fc 佔 59%。
- Filter explanation 用到的兩層：**cnn[6] = 第 1 stage 第 3 個 Conv2d(128,128)**，輸出 (N, 128, 128, 128)；**cnn[23] = 第 3 stage 第 2 個 Conv2d(256,256)**，輸出 (N, 256, 32, 32)。兩個都是 **Conv 層本身（在 BN、ReLU 之前）**，所以 activation 有正有負（下面「filter」實測 zero fraction 0.0，filter 0 的總和是負的）。
- `forward`：`out.reshape(out.size()[0], -1)` 把 (N,512,4,4) 攤成 (N,8192)。

### 預測（10 張全對）
| 編號 | 標籤 | 預測 | p(標籤) | 第 2 名 |
|---|---|---|---|---|
| 0 | Bread | 0 | 1.0000 | Fried food 0.0000 |
| 1 | Dairy product | 1 | 1.0000 | Dessert 0.0000 |
| 2 | Dairy product | 1 | 0.9986 | Dessert 0.0007 |
| 3 | Dessert | 2 | 1.0000 | Seafood 0.0000 |
| 4 | Dessert | 2 | 1.0000 | Noodles/Pasta 0.0000 |
| 5 | Egg | 3 | 1.0000 | Meat 0.0000 |
| 6 | Meat | 5 | 1.0000 | Fried food 0.0000 |
| 7 | Noodles/Pasta | 6 | 1.0000 | Meat 0.0000 |
| 8 | Seafood | 8 | 1.0000 | Vegetable/Fruit 0.0000 |
| 9 | Soup | 9 | 1.0000 | Dessert 0.0000 |
- 10/10 正確，而且除了圖 2 之外 softmax 都四捨五入到 1.0000。這些圖很可能是訓練集裡的圖（Colab 變數就叫 `train_set`）。**這件事讓 saliency 的梯度極小**（見下）。
- 目標類別的 logit（IG 實測一併量到）：圖 0–9 分別 10.331、19.787、7.808、9.024、12.216、12.389、22.948、11.446、13.004、14.110。

### LIME（Q1–4，explain_cnn.py:39–74）
- `explain_instance` 預設：`num_samples=1000`、`batch_size=10`、`top_labels=5`、`hide_color=None`（被關掉的 superpixel 換成該塊的平均色）、`distance_metric='cosine'`。所以每張圖呼叫 `predict` **100 次**，每次輸入 (10, 128, 128, 3)。
- 每張圖約 1.3–1.6 s（圖 1 是 2.81 s）。
- `np.random.seed(16)` 在迴圈外設一次；`LimeImageExplainer()` 沒給 `random_state`，所以用 numpy 全域亂數 → 整段可重現（實測同 seed 重跑兩次權重完全相同）。但第 i 張圖的結果依賴前面 i−1 張用掉了多少亂數。
- `slic(n_segments=200, compactness=1, sigma=1, start_label=1)` 實際切出的 superpixel 數**不是 200**：圖 0–9 分別 107、141、151、101、66、120、108、94、128、109 塊。
- `predict` 回傳的是 **logits**（`model(...)` 直接輸出，沒有 softmax），LIME 文件預期的是機率。實測圖 0：
  - logits（原版）：權重前 5 名 (21, 5.52)、(25, 3.59)、(38, 3.12)、(27, 3.02)、(40, 2.94)；|w| ≥ 0.05 的有 83 塊；local model 的 R²（`exp.score`）0.842。
  - 改成 softmax 機率：前 5 名 (21, 0.219)、(40, 0.198)、(25, 0.191)、(27, 0.145)、(23, 0.127)；|w| ≥ 0.05 只剩 13 塊；R² 0.541。
  - 前幾名的 superpixel 大致相同，但 `min_weight=0.05` 這個門檻的意義完全不同（logit 尺度 vs 機率尺度）。
- **差一錯誤（off-by-one，實測確認）**：`start_label=1` 讓 superpixel 編號是 1..n，但 LIME 0.2.0.1 把第 z 個特徵對應到 `segments == z`，z = 0..n−1。結果：
  - 特徵 0 對應不到任何像素（圖 0 它仍拿到權重 0.158，純雜訊）。
  - 編號 n 的那一塊**永遠不會被遮**，也不會出現在解釋裡（圖 0 是第 107 塊，87 個像素）。
  - 改 `start_label=0` 後，圖 0 前 5 名變成 (20, 5.27)、(24, 3.41)、(37, 3.32)、(39, 3.16)、(26, 3.11)，R² 0.862。編號整體平移 1，排名大致相同。原版 notebook 也是 `start_label=1`。
- `get_image_and_mask(label, positive_only=False, hide_rest=False, num_features=11, min_weight=0.05)`（lime_image.py 實作）：按 |權重| 排序取前 11 個，略過 |w| < 0.05 的；**正權重**的那塊把 G 通道設成影像最大值（**綠**，支持這個類別），**負權重**的把 R 通道設成最大值（**紅**，反對）。
- 每張圖 |w| ≥ 0.05 的 superpixel 數（正/負），以及畫出來的前 11 塊（正/負）：
  | 圖 | 塊數 | |w|≥0.05 (正/負) | 畫出 11 塊 (正/負) | 前 3 名 (塊, 權重) |
  |---|---|---|---|---|
  | 0 | 107 | 83 (47/36) | 11/0 | (21, 5.52) (25, 3.59) (38, 3.12) |
  | 1 | 141 | 33 (16/17) | 4/7 | (55, −1.16) (25, −0.39) (42, −0.34) |
  | 2 | 151 | 30 (14/16) | 2/9 | (95, −1.77) (128, −0.95) (75, −0.78) |
  | 3 | 101 | 60 (31/29) | 7/4 | (40, 1.06) (57, 0.94) (86, −0.80) |
  | 4 | 66 | 52 (32/20) | 10/1 | (5, 3.56) (15, 1.75) (26, 1.60) |
  | 5 | 120 | 76 (54/22) | 10/1 | (92, 1.00) (109, 0.89) (12, 0.87) |
  | 6 | 108 | 88 (63/25) | 11/0 | (30, 5.22) (15, 4.97) (40, 4.19) |
  | 7 | 94 | 83 (58/25) | 11/0 | (48, 14.02) (28, 12.87) (51, 8.74) |
  | 8 | 128 | 91 (49/42) | 11/0 | (58, 8.27) (78, 4.93) (69, 2.69) |
  | 9 | 109 | 84 (52/32) | 8/3 | (22, 6.63) (5, 3.50) (15, 2.26) |
- R²（`exp.score`）圖 0–9：0.842、0.874、0.956、0.794、0.796、0.814、0.925、0.951、0.961、0.908。
- 觀察：兩張 Dairy product（圖 1、2）的最強權重都是**負**的，畫出來紅多於綠。其他圖幾乎全綠。可以讓讀者看 img/lime.png 自己找。

### Saliency map（Q5–9，explain_cnn.py:79–113）
- 10 張圖一起算 `CrossEntropyLoss`（mean），loss = **0.0001**（因為模型幾乎 100% 確定）。
- `x.grad` 形狀 (10, 3, 128, 128)；`torch.max(..., dim=1)` 對 RGB 取最大 → saliency (10, 128, 128)。各圖取到最大值的通道比例大約 R 35%、G 41%、B 22%。
- 正規化前每張圖 |梯度| 的最大值（差了很多個數量級）：
  | 圖 | 最大值 | 平均 |
  |---|---|---|
  | 0 | 8.48e-07 | 3.75e-08 |
  | 1 | 3.22e-12 | 9.47e-14 |
  | 2 | 3.75e-04 | 1.29e-05 |
  | 3 | 1.37e-07 | 5.48e-09 |
  | 4 | 4.24e-08 | 2.76e-09 |
  | 5 | 1.27e-07 | 8.38e-09 |
  | 6 | 1.52e-17 | 9.08e-19 |
  | 7 | 1.51e-06 | 8.50e-08 |
  | 8 | 1.08e-09 | 5.84e-11 |
  | 9 | 8.52e-08 | 3.33e-09 |
  最大／最小 ≈ **2.5 × 10¹³**（圖 2 對圖 6）。圖 2 是唯一 p(標籤) 不到 1.0000 的（0.9986），梯度也最大。這就是程式註解說「每張要各自 normalize，因為梯度的尺度可能差很多」的實際情況。
- 理由：CE 對 logit 的梯度是 softmax − one-hot，p → 1 時趨近 0。
- 投影片 p.8 說「gradient of output category」，程式算的是 **CE loss** 的梯度：∂L/∂x = Σ_j (p_j − 1[j=y]) ∂z_j/∂x，是所有類別 logit 梯度的加權和，權重在 p_y → 1 時全部趨近 0。和「目標類別分數的梯度」∂z_y/∂x 不是同一個量（IG 那段用的才是後者）。
- 「不各自正規化會幾乎全黑」是由上表推得（例如圖 6 是圖 2 的 4×10⁻¹⁴ 倍），沒有另外畫圖驗證。
- `normalize` = (x − min)/(max − min)，逐張做；顯示用 `cmap=plt.cm.hot`。

### SmoothGrad（Q10–13，explain_cnn.py:120–157）
- 呼叫參數：`epoch=500`（取樣次數）、`param_sigma_multiplier=0.4`。
- 程式：`sigma = 0.4 / (max(x) − min(x))`，雜訊 `normal_(mean, sigma**2)`。`Tensor.normal_` 的第二個參數是**標準差**，所以實際雜訊 std = (0.4/範圍)²。
- 論文（Smilkov et al. 2017, arXiv 1706.03825）的定義是 σ = 雜訊比例 × (x_max − x_min)，是**乘**，而且直接當標準差。這份程式是**除再平方**（原版 notebook 就是這樣，本 repo 照搬）。
- 因為圖片值域都接近 0–1，兩種算法數字差距不大，但不是同一件事：
  | 圖 | max−min | 程式的 sigma | 實際 std（sigma²） | 論文式 0.4×範圍 |
  |---|---|---|---|---|
  | 0 | 0.9961 | 0.4016 | 0.1613 | 0.3984 |
  | 1 | 0.9490 | 0.4215 | 0.1777 | 0.3796 |
  | 6 | 1.0000 | 0.4000 | 0.1600 | 0.4000 |
  其他圖 std 在 0.160–0.178 之間。實際雜訊約是論文式的 **40%**。
- 每次取樣都是 1 張圖單獨 forward/backward；圖 0 跑 500 次約 2.45 s（10 張約 25 s，是 explain_cnn.py 最慢的一段）。
- 正規化前的平均梯度（圖 0）：1 次取樣 max 0.902、min 2.3e-07；10 次 max 0.311；50 次 max 0.271；500 次 max 0.256、min 3.0e-03。**加了雜訊之後梯度大了好幾個數量級**（對比 saliency 圖 0 的 8.48e-07）：雜訊把圖推離模型「100% 確定」的區域，CE 梯度不再接近 0。取樣越多，極端值被平均掉（max 從 0.90 降到 0.26）。
- 結果保留 3 個通道（沒有對 RGB 取 max），所以 img/smoothgrad.png 第二列是**彩色**的，不是 hot colormap。
- 程式註解 `# smooth = smooth / epoch # try this line to answer the question`：不 normalize 時直接 imshow 浮點數，matplotlib 會把 >1 的截斷；圖 0 的值域約 3e-3..0.26（500 次），會整張偏暗。<!-- 沒有存這張圖；若章節要放，標 TODO(本機實測) -->
- 程式裡 `x.cuda().unsqueeze(0)` 只用來取形狀 (1,3,128,128) 給 `np.zeros`。

### Filter explanation（Q14–17，explain_cnn.py:164–221）
- 用 `register_forward_hook` 抓中間層輸出，存到模組層全域變數 `layer_activations`（hook 裡 `global layer_activations`）。用完 `hook_handle.remove()`。
- 參數：`filterid=0`、`iteration=100`、`lr=0.1`（呼叫端給的；函式預設 lr=1）。優化器 Adam，對象是**輸入圖 x 本身**，目標 = −(filter 0 activation 的總和)，也就是 gradient ascent。
- 10 張圖一起優化（batch），目標是 10 張的 activation 總和。
- **起點是那 10 張原圖，不是白雜訊**。投影片 p.10 寫「start from white noise」，程式和原版 notebook 都是從 `images` 開始。所以第三列看起來會留著原圖的輪廓。
- 優化時沒有把 x 限制在 [0,1]，100 步後：cnn[6] x 值域 **−11.87 .. 12.32**；cnn[23] **−6.44 .. 7.49**。顯示前用 `normalize` 拉回 0–1。
- cnn[6]，filter 0 activation 形狀 (10, 128, 128)：
  - 原圖時每張的總和：−31896、−44248、−47659、−24998、−45768、−27898、−31637、−43263、−24900、−33663（全部是負的）。
  - 10 張合計，第 1、2、10、50、100 步（優化前的值）：−355,931 → −364,330 → 598,087 → 3,828,560 → 9,108,426。
- cnn[23]，filter 0 activation 形狀 (10, 32, 32)：
  - 原圖時每張的總和：−32662、−13225、−17830、−33577、−40429、−36473、−39921、−31147、−26858、−42131。
  - 10 張合計：−314,252 → −532,294 → −5,249 → 239,570 → 371,643。第 2 步反而更負（Adam 起步）。
- 「filter activation」那一列（第二列）畫的是**優化前**原圖的 activation map（`model(x.cuda())` 那一次 forward）。

### Integrated Gradients（Q18–20，explain_cnn.py:225–281）
- `steps=10`；`generate_images_on_linear_path` 產生 `input * step/steps`，step = 0..9，也就是 α = 0.0, 0.1, …, 0.9（**不含 α=1**，左黎曼和）。baseline = 全黑圖（全 0）。投影片 p.11 寫「Flexible baseline」，程式固定用 0。
- 梯度對象是**目標類別的 logit**（`model_output.backward(gradient=one_hot_output)`），不是 loss、也不是 softmax 機率。
- **程式少乘了 (x − baseline)**：論文（Sundararajan et al. 2017, arXiv 1703.01365）的 IG = (x − x') × 路徑上梯度的平均；程式只回傳梯度平均。所以畫出來的是「路徑平均梯度」，不滿足 IG 的 completeness（歸因總和 = f(x) − f(baseline)）。實測：
  | 圖 | f(x) | f(0) | 差 | Σ 平均梯度（程式的） | Σ 平均梯度 × x（論文的） |
  |---|---|---|---|---|---|
  | 0 | 10.331 | −7.580 | 17.911 | 0.273 | 19.119 |
  | 1 | 19.787 | −3.327 | 23.114 | 9.356 | 24.851 |
  | 2 | 7.808 | −3.327 | 11.135 | 8.418 | 6.439 |
  | 3 | 9.024 | −3.658 | 12.682 | −7.373 | 10.975 |
  | 4 | 12.216 | −3.658 | 15.873 | 20.791 | 17.960 |
  | 5 | 12.389 | −1.921 | 14.310 | 9.317 | 12.767 |
  | 6 | 22.948 | −3.364 | 26.311 | 1.506 | 23.017 |
  | 7 | 11.446 | −5.953 | 17.399 | −7.293 | 16.557 |
  | 8 | 13.004 | −3.524 | 16.528 | −7.785 | 12.057 |
  | 9 | 14.110 | −0.412 | 14.522 | 14.499 | 16.327 |
  乘上 x 之後，10 步就已經接近差值；改用中點法且步數加多，圖 0：10 步 15.704、50 步 18.141、200 步 17.851（→ 17.911）。
- 全黑圖的 logit f(0) 對 Dairy product（圖 1、2）都是 −3.327、對 Dessert（圖 3、4）都是 −3.658：同一個類別的 baseline 輸出相同，這是合理的。
- 每個 xbar 單獨 forward/backward；`self.model.zero_grad()` 清的是模型參數的梯度，輸入的梯度因為每個 xbar 都是新 tensor 所以不會累積。
- 顯示：`np.moveaxis(normalize(img), 0, -1)`，3 通道彩色（和 SmoothGrad 一樣不取 max）。

## BERT 實測

### Q21–24 exBERT（只有網站，無程式）
見「原版 vs 本 repo」。exbert.net 目前連不上；HF Space 網址存在。

### Q25–27 hidden states 視覺化（bert_hidden_states.py，用 deepset/bert-base-cased-squad2）
- 模型：BERT-base cased，vocab 28,996，`do_lower_case=False`；12 層，hidden 768。`hidden_states` 是 13 個元素的 tuple（第 0 個是 embedding 層輸出，1–12 是各層），每個 (1, seq_len, 768)。程式跳過第 0 個，畫 Layer 1–12。
- PCA 降到 2 維（`random_state=0`）；每張圖畫 question（紅）、context（綠）、answer（藍菱形），[CLS]/[SEP] 不畫。
- 三題：
  | 題 | 問題 | 答案 | seq len | question 位置 | context 位置 | 被標成 answer 的 token | 這個模型實際預測 |
  |---|---|---|---|---|---|---|---|
  | Q1 | In what year was Nikola Tesla born? | 1856 | 78 | 1..9 | 11..76 | 位置 32 `1856` | 32..32 = `1856` ✔ |
  | Q2 | What is a common punishment in the UK and Ireland? | detention | 120 | 1..11 | 13..118 | 位置 14、86、94，三個 `detention` 都標藍 | 0..0 = `[CLS]`（SQuAD 2.0 的「無答案」） |
  | Q3 | What is Emily afraid of? | cats | 56 | 1..6 | 8..54 | 位置 12 `cats`（"Wolves are afraid of cats"）；後面的 `Cats` 大寫不算 | 0..0 = `[CLS]`（無答案） |
- 答案判斷是 `word in answers.split()`，逐 token 字串比對，所以 Q2 會有三個藍點；cased 模型下 `Cats` ≠ `cats`。
- context 字串是用 `\` 續行寫的，續行裡的縮排空白會進到字串裡，tokenizer 會吃掉，不影響 token。
- Q1 的 tokens：`['[CLS]', 'In', 'what', 'year', 'was', 'Nikola', 'Te', '##sla', 'born', '?', '[SEP]', 'Nikola', 'Te', '##sla', '(', 'Serbian', 'Cyrillic', ':', 'Н', '##и', '##к', '##о', '##л', '##а', 'Т', '##е', '##с', '##л', '##а', ';', '10', 'July', '1856', '–', '7', 'January', '1943', ')', 'was', 'a', 'Serbian', 'American', 'inventor', …]`（西里爾字母被拆成單字母 wordpiece）。圖上的字是 `Tokenizer.decode(token_id)` 單獨解碼，所以會出現 `##sla` 這種片段。
- Q3 tokens：`['[CLS]', 'What', 'is', 'Emily', 'afraid', 'of', '?', '[SEP]', 'Wolves', 'are', 'afraid', 'of', 'cats', '.', 'She', '##ep', 'are', 'afraid', 'of', 'wolves', '.', 'Mi', '##ce', 'are', 'afraid', 'of', 'sheep', '.', 'Gertrude', 'is', 'a', 'mouse', '.', 'Jessica', 'is', 'a', 'mouse', '.', 'Emily', 'is', 'a', 'wolf', '.', 'Cats', 'are', 'afraid', 'of', 'sheep', '.', 'Win', '##ona', 'is', 'a', 'wolf', '.', '[SEP]']`（`Sheep` → `She ##ep`、`Mice` → `Mi ##ce`）。
- 前 2 個主成分解釋的變異比例，Layer 1→12：
  - Q1：0.14 0.15 0.14 0.14 0.14 0.15 0.15 0.17 0.19 0.22 0.24 **0.38**
  - Q2：0.14 0.13 0.12 0.12 0.12 0.13 0.13 0.15 0.17 0.19 0.22 **0.48**
  - Q3：0.26 0.26 0.24 0.23 0.23 0.23 0.22 0.22 0.22 0.23 0.26 **0.49**
  → 前面的層 2D 投影只保留 12–26% 的資訊，圖上的距離不太可靠；最後一層結構最集中。
- 觀察 img/bert_q3_layer12.png：句號 `.` 聚成一群、問題 tokens 與 `Emily`/`is`/`afraid` 等聚在右側、動物名詞（sheep、wolves、cats、wolf）聚在左側，答案 `cats` 在動物群裡。其他層請看圖，**不要編造**沒看過的圖的描述；需要描述哪一層時，若這裡沒寫，標 TODO(本機實測)。
- 原作業的「四個步驟對應哪幾層」是用助教的模型出的題；換了模型之後**沒有標準答案**，教材要明說這點。

### Q28–30 embedding 分析（bert_embedding.py，bert-base-chinese）
- 10 句、`select_word_index` 指向「蘋」；第二行（註解掉的）指向「果」。`word_to_tokens(i).start` 把字的位置換成 token 位置（[CLS] 佔 0，所以通常 +1）。
- 斷詞（bert-base-chinese 一字一 token；英文不在字表裡變 `[UNK]`）：
  | # | 句子 | tokens 數 | 蘋 index→token | 果 index→token | 備註 |
  |---|---|---|---|---|---|
  | 0 | 今天買了蘋果來吃 | 10 | 4→5 | 5→6 | |
  | 1 | 進口蘋果（富士)平均每公斤下跌12.3% | 21 | 2→3 | 3→4 | `12`、`.`、`3`、`%` |
  | 2 | 蘋果茶真難喝 | 8 | 0→1 | 1→2 | |
  | 3 | 老饕都知道智利的蘋果季節即將到來 | 18 | 8→9 | 9→10 | |
  | 4 | 進口蘋果因防止水分流失故添加人工果糖 | 20 | 2→3 | 3→4 | |
  | 5 | 蘋果即將於下月發振新款iPhone | 14 | 0→1 | 1→2 | `iPhone` → `[UNK]` |
  | 6 | 蘋果獲新Face ID專利 | 10 | 0→1 | 1→2 | `Face`、`ID` → `[UNK]` `[UNK]` |
  | 7 | 今天買了蘋果手機 | 10 | 4→5 | 5→6 | |
  | 8 | 蘋果的股價又跌了 | 10 | 0→1 | 1→2 | |
  | 9 | 蘋果押寶指紋辨識技術 | 12 | 0→1 | 1→2 | |
  （句 1 原文用全形左括號「（」配半形右括號「)」，原版就是這樣。句 5「發振」應為「發表」之類，原版錯字，照抄。）
- 語意分組（教材用）：水果 = 0、1、2、3、4（2 是蘋果茶）；公司 = 5、6、7、8、9（7 是蘋果手機）。
- Layer 12、「蘋」、歐氏距離矩陣（= img/bert_embedding.png 上的數字）：
  ```
  [[ 0.   11.83 16.95 11.74 11.49 25.86 20.01 14.29 17.03 19.97]
   [11.83  0.   18.85 11.75 10.06 25.45 19.37 15.44 16.34 19.21]
   [16.95 18.85  0.   17.58 15.71 16.11 12.85 20.73 14.84 15.16]
   [11.74 11.75 17.58  0.   11.8  24.98 19.18 15.12 16.93 18.83]
   [11.49 10.06 15.71 11.8   0.   24.27 17.9  17.29 16.51 18.39]
   [25.86 25.45 16.11 24.98 24.27  0.   12.2  25.77 17.92 16.35]
   [20.01 19.37 12.85 19.18 17.9  12.2   0.   19.89 11.46  9.79]
   [14.29 15.44 20.73 15.12 17.29 25.77 19.89  0.   15.71 17.85]
   [17.03 16.34 14.84 16.93 16.51 17.92 11.46 15.71  0.    9.58]
   [19.97 19.21 15.16 18.83 18.39 16.35  9.79 17.85  9.58  0.  ]]
  ```
- 同條件的 cosine similarity 矩陣（把 `METRIC = cosine_similarity` 的結果）：
  ```
  [[1.   0.85 0.69 0.86 0.86 0.31 0.57 0.79 0.68 0.57]
   [0.85 1.   0.62 0.86 0.89 0.34 0.6  0.76 0.71 0.61]
   [0.69 0.62 1.   0.66 0.73 0.72 0.82 0.55 0.75 0.74]
   [0.86 0.86 0.66 1.   0.85 0.36 0.6  0.77 0.69 0.62]
   [0.86 0.89 0.73 0.85 1.   0.39 0.65 0.69 0.7  0.63]
   [0.31 0.34 0.72 0.36 0.39 1.   0.84 0.33 0.65 0.72]
   [0.57 0.6  0.82 0.6  0.65 0.84 1.   0.59 0.85 0.89]
   [0.79 0.76 0.55 0.77 0.69 0.33 0.59 1.   0.74 0.67]
   [0.68 0.71 0.75 0.69 0.7  0.65 0.85 0.74 1.   0.9 ]
   [0.57 0.61 0.74 0.62 0.63 0.72 0.89 0.67 0.9  1.  ]]
  ```
  注意：`pairwise_distances(metric=cosine_similarity)` 會把相似度當「距離」填進去，對角線是 1 而不是 0；顏色亮 = 像（和歐氏距離剛好相反）。
- 讀矩陣的觀察（layer 12、蘋）：
  - 0、1、3、4（真的在講吃的水果）彼此距離 10.06–11.83，最緊。
  - **句 2「蘋果茶真難喝」反而離公司那群比較近**：到 6 是 12.85、到 8 是 14.84，比到 1 的 18.85、到 3 的 17.58 都近。
  - **句 7「今天買了蘋果手機」離句 0「今天買了蘋果來吃」最近**（14.29），離其他公司句 15.71–25.77；表層句型（今天買了…）壓過了語意。
  - 句 5（iPhone 那句，有 `[UNK]`）是離群值：到水果句 24–26。
  - 公司句 6、8、9 彼此 9.58–11.46，很緊。
- 各層的組內／組間平均距離（歐氏，蘋；水果 = 0–4、公司 = 5–9；「組間」越大於「組內」= 越分得開）：
  | layer | 水果組內 | 公司組內 | 組間 |
  |---|---|---|---|
  | 0 | 6.142 | 3.064 | 5.596 |
  | 1 | 10.386 | 9.454 | 11.121 |
  | 2 | 11.427 | 10.138 | 12.326 |
  | 3 | 11.149 | 9.928 | 12.571 |
  | 4 | 12.384 | 11.772 | 15.230 |
  | 5 | 11.591 | 11.726 | 15.200 |
  | 6 | 11.438 | 11.754 | 15.643 |
  | 7 | 12.169 | 12.299 | 16.007 |
  | 8 | 11.563 | 11.447 | 15.804 |
  | 9 | 11.450 | 10.945 | 15.867 |
  | 10 | 10.745 | 10.695 | 14.956 |
  | 11 | 10.618 | 10.910 | 14.818 |
  | 12 | 13.776 | 15.652 | 18.481 |
  - Layer 0（embedding 層，還沒看上下文）：組間 5.596 **小於**水果組內 6.142 → 完全分不開；同一個字「蘋」的向量只差在位置 embedding（公司句的「蘋」幾乎都在句首，所以公司組內只有 3.064）。
  - 從 layer 4 起組間明顯大於兩個組內（差 +2.8 到 +4.9）；layer 8–9 分得最開（組間 − 組內 = 4.2–4.9）。layer 12 對水果組 +4.7、對公司組只剩 +2.8。
  - 歐氏距離的絕對值會隨層變（layer 12 整體變大），跨層比較要看比值或用 cosine。
- 同上，cosine similarity（蘋）：layer 0 水果 0.956／公司 0.976／組間 0.959；layer 9 0.886／0.892／0.781；layer 12 0.788／0.719／0.625。
- 「果」（第二行 index）的趨勢相同：layer 0 組間 5.944 < 水果組內 6.602；layer 8 組內 12.646／12.342、組間 17.445；layer 12 組內 12.375／11.570、組間 17.130。cosine layer 12：0.854／0.874／0.726。
- 改 `LAYER`、`METRIC`、或換「果」都只需改 TODO 區；完整數字可用 `hw09_facts.py embed` 重印。

## 圖檔清單（docs/HW09/img/，2026-10-03 產生，與上面數字同一次執行）
- `images.png`：10 張原圖（1×10）。
- `lime.png`：LIME 疊色（1×10）。
- `saliency.png`：上原圖、下 saliency 熱圖（2×10，hot colormap）。
- `smoothgrad.png`：上原圖、下 SmoothGrad（2×10，彩色）。
- `filter_cnn6.png`、`filter_cnn23.png`：上原圖、中 filter 0 activation、下 filter visualization（3×10）。
- `integrated_gradients.png`：上原圖、下 IG（2×10，彩色）。
- `bert_q{1,2,3}_layer{1..12}.png`：36 張 PCA 散布圖。
- `bert_embedding.png`：10×10 歐氏距離矩陣（layer 12、蘋）。
- 所有 CNN 圖都有 matplotlib 預設的座標軸刻度（0–100），原版 notebook 也有。
- 雲端 session 可以直接打開這些 PNG 看；描述圖的內容時只寫看得到的，不確定就標 TODO。

## 建議章節（給雲端寫 outline 時參考，可調整，outline 要先給使用者確認）
- index：「先說在前面：2022 vs 2026」（Colab → 本機、lime/transformers 版本、hw9_bert 下架、exBERT 連不上）。
- ch00 導讀：任務（投影片 30 題、不用訓練）、模型總覽（Classifier 的 5 個 stage、形狀、14,162,827 參數）、三支腳本怎麼跑、輸出在哪。
- ch01 資料與模型載入：checkpoint 結構、10 張圖、getbatch、預測全對且接近 100% 確定（後面每章都會用到這個事實）。
- ch02 LIME：superpixel、取樣 1000 次、logits vs 機率、start_label 差一錯誤、綠/紅的意思。
- ch03 Saliency map 與 SmoothGrad：對輸入取梯度、梯度消失到 1e-17、為什麼要逐張 normalize、雜訊 std 的算法。
- ch04 Filter explanation：forward hook、gradient ascent、從原圖而不是雜訊開始、x 跑出 [0,1]。
- ch05 Integrated Gradients：路徑積分、completeness、程式少乘 x。
- ch06 BERT hidden states：hidden_states tuple、PCA、換模型的影響、exBERT。
- ch07 BERT embedding 分析：蘋果的 10 句、距離矩陣、各層的組內／組間。
- appendix：名詞表、指令速查、關鍵數字、repo 問題清單。

## repo 問題清單（教材附錄素材；本 repo 照原版保留，未修）
1. LIME `start_label=1` 差一：最後一塊 superpixel 永遠不被解釋，特徵 0 是假特徵。
2. LIME `classifier_fn` 回傳 logits 而非機率，`min_weight=0.05` 的意義因此改變。
3. SmoothGrad 雜訊 std = (0.4/範圍)²，論文是 0.4×範圍。
4. IG 少乘 (x − baseline)，baseline 固定為 0，α 不含 1。
5. Filter visualization 從原圖開始而非雜訊（與投影片 p.10 不同），且沒有把 x 限制在 [0,1]。
6. Saliency 對 loss 而非類別分數取梯度（與投影片 p.8 用詞不同），在模型極度自信時梯度接近 0。
7. `hw9_bert.zip` 下架，Q25–27 無法重現原作業的圖。
8. bert_embedding 句 5「發振」錯字、句 1 括號全半形混用（原版如此）。

## 已在前面章節定義過的名詞
（每章寫完由雲端 session 追加，格式比照 docs/HW01/FACTS.md。）
- index（目錄頁，2026-10-03）：Explainable AI／XAI（「模型根據什麼給出答案」）、tokenizer（一句：把文字切成 token 的工具）、token、hidden states（BERT 每一層的輸出向量）、Hugging Face（公開分享模型的網站，transformers 的開發者）、Hugging Face Space（一句：放網頁小程式的地方）、BERT-base（12 層的標準尺寸）、微調（一句）、SQuAD 2.0（問答資料集）、attention（一句：每個 token 參考哪些其他 token 與比重）、exBERT（看 attention 的網站）、PCA（一句：把高維向量壓成 2 維好畫圖）、superpixel（一句：LIME 把圖切成的小塊）、logits（一句：還沒經過 softmax 的原始分數）、filter（一句：卷積層裡的小圖案偵測器）、activation（filter 的輸出）、基準圖（IG 的全黑圖，一句）、Captum（一句）、`Variable`／`.data` → `.detach()`、`matplotlib.use('Agg')`。以上大多只在目錄頁給了一句話，正式定義仍要在各章首次使用處寫完整。
  index 的方法名稱：Filter explanation（投影片稱 Filter visualization，兩者同一個方法）。全書配色：藍＝輸入資料、綠＝模型 forward、黃＝梯度、紫＝解釋方法自己的步驟；全書地圖 SVG 用綠框＝只做 forward、黃框＝要對輸入 backward。
  outline 的用語：bert_hidden_states.py 的三組問答稱「第 1／2／3 組問答」，**不要**寫 Q1–Q3（會和作業題號撞名）。

## index／outline 審稿補測（2026-10-03，本機）
- 舊版套件能不能裝（用 `uv pip install --target <scratch>` 試，不動 .venv）：
  - `lime==0.1.1.37`：**裝得起來**，在 Python 3.12 上也能跑。它另外依賴 `progressbar`（2.5），pip 會一起裝。舊版的進度條是 `progressbar` 的 `|####|` 樣式，不是 tqdm。用它對圖 0 跑同一段 LIME（seed 16、start_label=1、logits）：前 5 名 (21, 5.5216)、(25, 3.5899)、(38, 3.1247)、(27, 3.0233)、(40, 2.9437)，R² 0.8423，**和 0.2.0.1 完全相同**。所以換版本不影響 Q1–4 的結果。
  - `transformers==4.5.0`：**裝不起來**。它依賴的 `tokenizers` 0.10.3 在 Python 3.12 編譯失敗（uv：`Build failures usually indicate a problem with the package or the build environment`）。
- Captum 仍在維護：PyPI 最新版 0.9.0，2026-04-17 發佈（index「現在的做法」框可引用）。
- outline 的 `requirements.txt:10` 應為 `:11`（第 11 行是 `transformers==5.18.0`，第 10 行是 matplotlib），已修正。
- outline 與 index 引用的其他 file:line 範圍都核對過，邊界正確。

## ch00 實測（2026-10-03，本機；補大綱預告的 TODO）
- 讀者可以貼上執行的參數計數（在 `HW09/` 裡，不需要 checkpoint，CPU 上建模型即可）：
  ```
  ../.venv/bin/python -c "
  from model import Classifier
  m = Classifier()
  print(sum(p.numel() for p in m.parameters()))
  print(sum(p.numel() for p in m.cnn.parameters()), sum(p.numel() for p in m.fc.parameters()))
  "
  ```
  逐字輸出：
  ```
  14162827
  5761920 8400907
  ```
- 每個 MaxPool 之後的形狀（用一張全 0 的圖走一遍 `model.cnn`，需要 GPU）：
  ```
  ../.venv/bin/python -c "
  import torch
  from model import Classifier
  m = Classifier().cuda().eval()
  x = torch.zeros(1, 3, 128, 128).cuda()
  for i, layer in enumerate(m.cnn):
      x = layer(x)
      if isinstance(layer, torch.nn.MaxPool2d):
          print(i, tuple(x.shape))
  print(\"fc in\", x.reshape(1, -1).shape[1], \"out\", tuple(m.fc(x.reshape(1, -1)).shape))
  "
  ```
  逐字輸出：
  ```
  9 (1, 128, 64, 64)
  19 (1, 128, 32, 32)
  29 (1, 256, 16, 16)
  33 (1, 512, 8, 8)
  37 (1, 512, 4, 4)
  fc in 8192 out (1, 11)
  ```
- 手算用的逐層參數（Conv = 3·3·c_in·c_out + c_out；BN = 2·c，只有 weight 與 bias 是參數）：
  | cnn index | 層 | 參數 |
  |---|---|---|
  | 0 | Conv 3→128 | 3,584 |
  | 3、6、10、13、16 | Conv 128→128（各） | 147,584 |
  | 20 | Conv 128→256 | 295,168 |
  | 23、26 | Conv 256→256（各） | 590,080 |
  | 30 | Conv 256→512 | 1,180,160 |
  | 34 | Conv 512→512 | 2,359,808 |
  | 1、4、7、11、14、17 | BN 128（各） | 256 |
  | 21、24、27 | BN 256（各） | 512 |
  | 31、35 | BN 512（各） | 1,024 |
  加總：Conv 5,756,800 + BN 5,120 = cnn 5,761,920；再加 fc 8,400,907 = 14,162,827。
- BN 另有 **buffer**（不是參數、不訓練，但存在 state_dict 裡）：running_mean、running_var 各 c 個，加上 num_batches_tracked 1 個；11 個 BN 共 2,560 個 channel，所以 buffer 共 2×2,560 + 11 = **5,131**。
- `model_state_dict` 的 81 個 entry = 11 個 Conv × 2（weight、bias）+ 11 個 BN × 5（weight、bias、running_mean、running_var、num_batches_tracked）+ fc 2 個 Linear × 2。
- ch00（第 0 章，2026-10-03 雲端）：
  - 圖號：圖 0.1（Classifier 架構：5 個 stage + 攤平 + 2 個 Linear，每段輸出形狀與參數量）、圖 0.2（參數分布長條：cnn 41%、fc 59%）。
  - 本章正式定義的名詞：stage（本書稱呼：`stack_blocks` 的一次呼叫，程式裡沒有這個名字）、(N, C, H, W) 形狀記法、channel、Conv2d（filter = c_in×3×3 權重塊、activation／feature map、kernel 3／stride 1／padding 1 不改長寬）、BatchNorm（含 running mean／var）、ReLU、MaxPool、Linear（全連接）、Dropout、`model.eval()`（訓練／推論模式）、logits 的完整定義與 softmax 公式、參數（parameter）、buffer、`p.numel()`、global average pooling（「現在的做法」框）、`.gitignore`、stdout／stderr、tqdm、real／user time、Gradescope、Kaggle、LOAD REPORT 的 UNEXPECTED、pooler、MLM／NSP、預訓練／微調（一句）、BERT（一句）、token（一句）、embedding = 各層輸出向量 = hidden states（一句）、`python -c`。
  - 回指（不重講）：index 的 Hugging Face、tokenizer、hidden states、SQuAD 2.0、PCA、attention、基準圖；HW01 ch00 §0.3 的環境建置。
  - 本章用的手算（由上面逐層表推得，不是另外量的）：各 stage 參數 299,520／443,520／1,476,864／1,181,184／2,360,832；GAP 版 fc `Linear(512, 11)` = 5,643，全模型 5,767,563。
  - **更正「環境」一節**：只有 `explain_cnn.py` 寫死 `.cuda()`；`bert_hidden_states.py`、`bert_embedding.py` 沒有 `.cuda()`（只有 `same_seeds` 裡先檢查 `torch.cuda.is_available()` 的種子設定），模型與輸入都在 CPU。ch00 照原始碼寫；index（「三支腳本都把模型和資料寫死成 `.cuda()`」）與 outline #findings 末段的同一句需要改。
  - 在 repo 根目錄跑 `HW09/explain_cnn.py`：import 會成功（sys.path[0] 是腳本所在目錄），會在根目錄建 `output/`，然後 `torch.load('./checkpoint.pth')` 失敗。這是由 Python 規則推得，沒有實跑。

## ch00 審稿補測（2026-10-03，本機）
- 雲端的更正正確：只有 `explain_cnn.py` 有 `.cuda()`；兩支 BERT 腳本在 CPU（「環境」一節、index、outline 已改）。
- ch00 引用的所有行號（model.py、explain_cnn.py、bert_*.py）逐一核對，全對。
- **在 repo 根目錄跑 `.venv/bin/python HW09/explain_cnn.py`（實跑）**：根目錄多出空的 `output/`，最後一行 `FileNotFoundError: [Errno 2] No such file or directory: './checkpoint.pth'`（traceback 經 torch/serialization.py 的 `_open_file_like`）。
- **重新計時（`/usr/bin/time`，模型已在 HF 快取）**：explain_cnn.py real 50.65 s／user 653.59 s／sys 7.57 s；bert_hidden_states.py real 12.80 s／user 33.71 s；bert_embedding.py real 5.92 s／user 12.04 s。先前的 12.3 s、6.7 s 也是快取後的時間。注意：「執行實測」裡的 user 17 分 31 秒是**第一次（real 1 分 52 秒）**那次的，不能和 50.9 s 配對。
- **explain_cnn.py 逐段計時**（同一 process、模型已載入、`torch.cuda.synchronize()` 後量）：LIME 14.7 s、Saliency 0.4 s、SmoothGrad 24.1 s、Filter cnn[6] 2.3 s、cnn[23] 2.9 s、IG 0.7 s。
- **重跑的圖是否相同**（同一台機器，和 docs/HW09/img/ 逐位元比）：44 張中 40 張相同（images、lime、saliency、37 張 BERT）。不同的 4 張：
  - `smoothgrad.png`：雜訊用 PyTorch 亂數，explain_cnn.py 只設了 `np.random.seed(16)`，沒設 torch seed → 每次不同。像素差最大 38/255，約 16% 像素有差、0.3% 差超過 16。
  - `filter_cnn6.png`、`filter_cnn23.png`、`integrated_gradients.png`：GPU 非確定性（沒開 cudnn.deterministic），浮點尾數不同。filter_cnn6 最大 91/255 但只有 0.05% 像素差超過 16；filter_cnn23 最大 30；IG 最大 1。數值上：filter visualization（cnn[6]）總和兩次分別 299581.10、299577.31；IG 圖 0 總和 0.2728119、0.2727914。
- **下載大小**（HF 快取裡的 `model.safetensors`）：deepset/bert-base-cased-squad2 433,270,764 bytes；bert-base-chinese 411,553,788 bytes。
- **資料來源**：`ml2022spring-hw9.zip` 127,524,956 bytes。Kaggle 沒有 `ml2022spring-hw9` 競賽頁（`kaggle.com/competitions/ml2022spring-hw9` 回 404，同網址 hw1、hw8 回 200；投影片 p.17 也說 HW09 沒有排行榜）。原版 Colab 的三個 Google Drive id（food.zip `1QntUQuWJoVR8h5FoeDa56xrQSdcCwFeD`、checkpoint `1-Qw-oIJ0cSo2iG_n_U9mcJqXc2-LCSdV`、字型 `1JWHUSlcPwoEzmr0VE6J71jcnwinH10G6`）2026-10-03 全部 404。目前沒有公開下載來源。
- **zip 結構**：外層 `ml2022spring-hw9/checkpoint.pth`、`ml2022spring-hw9/food.zip`；`food.zip` 內含 `food/` 資料夾與 10 張 jpg。解壓指令（在 HW09/）：`unzip -j <路徑>/ml2022spring-hw9.zip ml2022spring-hw9/checkpoint.pth ml2022spring-hw9/food.zip`、`unzip food.zip`、`rm food.zip`。
- ch00 0.6 的版本指令實測輸出：`['2.11.0+cu128', '0.2.0.1', '5.18.0', '0.26.0', '1.9.1']`。

## ch01 實測（2026-10-03，本機；指令都在 HW09/ 裡執行）
### checkpoint.pth 裡有什麼
指令：
```
../.venv/bin/python -c "
import torch
ck = torch.load(\"checkpoint.pth\")
print(type(ck).__name__, list(ck.keys()))
print(\"epoch\", ck[\"epoch\"])
sd = ck[\"model_state_dict\"]
print(len(sd), list(sd.keys())[:6])
opt = ck[\"optimizer_state_dict\"]
print(list(opt.keys()), len(opt[\"state\"]))
print({k: v for k, v in opt[\"param_groups\"][0].items() if k != \"params\"})
"
```
逐字輸出：
```
dict ['epoch', 'model_state_dict', 'optimizer_state_dict']
epoch 208
81 ['cnn.0.weight', 'cnn.0.bias', 'cnn.1.weight', 'cnn.1.bias', 'cnn.1.running_mean', 'cnn.1.running_var']
['state', 'param_groups'] 48
{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}
```
- state_dict 的 key 名稱 = 屬性路徑： 是 ； 是 BN 的 buffer。
- 優化器是 **Adam**（lr 0.001、betas (0.9, 0.999)、eps 1e-8、無 weight decay）。 有 48 個 entry = 模型的 48 個參數張量（11 Conv × 2 + 11 BN × 2 + 2 Linear × 2）。每個 entry 有 、、（Adam 的一階、二階動量，形狀與參數相同，例如第一個是 (128, 3, 3, 3)）。
- **舊版 PyTorch 存的痕跡**： 只有  六個 key（新版還有 maximize、foreach、capturable 等）； 是 Python int （新版存成 tensor）； 的 key 是很大的整數，例如 ，不是新版的 0..47。這些不影響本作業：腳本只讀 。
-  32395 = 155 × 209。**推論**（沒有其他資料佐證）：epoch 從 0 數，存檔時是第 209 個 epoch，每個 epoch 155 個 batch；HW3 訓練集 9,866 張、batch 64 時正好是 ⌈9866/64⌉ = 155。
- **170 MB 的組成**（逐 tensor 加總 bytes）：
  - model_state_dict 56,671,876 bytes = 參數 14,162,827 × 4 bytes（float32）= 56,651,308，加上 BN buffer：running_mean/var 共 5,120 個 float32 = 20,480，再加 11 個 int64 的 num_batches_tracked = 88。dtype 只有 float32 與 int64。
  - optimizer 的 exp_avg + exp_avg_sq：113,302,616 bytes = 2 × 56,651,308。
  - 兩者合計 169,974,492；檔案 170,002,879；差的 28,387 bytes 是存檔格式本身（pickle 結構、key 名稱等）。
  - 結論：檔案的 **2/3 是 Adam 的動量**，推論時完全用不到。
-  在 torch 2.11 沒有指定 ，預設是 True（2.6 起）；這個檔案只含 dict、tensor、int、float、tuple，所以照樣載得進來，沒有警告。

### 10 張圖的檔名與順序
```
../.venv/bin/python -c "
import os
print(os.listdir('food'))
print(sorted(os.listdir('food')))
"
```
逐字輸出（第一行的順序取決於檔案系統，本機 WSL2 ext4；換機器可能不同）：
```
['9_9.jpg', '1_1.jpg', '2_4.jpg', '8_8.jpg', '0_0.jpg', '2_3.jpg', '3_5.jpg', '1_2.jpg', '5_6.jpg', '6_7.jpg']
['0_0.jpg', '1_1.jpg', '1_2.jpg', '2_3.jpg', '2_4.jpg', '3_5.jpg', '5_6.jpg', '6_7.jpg', '8_8.jpg', '9_9.jpg']
```
-  不保證順序，所以  一定要排序。
- 這 10 個檔名用普通字串排序（）剛好也得到同一個順序，因為類別都是個位數。 的用處在類別有兩位數時：
  ```
  ../.venv/bin/python -c "
  names = ['1_2.jpg', '10_3.jpg', '2_1.jpg']
  print(sorted(names))
  key = lambda n: int(n.replace('.jpg','').split('_')[1]) + 1000000 * int(n.split('_')[0])
  print(sorted(names, key=key), [key(n) for n in sorted(names, key=key)])
  "
  ```
  逐字輸出：
  ```
  ['10_3.jpg', '1_2.jpg', '2_1.jpg']
  ['1_2.jpg', '2_1.jpg', '10_3.jpg'] [1000002, 2000001, 10000003]
  ```
  字串排序把  排在  前面（逐字元比，'0' < '_'）； 把「類別 × 1,000,000 + 編號」當數字比。這要求編號小於 1,000,000。
-  的輸出：
  ```
  ['./food/0_0.jpg', './food/1_1.jpg', './food/1_2.jpg', './food/2_3.jpg', './food/2_4.jpg', './food/3_5.jpg', './food/5_6.jpg', './food/6_7.jpg', './food/8_8.jpg', './food/9_9.jpg']
  [0, 1, 1, 2, 2, 3, 5, 6, 8, 9]
  ```
  標籤是 Python list of int；路徑用  接， 得到 。

### FoodDataset 與 getbatch
```
../.venv/bin/python -c "
from dataset import FoodDataset, get_paths_labels
paths, labels = get_paths_labels('./food/')
train_set = FoodDataset(paths, labels, mode='eval')
images, labels = train_set.getbatch(range(10))
print(images.shape, images.dtype, labels.shape, labels.dtype)
print(images.min().item(), images.max().item())
x, y = train_set[3]
print(x.shape, y, type(y).__name__)
"
```
逐字輸出：
```
torch.Size([10, 3, 128, 128]) torch.float32 torch.Size([10]) torch.int64
0.0 1.0
torch.Size([3, 128, 128]) 2 int
```
-  會呼叫 ，回傳 (tensor, int)； 用  疊圖、 把 int list 變成 int64 tensor。
-  把 PIL 圖（0–255 的 uint8，H×W×C）轉成 0.0–1.0 的 float32，並換成 C×H×W。
- Resize(size=(128, 128)) 給的是 (高, 寬) 兩個數，所以**不保持長寬比**：
  | 檔名 | 原尺寸 W×H | 寬高比 | 水平縮放 | 垂直縮放 |
  |---|---|---|---|---|
  | 2_3.jpg（圖 3） | 512×341 | 1.501 | 0.250 | 0.375 |
  | 1_1.jpg（圖 1） | 849×565 | 1.503 | 0.151 | 0.227 |
  | 1_2.jpg（圖 2） | 1294×1300 | 0.995 | 0.099 | 0.098 |
  | 9_9.jpg（圖 9） | 512×384 | 1.333 | 0.250 | 0.333 |
  圖 1、3 的垂直方向被多壓 1.5 倍（看起來變矮胖）、圖 9 1.33 倍；其餘 6 張 512×512 只是等比縮小。
- 10 張都是 RGB，所以  不用  也沒事（灰階或 RGBA 圖會變成 1 或 4 通道，模型吃不進去）。
-  的 RandomHorizontalFlip、RandomRotation(15) 在本作業從來沒用到；變數名  與這 10 張圖可能來自訓練集有關，但 eval 轉換才是實際使用的。

### 模型對 10 張圖的預測
指令（需要 GPU）：
```
../.venv/bin/python -c "
import torch
from model import Classifier
from dataset import FoodDataset, get_paths_labels
model = Classifier().cuda()
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()
paths, labels = get_paths_labels('./food/')
images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
with torch.no_grad():
    logits = model(images.cuda()).cpu()
prob = logits.softmax(dim=1)
for i in range(10):
    y = labels[i].item()
    print(i, y, logits[i].argmax().item(), '%.3f' % logits[i, y].item(), '%.4f' % prob[i, y].item())
"
```
逐字輸出（欄位：圖編號、標籤、預測、標籤類別的 logit、softmax 機率）：
```
0 0 0 10.333 1.0000
1 1 1 19.785 1.0000
2 1 1 7.808 0.9986
3 2 2 9.011 1.0000
4 2 2 12.215 1.0000
5 3 3 12.385 1.0000
6 5 5 22.956 1.0000
7 6 6 11.447 1.0000
8 8 8 13.015 1.0000
9 9 9 14.118 1.0000
```
- 1 − p(標籤)：圖 0 2.50e-06、1 0（float32 下剛好等於 1）、2 1.38e-03、3 2.38e-07、4 1.19e-07、5 3.58e-07、6 0、7 6.08e-06、8 0、9 1.19e-07。圖 1、6、8 的機率在 float32 裡就是 1。
- 第一名和第二名的 logit 差：7.262（圖 2）到 37.930（圖 6）；圖 0 13.662、1 24.314、3 15.473、4 16.519、5 14.791、7 12.831、8 19.700、9 15.687。差 7 以上，softmax 機率就到 0.999 以上（e^−7 ≈ 0.0009）。
- **logit 的小數第 3 位會隨 batch 組成變**：同一張圖 0，10 張一起算是 10.3332，單獨算是 10.3312（差 8.3e-03，GPU 依 batch 大小選不同的卷積演算法）。CNN 實測「IG」表裡的 logit 是單張算的，所以和這裡的批次結果差在小數第 2–3 位（例如圖 3：9.024 vs 9.011）。同一個 batch 重算兩次則逐位元相同。
- **eval 與 train 模式**：同 10 張圖改 ，argmax 仍全對，但 logit 最多差 18.9（Dropout 隨機丟掉、BN 改用這一批的統計量）；而且 train 模式的 forward 會**改寫 BN 的 running_mean／running_var**（buffer 被這 10 張圖更新），之後切回 eval 結果也變了。所以解釋方法前一定要 ，且不要在 train 模式下 forward。

### normalize 與 save_fig（explain_cnn.py:24–32）
- ，把任何值域線性拉到 0–1，numpy 與 torch 都能用（只用到 、 和算術）。若整張圖是常數，分母為 0，會得到 NaN；本作業的資料沒有遇到。
-  用  裁掉多餘白邊，存完  釋放記憶體，再印 。
