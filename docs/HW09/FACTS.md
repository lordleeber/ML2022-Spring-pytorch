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
- `x.grad` 形狀 (10, 3, 128, 128)；`torch.max(..., dim=1)` 對 RGB 取最大 → saliency (10, 128, 128)。各圖取到最大值的通道比例：R 36.0%、G 41.7%、B 22.4%，沒有平手（ch03 審稿補測量的精確值）。
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
- ch01（第 1 章，2026-10-03 雲端）：
  - 圖號：圖 1.1（checkpoint.pth 170 MB 組成長條：model_state_dict 33.3% 綠、Adam 動量 66.6% 黃）、圖 1.2（food/ → os.listdir → my_key 排序 → paths/labels → FoodDataset → getbatch → (10,3,128,128) 的路線圖，藍）、圖 1.3（img/images.png）、圖 1.4（10 張圖第 1、2 名 logit 差長條，圖 2 用黃標出，虛線在差 7）。
  - 本章正式定義的名詞：state_dict（「名字 → 張量」對照表；key 是屬性路徑，如 `cnn.0.weight`）、checkpoint（含 epoch／model_state_dict／optimizer_state_dict 的容器）、`load_state_dict`（名字逐一對上；預設 strict）、Adam 的 `step`／`exp_avg`（一階動量）／`exp_avg_sq`（二階動量）與 param_groups、pickle（一句）、`weights_only`（2.6 起預設 True）、safetensors（一句）、`os.listdir` 不保證順序、`my_key`（類別×1,000,000+編號；tuple key 的替代寫法）、food-11（一句）、`Dataset`／`__len__`／`__getitem__`、`transforms.Compose`、資料增強（data augmentation）、Resize 給 (高, 寬) 不保持長寬比、ToTensor（HWC uint8 → CHW float32 0–1）、PIL／Pillow（一句）、`torch.stack`、`DataLoader`（一句）、`normalize`（整個陣列 min-max；常數陣列 → NaN）、`save_fig`（bbox_inches='tight'、plt.close）、`plt.subplots`、`imshow` 與 `permute(1, 2, 0)`、`torch.no_grad()`（只管梯度、不擋 BN buffer 更新）、1 − p(標籤)、第 1／2 名 logit 差（機率比 = e^差）、float32 在 1 附近的間距 2^−24、train 模式 forward 會改寫 BN buffer。
  - 本章用語：「10 張一起算」＝批次結果；引用 logit 時註明批次或單張。
  - 回指（不重講）：ch00 的模型總覽、逐層表、buffer、`model.eval()`、logits／softmax 定義、相對路徑（0.5 節）、資料來源（0.4 節）。
  - 本章手算（由 FACTS 推得，非另外量）：e^−7.262 ≈ 0.0007（= 圖 2 第 2 名機率）、e^−37.930 ≈ 3e-17、e^7 ≈ 1,100、1.19e-07 = 2×2^−24；只存 model_state_dict 約 56.7 MB。
  - **雲端發現的 FACTS 疑點**：(1)「ch01 實測 → FoodDataset」說圖 1、3「看起來被壓扁、變寬」：寬圖被擠成正方形，內容其實是**變窄、變高**，ch01 照後者寫。(2)「logit 的小數第 3 位會隨 batch 組成變」：10.3332 − 10.3312 = 0.0020，不是 8.3e-03；ch01 已拿掉差值並標 TODO。(3)「差 7 以上時 softmax 機率就超過 0.999」只看第 2 名；10 類一起算的下界約 0.993，ch01 沒有引用這句。

- ch02（第 2 章，2026-10-03 雲端）：
  - 圖號：圖 2.1（img/lime.png）、圖 2.2（LIME 流程 SVG：原圖 → ①切 superpixel → ②取樣 0/1 → ③換平均色 → ④模型 predict → ⑤加權 Ridge → ⑥每塊一個權重 → ⑦get_image_and_mask 上色；藍＝輸入、綠＝forward、紫＝LIME 步驟）、圖 2.3（img/ch02_segments.png）、圖 2.4（img/ch02_perturb.png）、圖 2.5（img/ch02_img0_compare.png）、圖 2.6（img/ch02_lime_softmax.png）、圖 2.7（差一錯誤 SVG：SLIC 塊 1..107 vs 特徵 0..106，紅＝對不上）。
  - 本章正式定義的名詞：LIME（Local Interpretable Model-agnostic Explanations，三個字各自的意思）、model-agnostic、代理模型（surrogate model）、superpixel（完整定義）、SLIC 與 `n_segments`／`compactness`／`sigma`／`start_label`、特徵（每塊一個）、樣本（0/1 向量，第 0 列＝原圖）、「遮住」＝換成該塊平均色（`hide_color=None`）、`explain_instance` 的預設值（num_samples、batch_size、top_labels、hide_color、distance_metric、kernel_width、random_state）、cosine 距離、核函數（kernel）與樣本權重（≠ 每塊的權重）、Ridge 回歸與 alpha、R²（含加權）、截距、`local_exp`／`intercept`／`exp.score`（只有一個值）／`local_pred`、`top_labels`、`get_image_and_mask` 五個參數與 mask、疊色、差一錯誤（off-by-one）、偽亂數／種子／NumPy 全域亂數產生器、`2>/dev/null`。
  - 本章手算（由本 FACTS 推得）：0/1 向量與全 1 向量的 cosine = √(k/n)；圖 0 平均保留 53.4 塊 → d ≈ 0.29、樣本權重 ≈ 0.50；最像原圖的樣本權重約為最不像的 3 倍（0.760／0.231）；−21.383 + 33.926 = 12.543 = local_pred；num_features=200 的 46 綠 = 47 正 − 特徵 0。
  - 推論（未實測，ch02 標了 TODO）：紅色疊在亮處（R 已接近最大值）看不出來，用來解釋圖 1、2 在 lime.png 上幾乎看不到紅色；`LimeImageExplainer(random_state=16)` 可讓每張圖的結果不依賴順序。
  - 回指（不重講）：ch00 的 logits／softmax、tqdm、逐段計時 14.7 s、0.6 節重跑逐位元相同；ch01 的預測表、批次 vs 單張 logit、model.eval()、HWC／permute、save_fig；index 的 lime 0.1.1.37 vs 0.2.0.1。

- ch03（第 3 章，2026-10-04 雲端）：
  - 圖號：圖 3.1（img/saliency.png）、圖 3.2（forward + backward 路徑 SVG：x → model(x) → logits → CE loss（綠）；loss.backward() → ∂L/∂z =(softmax − one-hot)÷10 → x.grad（黃）；W.grad 虛線框「也被寫入，但沒有人讀」；紫框＝取絕對值、RGB 取最大、各自 normalize）、圖 3.3（10 張圖梯度最大值的對數刻度長條圖，左欄是 1 − p）、圖 3.4（img/ch03_saliency_global.png）、圖 3.5（img/ch03_saliency_logit.png）、圖 3.6（img/smoothgrad.png）、圖 3.7（img/ch03_smoothgrad_variants.png）、圖 3.8（img/ch03_smoothgrad_n.png）。
  - 本章正式定義的名詞：backward、鏈鎖律、偏微分、梯度（gradient）、對輸入取梯度 vs 對權重取梯度、autograd、`requires_grad`／`requires_grad_()`（結尾底線＝就地修改）、`.grad`（會累加）、`model.zero_grad()`（兩個函式都沒呼叫）、saliency map（顯著圖）、CE loss（cross-entropy，內含 softmax、−log p_y、批次預設取平均 `reduction='mean'`）、one-hot、∂L/∂z_j = p_j − 1[j=y]、`torch.max(..., dim=1)` 回傳 (值, 位置)、colormap（色表）與 `plt.cm.hot`、二維陣列套色表 vs (H,W,3) 當 RGB、相關係數（Pearson）、SmoothGrad、常態（高斯）分佈、標準差、變異數、`Tensor.normal_(mean, std)`、`x.new_empty`、`unsqueeze(0)`、`torch.manual_seed`、`torch.randn`、`torch.bincount`、`amax(dim=...)`、`torch.autograd.grad`（現在的做法框）、Captum `Saliency`／`NoiseTunnel`（一句）。
  - 本章用語：「對 loss 取梯度」（程式）vs「對標籤 logit 取梯度」（投影片的 output category，第 5 章 IG 也是後者）；「各自正規化」vs「一起正規化」；「程式的標準差」(0.4/範圍)² vs「論文式」0.4×範圍。
  - 本章手算（由本 FACTS 推得，非另外量）：3.75e-04 ÷ 1.52e-17 ≈ 2.5e13；圖 6 ÷ 圖 2 ≈ 4e-14；0.256 ÷ 8.48e-07 ≈ 30 萬倍、÷ 8.565e-06 ≈ 3 萬倍；雜訊比例圖 6 40%、圖 0 ≈ 40%、圖 1 ≈ 47%；0–255 範圍時論文式 102、程式式 ≈ 2.5e-06；loss 29.6 → p ≈ e^−29.6 ≈ 10^−13；圖 7 一起正規化 ≈ 1.51e-06 ÷ 3.75e-04 ≈ 4.0e-03。
  - 原本的 5 個 TODO 已在「ch03 審稿補測」量好並填入；本機另加 3.12 節「雜訊小到模型認得出來時」與圖 3.9（img/ch03_smoothgrad_small.png）。仍屬推測、沒有查證的：PyTorch 內部 log-softmax 的算法；模型為什麼對 G 通道比較敏感（ch03 寫「本書沒有答案」）。
  - 回指（不重講）：ch00 的 logits／softmax、0.5 節逐段計時（Saliency 0.4 s、SmoothGrad 24.1 s）、0.6 節 smoothgrad.png 每次重跑不同；ch01 的 1 − p 表、batch vs 單張 logit、model.eval()、normalize／save_fig、torch.no_grad()；ch02 的偽亂數與種子。

- ch04（第 4 章，2026-10-04 雲端）：
  - 圖號：圖 4.1（img/filter_cnn6.png）、圖 4.2（img/filter_cnn23.png）、圖 4.3（forward hook 位置 SVG：輸入 x（藍）→ cnn[0]–[5] → cnn[6] Conv（紫粗框，hook）→ cnn[7] BN → cnn[8] ReLU → cnn[9]–[37] 再加 fc → logits（灰虛線「沒有人用」）；hook → layer_activations（全域，第 164 行）→ objective −[:, 0, :, :].sum()（紫）；黃色 backward 回到 x；Adam([x]) 只改 x）、圖 4.4（img/ch04_trajectory.png）、圖 4.5（img/ch04_xrange.png）、圖 4.6（normalize 數線 SVG：cnn[6] 圖 0 −11.07..12.05 → 0..1，[0,1] → 0.479–0.522）、圖 4.7（img/ch04_variants_cnn6.png）、圖 4.8（img/ch04_variants_cnn23.png）、圖 4.9（img/ch04_crop.png）。
  - 本章正式定義的名詞：Filter activation／Filter visualization（docstring 158–159 的兩步；本書的 Filter explanation 取自第 157 行標題）、receptive field（感受野；7×7／38×38 與「步距」手算規則）、hook／forward hook、`register_forward_hook`、hook(module, input, output) 與「回傳值取代輸出」、handle 與 `hook_handle.remove()`、Python `global`（區域變數 vs 模組層變數）、「計算記錄」（autograd 在 forward 記下的運算）、優化器（optimizer：`step()`、`zero_grad()`）、Adam（一階／二階動量、第 1 步 ≈ lr × g 的正負號、ε 預設 1e-8）、lr（learning rate）、gradient ascent／gradient descent、viridis（matplotlib 預設色表；二維陣列預設依自身最小最大值對應色表）、BN 推論時是遞增一次函數（γ>0；除以 √running_var）、白雜訊（0–1 均勻分佈）、`torch.rand`、clamp（`x.clamp_(0, 1)`）、`.squeeze()`、`create_feature_extractor`／`torch.use_deterministic_algorithms`（現在的做法框，一句，未實測）。
  - 本章用語：「第 k 步」＝第 k 次迴圈的 forward（第 k 次更新前）；「k 次更新後」另做一次 forward；「第一次執行」＝hw09_facts.py、產生 img/ 圖、outline 引用的 −11.87..12.32／9,108,426；「第二次執行」＝hw09_ch04_filter.py 的 −11.90..12.31／9,102,480。兩組並列、不混用。
  - 本章手算（由本 FACTS 推得）：BN 驗算 −53.050 → −6.949；receptive field 7、38；38/128 ≈ 30%；10×128×128×128×4 bytes ≈ 84 MB；|Δx| 最小 0.0999 = 0.1 × 9.06e-6/(9.06e-6+1e-8)；第 50→100 步每步約 0.12；cnn[6] 每張 filter 0 最大值 22.90–48.33。
  - 標明為推測的：filter 0 像邊緣偵測器；第 2 步變差是 Adam 第一步太大；重跑與 batch vs 單張的差異被 Adam 放大；cnn[6] 紋理尺度和 receptive field 相符；clamp 拿走「放大數值」的捷徑。
  - 原本的 4 個 TODO 已在「ch04 審稿補測」量好並填入；本機另加 4.9 節最後的「hook 掛在 BN 或 ReLU 之後會怎樣」與圖 4.10（img/ch04_hook_bn_relu.png）。
  - 回指（不重講）：ch00 的 Conv／filter／activation／BN／ReLU／MaxPool、(N,C,H,W)、stage、0.2 逐層表、0.5 逐段計時（2.3／2.9 s）、0.6 重跑差異；ch01 的 normalize／save_fig、permute、model.eval()、Adam 的 exp_avg／exp_avg_sq、batch vs 單張 logit；ch03 的 backward、autograd、requires_grad_()、.grad 累加、x.cuda() 複本、colormap／hot、torch.manual_seed。

- ch05（第 5 章，2026-10-04 雲端）：
  - 圖號：圖 5.1（img/integrated_gradients.png）、圖 5.2（img/ch05_alpha_images.png）、圖 5.3（路徑與左端點黎曼和 SVG：上排 11 個示意方塊 α 0.0–1.0、α=1.0 紅虛線「程式不算」、黃點＝每個 α 算一次梯度；下排圖 0 實測 Σ 梯度 × x 長條（α 0.0–0.9，寬 0.1，黃），紅圈 α=1 −0.523；紫框：10 根面積 19.126 vs 目標 17.911、程式到平均就停、論文再 × (x − baseline)、不乘 x 時總和 0.273）、圖 5.4（10 張圖數線 SVG：白線＝f(x) − f(0)、黃圈＝Σ 程式輸出、紫點＝Σ 程式輸出 × x）、圖 5.5（img/ch05_path.png）、圖 5.6（img/ch05_ig_variants.png）、圖 5.7（img/ch05_baselines.png）。
  - 本章正式定義的名詞：f（目標類別的 logit，f(x)、f(0)）、局部梯度與飽和（saturation；和 ch03 3.6 的 softmax 飽和區分：這裡是 logit 沿亮度飽和）、attribution（歸因，完整定義）、baseline（基準圖，完整定義；x′）、直線路徑 x′ + α(x − x′)、α、IG 的論文定義（(x_i − x′_i) × 路徑上梯度的平均）、completeness（含鏈鎖律推導）、黎曼和（左端點、中點）、路徑平均梯度（本書對程式輸出的稱呼）vs「論文的 IG」、葉節點（直接設 requires_grad 只能用在葉節點）、`backward(gradient=...)`（非純量輸出要傳權重向量 v，算 Σ v_j ∂z_j/∂x；one-hot ≡ `model_output[0, c].backward()`）、NumPy 廣播（從最後一維對齊）、`np.zeros` 預設 float64、`np.moveaxis(…, 0, -1)`（NumPy 版 permute(1, 2, 0)）、四種 baseline（black、gray 0.5、blur 31×31 平均模糊 replicate、noise）、Captum `IntegratedGradients` 的 `baselines`／`target`／`n_steps`／`return_convergence_delta`（現在的做法框，一句，未實測）。
  - 本章用語：「程式輸出」＝「路徑平均梯度」；「Σ 程式輸出 × x」＝論文的 IG（10 步左端點）；「左 10／中點 200」等是收斂表的欄名。f(x) 一律是單張算的 logit（照 ch01 1.6 的說法回指：圖 3 9.011 vs 9.024）。outline 的 15.704／18.141／17.851 稱「第一次執行」，收斂表（15.750 等）稱「本章另外的一次執行」，兩組不混用。
  - 本章手算（由本 FACTS 推得，非另外量）：圖 0 路徑表 10 個 Σ grad × x 的平均 = 19.126（= 收斂表左 10）；0 → (0 − min)/(max − min)：圖 0 0.672、圖 3 0.594；圖 2 少 42%、圖 8 少 27%；左 10 誤差 >10% 的有 7 張（2、3、4、5、6、8、9），偏高的是 0、1、4、9，誤差最小是圖 7（4.7%）；α 0.2、0.3 兩根占 10 根總和 191.262 的 64%；200 步最大誤差 左 圖 7 2.5%、中點 圖 2 1.8%。
  - 標明為推論的：α 0.2–0.4 那段貢獻最多（另由 10 點手算 64%）；10 步誤差大是因為峰附近只有 2 個點；乘 x 主要是整體縮放、暗像素被壓下去；blur 的歸因集中在細節；灰底深淺不同是因為各張 0 對到的位置不同；路徑起伏多和圖 2 的大誤差相符。
  - **雲端發現的 FACTS 疑點**：「ch05 實測 → 收斂」說「10 步時中點不一定比左端好（圖 2、4、9 中點 10 步反而更差）」——照同一張表算，中點 10 步比左 10 差的是**圖 0、4、7、9**；圖 2 中點（15.623，差 4.488）比左端（6.448，差 4.687）略好。ch05 照表寫「圖 0、4、7、9」。另：ch05_path.png 右圖在 α ≈ 0.15 附近有一段跌到約 −25（看圖得知，FACTS 沒記）。「ch05 實測」開頭說工具產生 5 張 ch05_*.png，實際是 4 張（path、alpha_images、ig_variants、baselines）。
  - 冷讀後補的：5.5 節加「本章另外量的數字從哪來」框（點名 `docs/tools/hw09_ch05_ig.py`、在 HW09/ 裡跑）；自我測驗第 1 題附照字面改第 265 行的示意碼（未執行，標 TODO）。
  - 回指（不重講）：ch00 0.1 題目不在 repo、0.5 IG 0.7 s、0.6 重跑極細微差異；ch01 1.4 getbatch、1.5 normalize／permute、1.6 預測表與批次 vs 單張、1.7 model.eval()；ch03 3.2 鏈鎖律、3.3 requires_grad_()／.grad 累加／zero_grad、3.6 logit 梯度與 softmax 飽和、3.7 unsqueeze 與 float64 累加器、3.10 彩色、Pearson、hot、torch.autograd.grad；ch04 第 203 行 hook 已拿掉、自我測驗第 2 題（權重被改）、4.7 batch 互不影響、白雜訊、requires_grad_(False)。

- ch06（第 6 章，2026-10-04 雲端）：
  - 圖號：圖 6.1（第 3 組問答 56 個 token 序列 SVG：位置 0–13、14/15/26/43/49/50/54/55，框上 token、框下位置與 token_type_ids；紅＝問題、綠＝文章、藍＝答案 cats(12)、灰＝[CLS]/[SEP]；Cats(43) 綠）、圖 6.2（13 個 hidden states SVG：input_ids → embedding 層 → hs[0]（灰虛線「程式跳過」，第 91 行 [1:]）→ 第 1…12 層 → hs[k] (1, 56, 768) → 紫框 PCA 768→2 → layerk.png；hs[12] → qa_outputs Linear(768, 2)）、圖 6.3（img/ch06_pca_variance.png）、圖 6.4（img/ch06_q3_layers.png）、圖 6.5（img/ch06_q1_layers.png）、圖 6.6（img/ch06_q2_layers.png）。
  - 本章正式定義的名詞：Topic II 三種方法（Attention Visualization／Embedding Visualization／Embedding analysis）、attention（直覺：權重和為 1 的加權混合）、attention head、feed-forward network（FFN，768→3072→768）、Transformer（一句：BERT 所屬的模型類別）、exBERT、Hugging Face Space（回指 index）、cased、`transformers` 套件（一句）、pooler（一句，回指 ch00 LOAD REPORT）、tokenizer、WordPiece 與 `##` 片段、token id、[CLS]（101，含問答的「無答案」用途）／[SEP]（102）／[PAD]（0）／[UNK]（100）、token_type_ids、attention_mask（一句）、`return_tensors='pt'`、`**inputs`、BertForQuestionAnswering 與 qa_outputs、起點／終點分數（start/end logit；答案分數 = 起點 + 終點）、SQuAD 2.0 的「無答案」、hidden_states（13 個元素；hidden states 名稱的意思）、embedding 層（查表 + 位置 + token_type，看不到上下文）、PCA（主成分、投影）、explained variance ratio、sklearn 的 `fit_transform`、PCA 的 random_state（近似算法的亂數）、cosine similarity（直覺一句，正式定義在第 7 章）、q-q／c-c／q-c／答案→問題排名（本章自訂的量）、步驟 1 與 3 的差別（本書的讀法）、t-SNE／UMAP、`output_attentions=True`、問答 pipeline（現在的做法框，一句，未實測）。
  - 本章用語：「第 1／2／3 組問答」（不寫 Q1–Q3）；「layer k」＝`hidden_states[k]`、layer 0＝embedding 層；「讀圖要注意的三件事」（6.7 warn 框：各層各自 fit、正負號任意、fit 含不畫的特殊 token）。
  - 本章手算／推得（由本 FACTS 推得，非另外量）：參數 107,721,218 ÷ 14,162,827 ≈ 7.6 倍；第 2 組 layer 12 y 範圍 −24.6 而畫出的點最低 −9.6 → 最低點是不畫的 [CLS] 或 [SEP]（哪一個標 TODO）；第 3 組句號 8 個。
  - 標明為推論的：第 3 組變異比例較高可能因為重複短句；layer 1 接近 embedding 層是因為一次 attention 混進的上下文還不多；layer 12 高 cosine 與高變異比例一致；步驟 1／3 的讀法。
  - TODO：第 2 組第 12 層 y = −24.6 的特殊 token 是哪一個；自我測驗第 6 題（第 103 行改成 `word.lower()`）照字面跑一次；量測工具 stdout 節錄（可選）。
  - **雲端發現的 FACTS 疑點**：「ch06 實測 → PCA 保留的變異比例」說「三條線在 layer 0–10 大致平（0.12–0.26），layer 11 起上升」——照同一張表，第 1、2 組從 layer 8 就逐步上升（第 1 組 0.150 → 0.169 → 0.188 → 0.223），只有第 3 組到 layer 10 持平。ch06 照表寫。
  - 前面頁面改動：index〈第二層〉exBERT 那句改成 2026-10-04 的現況（Space RUNNING、頁面載得到、互動功能沒試）；outline 預計 TODO 第 6 章那條改成已完成。
  - 回指（不重講）：ch00 0.1 題目不在 repo 與「第 N 組問答」約定、0.3 環境、0.5 主程式 125–138／stdout／12.3 s、LOAD REPORT、0.6 BERT 圖重跑逐位元相同與 img/ 複本；ch01 1.6 torch.no_grad()；index 的 Hugging Face、Space、〈第二層〉hw9_bert.zip 404。

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
- state_dict 的 key 名稱就是屬性路徑：`cnn.0.weight` 是 `model.cnn[0].weight`；`cnn.1.running_mean` 是 BN 的 buffer。
- 優化器是 **Adam**（lr 0.001、betas (0.9, 0.999)、eps 1e-8、沒有 weight decay）。`state` 有 48 個 entry，等於模型的 48 個參數張量（11 個 Conv × 2 + 11 個 BN × 2 + 2 個 Linear × 2）。每個 entry 有 `step`、`exp_avg`、`exp_avg_sq`（Adam 的一階、二階動量，形狀與參數相同，例如第一個是 (128, 3, 3, 3)）。
- **舊版 PyTorch 存檔的痕跡**：`param_groups` 只有 `lr, betas, eps, weight_decay, amsgrad, params` 六個 key（新版還有 maximize、foreach、capturable 等）；`step` 是 Python int `32395`（新版存成 tensor）；`state` 的 key 是很大的整數，例如 `140436588276144`，而不是新版的 0..47。這些不影響本作業，因為腳本只讀 `model_state_dict`。
- `step` 32395 = 155 × 209。**推論**（沒有其他資料佐證）：epoch 從 0 數，存檔時是第 209 個 epoch，每個 epoch 155 個 batch。HW3 訓練集 9,866 張、batch 64 時，正好是 ⌈9866/64⌉ = 155。
- **170 MB 的組成**（逐 tensor 加總 bytes）：
  - model_state_dict 56,671,876 bytes：參數 14,162,827 × 4 bytes（float32）= 56,651,308；BN buffer 的 running_mean／var 共 5,120 個 float32 = 20,480；11 個 int64 的 num_batches_tracked = 88。dtype 只有 float32 與 int64。
  - optimizer 的 exp_avg + exp_avg_sq：113,302,616 bytes = 2 × 56,651,308。
  - 兩者合計 169,974,492；檔案 170,002,879；差的 28,387 bytes 是存檔格式本身（pickle 結構、key 名稱等）。
  - 結論：檔案的 **2/3 是 Adam 的動量**，推論時完全用不到。
- `torch.load('checkpoint.pth')` 在 torch 2.11 沒有指定 `weights_only`，預設是 True（2.6 起）。這個檔案只含 dict、tensor、int、float、tuple，所以照樣載得進來，沒有警告。

### 10 張圖的檔名與順序
```
../.venv/bin/python -c "
import os
print(os.listdir('food'))
print(sorted(os.listdir('food')))
"
```
逐字輸出（第一行的順序取決於檔案系統，本機是 WSL2 ext4；換機器可能不同）：
```
['9_9.jpg', '1_1.jpg', '2_4.jpg', '8_8.jpg', '0_0.jpg', '2_3.jpg', '3_5.jpg', '1_2.jpg', '5_6.jpg', '6_7.jpg']
['0_0.jpg', '1_1.jpg', '1_2.jpg', '2_3.jpg', '2_4.jpg', '3_5.jpg', '5_6.jpg', '6_7.jpg', '8_8.jpg', '9_9.jpg']
```
- `os.listdir` 不保證順序，所以 `get_paths_labels` 一定要排序。
- 這 10 個檔名用普通字串排序（`sorted`）剛好也得到同一個順序，因為類別都是個位數。`my_key` 的用處在類別有兩位數時：
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
  字串排序把 `10_3` 排在 `1_2` 前面（逐字元比，'0' < '_'）；`my_key` 把「類別 × 1,000,000 + 編號」當數字比，前提是編號小於 1,000,000。
- `get_paths_labels('./food/')` 的回傳值（印出來）：
  ```
  ['./food/0_0.jpg', './food/1_1.jpg', './food/1_2.jpg', './food/2_3.jpg', './food/2_4.jpg', './food/3_5.jpg', './food/5_6.jpg', './food/6_7.jpg', './food/8_8.jpg', './food/9_9.jpg']
  [0, 1, 1, 2, 2, 3, 5, 6, 8, 9]
  ```
  標籤是 Python 的 int list；路徑用 `os.path.join` 接起來，`'./food/'` 加 `'0_0.jpg'` 得到 `./food/0_0.jpg`。

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
- `train_set[3]` 會呼叫 `__getitem__(3)`，回傳 (tensor, int)。`getbatch` 用 `torch.stack` 疊圖、用 `torch.tensor` 把 int list 變成 int64 tensor。
- `ToTensor` 把 PIL 圖（0–255 的 uint8，H×W×C）轉成 0.0–1.0 的 float32，並把維度換成 C×H×W。
- `Resize(size=(128, 128))` 給的是（高, 寬）兩個數，所以**不保持長寬比**：
  | 檔名 | 原尺寸 W×H | 寬高比 | 水平縮放 | 垂直縮放 |
  |---|---|---|---|---|
  | 2_3.jpg（圖 3） | 512×341 | 1.501 | 0.250 | 0.375 |
  | 1_1.jpg（圖 1） | 849×565 | 1.503 | 0.151 | 0.227 |
  | 1_2.jpg（圖 2） | 1294×1300 | 0.995 | 0.099 | 0.098 |
  | 9_9.jpg（圖 9） | 512×384 | 1.333 | 0.250 | 0.333 |
  圖 1、3 的水平方向縮得比垂直多 1.5 倍：寬圖被擠成正方形，圖裡的東西**變窄、變高**（ch01 審稿更正；原本誤寫成「壓扁、變寬」）。圖 9 是 1.33 倍；其餘 6 張 512×512 只是等比縮小。
- 10 張都是 RGB，所以 `Image.open` 不加 `.convert('RGB')` 也沒事（灰階或 RGBA 圖會變成 1 或 4 個通道，模型吃不進去）。
- `mode='train'` 的 RandomHorizontalFlip、RandomRotation(15) 在本作業從來沒用到；實際使用的是 eval 轉換。

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
- 1 − p(標籤)：圖 0 2.50e-06、圖 1 0、圖 2 1.38e-03、圖 3 2.38e-07、圖 4 1.19e-07、圖 5 3.58e-07、圖 6 0、圖 7 6.08e-06、圖 8 0、圖 9 1.19e-07。圖 1、6、8 的機率在 float32 裡剛好等於 1。
- 第一名與第二名的 logit 差：最小 7.262（圖 2），最大 37.930（圖 6）；圖 0 13.662、圖 1 24.314、圖 3 15.473、圖 4 16.519、圖 5 14.791、圖 7 12.831、圖 8 19.700、圖 9 15.687。差 7 時，第 2 名的機率約是第 1 名的 e^−7 ≈ 0.0009 倍；但 1 − p(標籤) 是其他 10 類的總和，所以只靠「差 7」推不出 p > 0.999（10 類都落後 7 時的下界約 0.991）。ch01 沒有引用原本那句。
- **logit 的小數第 3 位會隨 batch 組成變**：同一張圖 0 的標籤 logit，10 張一起算是 10.333248，單獨算是 10.331242，差 0.0020（ch01 審稿更正：原本寫的 8.3e-03 是 11 個 logit 中差最多的那一個，不是標籤 logit）。可能原因：GPU 依 batch 大小選不同的卷積演算法。CNN 實測「IG」表裡的 logit 是單張算的，所以和這裡的批次結果在小數第 2–3 位不同（例如圖 3：9.024 vs 9.011）。同一個 batch 重算兩次則逐位元相同。
- **eval 與 train 模式**：同樣 10 張圖改用 `model.train()`，argmax 仍然全對，但 logit 最多差 18.9（Dropout 隨機丟值、BN 改用這一批的統計量）。而且 train 模式的 forward 會**改寫 BN 的 running_mean／running_var**（buffer 被這 10 張圖更新），之後再切回 eval，結果也跟著變了。所以解釋方法前一定要 `model.eval()`，也不要在 train 模式下 forward。

### normalize 與 save_fig（explain_cnn.py:24–32）
- `normalize(x)` = (x − x.min()) / (x.max() − x.min())，把任何值域線性拉到 0–1；numpy 陣列和 torch tensor 都能用（只用到 `.min()`、`.max()` 和四則運算）。如果整張圖是常數，分母是 0，會得到 NaN；本作業的資料沒有遇到。
- `save_fig` 用 `bbox_inches='tight'` 裁掉多餘白邊，存完用 `plt.close(fig)` 釋放記憶體，再印出 `saved <路徑>`。

## ch01 審稿補測（2026-10-03，本機）
- **沒有 GPU 時連 torch.load 都會失敗**：checkpoint 裡所有張量的 device 是 `cuda:0`。`CUDA_VISIBLE_DEVICES="" ../.venv/bin/python -c "import torch; torch.load('checkpoint.pth')"` 的最後一行是 `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.`。加上 `map_location='cpu'` 後可以載入，device 變成 `cpu`。所以「ch01 實測」裡讀 checkpoint 的那段指令**需要 GPU**（交給雲端的 prompt 寫成不需要，是錯的）。
- 圖 0 batch vs single：標籤 logit 10.333248 vs 10.331242，差 0.002007；11 個 logit 的最大差 0.008293。10 張各自單獨算，每張 11 個 logit 的最大差：0.0083、0.0076、0.0199、0.0505、0.0205、0.0206、0.0219、0.0177、0.0239、0.0289。
- 圖 2 的前 3 名：Dairy product 0.9986、Dessert 0.0007、Vegetable/Fruit 0.0007（見「CNN 實測 → 預測」）。
- **自我測驗 1 實跑**（在 food/ 的複本裡多放 `10_10.jpg`）：用 my_key，它排在 index 10，labels `[0, 1, 1, 2, 2, 3, 5, 6, 8, 9, 10]`；改成 `imgnames.sort()`，它排在 index 1，labels `[0, 10, 1, 1, 2, 2, 3, 5, 6, 8, 9]`。
- 雲端的三個 FACTS 疑點都成立，已在上面各段更正。
- ch01 引用的行號（dataset.py 15–58、explain_cnn.py 42/45–46/72/80/95/110/121/139/154/169/216/219/220/230/280/281/289/293–305）全部核對正確。

## ch02 實測（2026-10-04，本機；指令都在 HW09/ 裡執行）
工具：`docs/tools/hw09_ch02_lime.py`（在 HW09/ 裡跑；印出下面的數字，並產生 4 張 `docs/HW09/img/ch02_*.png`）。

### LIME 0.2.0.1 內部怎麼運作（讀 lime_image.py／lime_base.py 原始碼確認）
- `LimeImageExplainer()` 預設 `kernel_width=0.25`，核函數 `sqrt(exp(-d² / 0.25²))`；`random_state=None` → 用 numpy 全域亂數。
- `explain_instance` 的流程：
  1. 先抽一個 `random_seed = randint(0, 1000)`，只在沒給 segmentation_fn 時才用（本作業有給，所以白抽，但會消耗一次亂數）。
  2. 切 superpixel：`segments = segmentation_fn(image)`。
  3. 做「遮住用的圖」：`hide_color=None` 時，每塊 superpixel 換成該塊的 RGB 平均色（`fudged_image`）。
  4. 取樣：`data = randint(0, 2, (num_samples, n))`，每列是 n 個 0/1，0 = 把那塊換成平均色；`data[0, :] = 1`（第 0 個樣本是原圖）。每湊滿 batch_size=10 張就呼叫一次 classifier_fn → 1000 個樣本呼叫 100 次。
  5. 每個樣本和原圖（全 1 向量）的 cosine 距離 d，換成權重 `sqrt(exp(-d²/0.0625))`。
  6. `top_labels=5`：取原圖輸出最大的 5 個類別，逐一用加權 Ridge 回歸（`alpha=1`）把「哪幾塊沒遮」（0/1 向量）擬合到「該類別的輸出」。特徵選擇 `'auto'` 在 num_features=100000 > 6 時用 `'highest_weights'`，結果是 n 個特徵全部保留。
  7. 回傳的 `local_exp[label]` 是 (特徵編號, 權重) 依 |權重| 由大到小排好的 list；`intercept[label]` 是截距。
- **`exp.score` 與 `exp.local_pred` 只有一個值**：迴圈對 5 個類別都會覆寫，最後留下的是迴圈最後一個，也就是原圖輸出最大的那一類。本作業 10 張都預測正確，所以 `exp.score` 就是標籤那一類的 R²（加權）。
- 特徵 z 對應 `segments == z`（z = 0..n−1）。這就是 `start_label=1` 差一號的來源（見下）。

### superpixel（SLIC）
- 指令（不需要 GPU）：
  ```
  ../.venv/bin/python -c "
  import numpy as np
  from skimage.segmentation import slic
  from dataset import FoodDataset, get_paths_labels
  paths, labels = get_paths_labels('./food/')
  images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
  for i, image in enumerate(images.permute(0, 2, 3, 1).numpy()):
      s = slic(image.astype(np.double), n_segments=200, compactness=1, sigma=1, start_label=1)
      print(i, len(np.unique(s)), s.min(), s.max())
  "
  ```
  逐字輸出（欄位：圖、塊數、最小編號、最大編號）：
  ```
  0 107 1 107
  1 141 1 141
  2 151 1 151
  3 101 1 101
  4 66 1 66
  5 120 1 120
  6 108 1 108
  7 94 1 94
  8 128 1 128
  9 109 1 109
  ```
- `n_segments=200` 只是目標，實際 66–151 塊。每塊的像素數（128×128 = 16,384 個像素）：圖 0 最小 43、中位數 127、最大 667；圖 4（鬆餅）最大的一塊有 2,985 個像素（中位數 139），只切出 66 塊；其他圖每塊最小 41–47、中位數 96–150。
- `start_label=0` 切出的是**同一個分割**，只是編號整體減 1（圖 0：0..106，逐像素比對 `s0 + 1 == s1` 為 True）。
- 圖：`img/ch02_segments.png`（10 張圖疊上黃色邊界，標題是塊數）。

### 取樣與模型輸出（圖 0，重播 explain_instance 的亂數；已核對和真正傳進 predict 的 1000 張圖完全相同）
- 1000 個樣本 × 107 個 0/1。除了第 0 個（原圖），每個樣本被遮的塊數最少 36、平均 53.6、最多 72 —— **每次大約遮一半**，不是遮一小塊。
- 特徵 0（沒有像素）在 478 個樣本裡是 0，但遮了等於沒遮；編號 107 那塊從來沒被遮過。
- 標籤（Bread）的 logit 在 1000 個樣本上：原圖 10.333，最小 −21.274，中位數 −4.030，最大 10.333。p(Bread) 中位數 0.0009；≥ 0.99 的只有 5.2%，≤ 0.01 的有 62.3%；仍然預測成 Bread 的只有 18.3%。
- 和原圖的 cosine 距離：最小 0.185、中位數 0.296、最大 0.428；核權重最小 0.231、中位數 0.496、最大 0.760（原圖本身 d=0、權重 1）。
- 圖：`img/ch02_perturb.png`（原圖；所有塊都換成平均色；樣本 1、2、3 —— 分別遮了 50、49、48 塊，Bread 的 logit 2.970、1.749、−2.365）。
- **10 張圖的 1000 個樣本**（照原程式的順序，同一個 seed 16 連續跑）：
  | 圖 | 原圖 logit | 樣本 logit 最小 | 中位數 | p(標籤) 中位數 | p ≥ 0.99 | p ≤ 0.01 | 仍預測正確 |
  |---|---|---|---|---|---|---|---|
  | 0 | 10.33 | −21.27 | −4.03 | 0.0009 | 5.2% | 62.3% | 18.3% |
  | 1 | 19.79 | 18.84 | 21.38 | 1.0000 | 100% | 0% | 100% |
  | 2 | 7.81 | 7.32 | 10.30 | 1.0000 | 100% | 0% | 100% |
  | 3 | 9.01 | 3.95 | 8.94 | 1.0000 | 99.3% | 0% | 100% |
  | 4 | 12.21 | −2.60 | 6.67 | 0.9997 | 62.4% | 16.1% | 75.5% |
  | 5 | 12.39 | 1.38 | 6.69 | 0.9999 | 95.5% | 0% | 99.8% |
  | 6 | 22.96 | −9.18 | 8.30 | 1.0000 | 82.3% | 5.8% | 88.8% |
  | 7 | 11.45 | −60.85 | −24.08 | 0.0000 | 0.2% | 96.2% | 1.4% |
  | 8 | 13.01 | −11.18 | 3.16 | 0.8570 | 43.2% | 34.3% | 55.5% |
  | 9 | 14.12 | −6.51 | 4.47 | 0.7799 | 31.5% | 15.6% | 59.2% |
  - 圖 1、2、3、5：遮掉一半，機率幾乎都還 ≥ 0.99（機率「頂住」了），但 logit 仍在變（圖 1：最小 18.84、中位數 21.38）。
  - 圖 0、7：遮掉一半就幾乎認不出來（圖 7 只剩 1.4% 預測正確）。
  - 圖 1、2 的樣本 logit 中位數（21.38、10.30）**高於原圖**（19.79、7.81）：換成平均色反而讓模型更確定。這和兩張 Dairy product 的 LIME 權重以負為主（CNN 實測 → LIME 表）方向一致：遮掉某些塊會讓分數上升，那些塊的權重就是負的。
- **第 1 章 1.8 節原本的預告「如果看的是機率，大部分的圖已經頂在 1.0000，遮掉一小塊幾乎看不出變化」不準確**（LIME 每次遮約一半；只有圖 1、2、3、5 的機率頂住）。本次已在 master 上把 ch01 那句改掉（見下面「ch01 修正」）。

### 圖 0 的解釋（logits，原程式）
- 可貼上的指令（需要 GPU；`2>/dev/null` 把 tqdm 進度條藏起來）：
  ```
  ../.venv/bin/python -c "
  import numpy as np, torch
  from skimage.segmentation import slic
  from lime import lime_image
  from model import Classifier
  from dataset import FoodDataset, get_paths_labels
  model = Classifier().cuda()
  model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
  model.eval()
  paths, labels = get_paths_labels('./food/')
  images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
  calls = []
  def predict(input):
      calls.append(input.shape)
      with torch.no_grad():
          return model(torch.FloatTensor(input).permute(0, 3, 1, 2).cuda()).cpu().numpy()
  def segmentation(input):
      return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)
  np.random.seed(16)
  x = images[0].permute(1, 2, 0).numpy().astype(np.double)
  exp = lime_image.LimeImageExplainer().explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
  print(len(calls), calls[0])
  print(exp.top_labels)
  print([(int(s), round(float(w), 3)) for s, w in exp.local_exp[0][:5]])
  print(round(exp.score, 4))
  w = dict(exp.local_exp[0])
  print(len(w), round(w[0], 3), 107 in w)
  " 2>/dev/null
  ```
  逐字輸出：
  ```
  100 (10, 128, 128, 3)
  [np.int64(0), np.int64(4), np.int64(3), np.int64(2), np.int64(5)]
  [(21, 5.522), (25, 3.59), (38, 3.125), (27, 3.023), (40, 2.944)]
  0.8423
  107 0.158 False
  ```
  - 第 1 行：predict 被呼叫 100 次，每次 10 張 128×128×3（numpy 的 HWC）。
  - 第 2 行：top_labels 是 Bread(0)、Fried food(4)、Egg(3)、Dessert(2)、Meat(5)，和 ch01 的第 2 名（Fried food）一致。`np.int64(...)` 是 numpy 2 印純量的樣子。
  - 第 5 行：107 個特徵（編號 0..106）；特徵 0 權重 0.158（沒有對應像素，純雜訊）；107 號不在裡面。
- 截距 −21.383、local_pred 12.543（線性模型對原圖的預測；實際 logit 10.333）；107 個權重加總 33.926。
- 本機逐字與 explain_cnn.py 內同一張圖的結果相同（同 seed、同順序的第一張）。

### get_image_and_mask 的參數（圖 0，logits 版）
- 預設（程式用的）：`positive_only=False, hide_rest=False, num_features=11, min_weight=0.05` → 11 塊綠、0 塊紅，3,077 個像素被上色。
- `positive_only=True`：mask 上一樣選出 11 塊，但回傳的圖**沒有上色**、和原圖相同（見「ch02 審稿補測」；原本寫「一樣 11 塊綠」是錯的）。
- `num_features=5`：5 塊綠，1,386 個像素。
- `min_weight=0.0`：一樣 11 塊綠（前 11 名都遠大於 0.05）。
- `num_features=200`（等於全部）：46 塊綠、36 塊紅，綠 8,476、紅 5,074 個像素（其餘 |w| < 0.05 的不畫）。
- 上色方式：正權重那塊的 G 通道設成 `np.max(image)`（圖 0 是 0.9961），負權重的 R 通道設成最大值；其他兩個通道保留原圖 → 綠／紅是「疊色」，不是蓋掉。
- 注意：positive_only=False 時是先取 |權重| 前 num_features 名，再丟掉 |w| < min_weight 的；所以「畫幾塊」由 num_features 和 min_weight 一起決定。

### logits vs 機率、start_label=0（圖 0）
- softmax 版：截距 −0.6451、R² 0.5409；前 5 名 (21, 0.2186)、(40, 0.1977)、(25, 0.1905)、(27, 0.1445)、(23, 0.1274)；|w| ≥ 0.05 有 13 塊。
- start_label=0 版：R² 0.8617；前 5 名 (20, 5.2675)、(24, 3.4096)、(37, 3.3193)、(39, 3.157)、(26, 3.1135)。
- 圖：`img/ch02_img0_compare.png`（三格：logits（repo 原樣）／softmax 機率／logits + start_label=0）。看得到的差別：logits 版與 start_label=0 版的綠色區域幾乎一樣（披薩上半部到右上的餅皮）；softmax 版綠色區域相近，但在披薩尖端附近多了一塊**紅色**。
- **10 張都改用 softmax**（同 seed 16 連續跑；`img/ch02_lime_softmax.png`）：
  | 圖 | R² | |w| ≥ 0.05（正/負） | 畫出的綠／紅塊 |
  |---|---|---|---|
  | 0 | 0.541 | 13（12/1） | 10／1 |
  | 1 | 1.000 | 0 | 0／0 |
  | 2 | 0.563 | 0 | 0／0 |
  | 3 | 0.099 | 0 | 0／0 |
  | 4 | 0.566 | 12（12/0） | 11／0 |
  | 5 | 0.180 | 0 | 0／0 |
  | 6 | 0.396 | 9（9/0） | 9／0 |
  | 7 | 0.267 | 0 | 0／0 |
  | 8 | 0.840 | 8（8/0） | 8／0 |
  | 9 | 0.760 | 16（11/5） | 9／2 |
  - 圖 1、2、3、5、7 共 5 張完全沒有上色：每一塊的 |權重| 都小於 0.05。
  - 圖 1、2、3、5：機率在樣本上幾乎不動（上表 p ≥ 0.99 的比例 95–100%），所以沒有一塊的權重夠大。圖 1 的 R² 1.000 是因為要擬合的值幾乎是常數。
  - 圖 7：機率幾乎都是 0（p ≤ 0.01 佔 96.2%），同樣沒什麼可解釋。
  - 對照原程式（logits）的 lime.png：10 張都有上色（CNN 實測 → LIME 表，畫出 11 塊或接近 11 塊）。

### 亂數的影響
- 圖 3 單獨跑（seed 16、第一個跑）前 5 名 [40, 57, 86, 52, 39]；照原程式排在第 4 個跑：[40, 57, 86, 52, 48]。前 11 名重疊 10 塊。
- 圖 0 改用 seed 0：前 5 名 (21, 5.158)、(25, 3.715)、(38, 3.255)、(40, 3.162)、(27, 3.055)，R² 0.8453；和 seed 16 的前 11 名重疊 10 塊。
- 結論：換 seed 或換順序，最重要的幾塊大致不變，但排名後段會換人；要和投影片／書上的圖逐塊對照，必須照原程式的 seed 與順序。

### ch01 修正（本次一起改，已在 master）
- ch01 1.8 節的 LIME 預告改成：LIME 每次會把大約一半的 superpixel 換成平均色；改看機率的話，有幾張圖（圖 1、2、3、5）就算遮掉一半，機率也幾乎都還在 0.99 以上，分不出哪一塊重要；logits 則仍會隨遮掉的部分變動。

## ch02 審稿補測（2026-10-04，本機）
- **更正「ch02 實測 → get_image_and_mask」的 `positive_only=True` 那一行**：mask 上一樣選出 11 塊（3,077 個像素），但**回傳的圖和原圖逐像素完全相同，沒有任何上色**。lime 0.2.0.1 的 positive_only 分支只做 `temp[segments == f] = image[segments == f].copy()`，沒有把 G 通道設成最大值。搭配 `hide_rest=True` 時，回傳的圖只剩這 3,077 個非零像素，其餘是 0（黑）。原本寫的「一樣 11 塊綠」只數了 mask，是錯的。
- **紅色疊在亮處看不出來（實測）**：圖 1、圖 2 的 `np.max(image)` 都是 1.0000。
  - 圖 1：紅 1,227 px，原圖平均 R 0.985、G 0.925、B 0.767；95% 的紅像素原本 R ≥ 0.9。整張圖平均 R 0.994。7 塊紅色幾乎都在白色背景。綠 828 px，原圖 G 平均 0.857。
  - 圖 2：紅 1,427 px，原圖平均 R 0.813、G 0.753、B 0.702；44% 的紅像素原本 R ≥ 0.9。疊色後是淡粉紅，放大才看得出來。綠 243 px。
  - 10 張的 mask 像素數（綠／紅）：0：3077／0；1：828／1227；2：243／1427；3：1663／667；4：7025／156；5：1906／350；6：4402／0；7：3856／0；8：2941／0；9：2119／765。
- **`LimeImageExplainer(random_state=16)`（實測）**：不呼叫 np.random.seed，每張圖新建 `LimeImageExplainer(random_state=16)`。圖 3 單獨跑與在迴圈中第 4 個跑，權重完全相同；前 5 名 [40, 57, 86, 52, 39]，和「np.random.seed(16) 後單獨第一個跑」相同，因為兩者都是從 seed 16 的起點開始取數。
- 雲端手算的幾個數都核對過：cos(0/1 向量, 全 1) = √(k/n)；−21.383 + 33.926 = 12.543；num_features=200 的 46 綠 = 47 個正權重扣掉特徵 0。

## ch03 實測（2026-10-04，本機；指令都在 HW09/ 裡執行）
工具：`docs/tools/hw09_ch03_grad.py`（在 HW09/ 裡跑；印出下面的數字，並產生 4 張 `docs/HW09/img/ch03_*.png`）。所有 SmoothGrad 變體都先 `torch.manual_seed(0)`，和 explain_cnn.py（沒設 torch seed）的 smoothgrad.png 不是同一批雜訊。

### Saliency：梯度為什麼這麼小
- 可貼上的指令（需要 GPU）：
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
  x = images.cuda().requires_grad_()
  loss = torch.nn.CrossEntropyLoss()(model(x), labels.cuda())
  loss.backward()
  print('loss %.6f' % loss.item())
  for i, s in enumerate(x.grad.abs().amax(dim=(1, 2, 3))):
      print(i, '%.3e' % s.item())
  "
  ```
  逐字輸出：
  ```
  loss 0.000139
  0 8.480e-07
  1 3.217e-12
  2 3.745e-04
  3 1.370e-07
  4 4.239e-08
  5 1.271e-07
  6 1.518e-17
  7 1.505e-06
  8 1.076e-09
  9 8.522e-08
  ```
  （「CNN 實測 → Saliency」表裡的「最大值」就是這一欄；那裡是先對 RGB 取 max 再取整張的 max，數值相同。）
- 每張圖的 CE loss 剛好等於 1 − p(標籤)（p 接近 1 時 −log p ≈ 1 − p）：圖 0 2.50e-06、圖 2 1.38e-03、圖 7 6.08e-06；圖 1、6、8 在 float32 下是 0（p 存成 1）。
- 梯度大小的排序大致跟著 1 − p 走：圖 2（1.38e-03）最大 3.7e-04；圖 7（6.1e-06）1.5e-06；圖 0（2.5e-06）8.5e-07。1 − p 是 0 的三張也有梯度，但極小（圖 6 1.5e-17、圖 1 3.2e-12、圖 8 1.1e-09）：float32 的 p 存成 1，梯度卻不是 0。**推測**（沒有另外查證）：CrossEntropyLoss 內部用 log_softmax 計算，其他類別的 p_j 雖然小到不影響 p_y 的顯示，仍是非零的極小值。
- batch 的 loss 是 10 張的**平均**，所以每張圖拿到的梯度是「單獨算」的 1/10：圖 0 單獨算 max 8.565e-06，在 batch 裡 8.480e-07。各自 normalize 後這個係數消失。
- **Saliency 表的兩欄**（ch01 已給 1 − p；本章給梯度）可並排成「越有把握 → 梯度越小」。

### 為什麼要逐張 normalize（圖 `img/ch03_saliency_global.png`）
- 三列：原圖；repo 的做法（逐張 min-max）；改成 10 張一起做一次 min-max。
- 一起做時各張的最大值：圖 2 = 1.00（定義上最大）；圖 7 4.02e-03、圖 0 2.26e-03、圖 3 3.66e-04、圖 5 3.39e-04、圖 9 2.28e-04、圖 4 1.13e-04、圖 8 2.87e-06、圖 1 8.59e-09、圖 6 4.05e-14。
- 看得到的：第三列只有圖 2 看得到紅黃色的熱點（牛奶壺左側的輪廓），其他 9 張全黑。第二列 10 張都有熱點。
- 這就是 ch01 1.8 節預告的「最沒把握的圖 2 為什麼是例外」：只有它的梯度大到在共同尺度上看得見。

### 投影片說「output category 的梯度」（圖 `img/ch03_saliency_logit.png`）
- 改成對**標籤 logit** 取梯度（`model(x).gather(1, labels).sum().backward()`），每張的最大值變成 1.25–4.90（圖 0 2.044、圖 1 1.402、圖 2 1.743、圖 3 2.403、圖 4 1.683、圖 5 1.251、圖 6 3.029、圖 7 1.685、圖 8 2.333、圖 9 4.896）—— 不再有 1e-17 這種數量級的差異。
- 兩種熱圖逐張 normalize 後的像素相關係數：圖 0 0.862、圖 1 0.637、圖 2 0.969、圖 3 0.931、圖 4 0.820、圖 5 0.880、圖 6 0.726、圖 7 0.974、圖 8 0.828、圖 9 0.924。形狀大多相似，圖 1（0.637）和圖 6（0.726）差最多。
- 看得到的：第三列（logit 梯度）整體比第二列（loss 梯度）亮、熱點更分散；圖 5 的荷包蛋輪廓（圓形）在 logit 版更清楚。
- 理由（公式）：∂L/∂x = Σ_j (p_j − 1[j=y]) ∂z_j/∂x；p 接近 one-hot 時，係數 (p_y − 1) 與其他 p_j 都接近 0，但熱圖的「形狀」主要由最大的那個係數決定，所以兩者形狀相近、數量級完全不同。

### SmoothGrad 的雜訊有多大
- 程式的 std 每張 0.160–0.178（「CNN 實測 → SmoothGrad」表）；論文式 0.4 × 範圍是 0.38–0.40。
- **這兩種雜訊下，模型全部認錯**：可貼上的指令（需要 GPU）：
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
  torch.manual_seed(0)
  x = images[0]
  std = (0.4 / (x.max() - x.min()).item()) ** 2
  noisy = x + torch.randn(500, 3, 128, 128) * std
  with torch.no_grad():
      pred = model(noisy.cuda()).argmax(dim=1).cpu()
  print('std %.4f' % std)
  print(torch.bincount(pred, minlength=11).tolist())
  "
  ```
  逐字輸出：
  ```
  std 0.1613
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500]
  ```
  500 個加雜訊的圖 0 **全部被判成類別 10（Vegetable/Fruit）**。
- 10 張圖各 500 個樣本（repo 的 std 與論文式 std 都一樣）：**每一張的 500 個樣本都被判成 Vegetable/Fruit**，標籤的平均 logit 是負的（repo std：圖 0 −16.60、1 −7.92、2 −10.50、3 −4.81、4 −3.65、5 −9.91、6 −11.32、7 −16.83、8 −8.84、9 −10.95）。（註：這裡的雜訊用 `torch.randn(500, …) * std` 一次產生，和 smooth_grad 逐次 `normal_` 不是同一批，但分佈相同。）
- 單一樣本（圖 0，`torch.manual_seed(0)`）：std 0 → p 1.0000、loss 2.5e-06、梯度 max 8.6e-06；std 0.01 → p 1.0000、梯度 1.1e-05；std 0.05 → p 0.6467、loss 0.436、梯度 1.16；std 0.1613（repo）→ p 0.0000、loss 29.6、梯度 1.03；std 0.3984（論文式）→ p 0.0000、loss 27.7、梯度 0.38。
- 模型對高斯雜訊的耐受度（每張 100 個樣本，10 張平均的正確率）：std 0.005 → 1.00；0.01 → 1.00；0.02 → 0.87；0.03 → 0.54；0.05 → 0.43；0.08 → 0.11；0.1 → 0.01。
- **意義**：SmoothGrad 原本的想法是「在原圖附近取樣、把梯度的雜訊平均掉」。在這個模型上，std 0.16 的雜訊已經把每個樣本推到「模型認為是 Vegetable/Fruit」的區域，所以平均的是 500 個**被認錯的圖**上、CE loss（約 30）的梯度。ch00/FACTS 原本說的「加雜訊後梯度大了好幾個數量級」原因就在這裡：不是雜訊讓模型「不那麼確定」，而是讓它**完全認錯**。教材應如實寫，並避免宣稱 SmoothGrad 圖顯示「模型判斷 Bread 的依據」。

### SmoothGrad 的變體（圖 `img/ch03_smoothgrad_variants.png`）
- 四列：原圖；repo（std (0.4/範圍)²，normalize）；不 normalize（程式註解要你試的 `smooth = smooth / epoch`）；論文式 std 0.4 × 範圍。
- **不 normalize** 時每張的值域（500 次平均）：圖 0 2.94e-03..0.259、1 8.36e-04..0.098、2 1.25e-03..0.496、3 1.48e-03..0.162、4 1.40e-03..0.138、5 1.91e-03..0.167、6 1.30e-03..0.176、7 2.75e-03..0.354、8 1.38e-03..0.129、9 1.85e-03..0.208。全部 < 1，沒有被截斷，只是**很暗**（最大值只有 0.1–0.5）。看得到的：第三列幾乎全黑，只有隱約的輪廓；圖 2 的草莓比較亮。
- 和 saliency 不同：不 normalize 時 10 張的亮度差不多（最大值同一個數量級），因為加了雜訊後每張的梯度都在 0.1–1 的量級，不再有 1e-17 的差距。
- 論文式 std（第四列）：整體比 repo 版亮、輪廓更模糊（雜訊大一倍多）。圖 2 兩種版本都只有草莓那一小塊是亮的。
- 看得到的（repo 版，第二列）：熱圖保留了食物的輪廓與紋理（鬆餅的格子、荷包蛋的圓、生魚片的條紋），彩色，因為 3 個通道各自保留。

### SmoothGrad 取樣次數（圖 `img/ch03_smoothgrad_n.png`，圖 0）
- 1、10、50、500 次取樣的熱圖，和 500 次的平均絕對差：1 次 0.1563、10 次 0.0585、50 次 0.0257。
- 看得到的：1 次是一片彩色雜點，看不出形狀；10 次開始看到披薩三角形與刀叉的輪廓；50 次和 500 次差不多，500 次最平滑。

### 其他程式細節（實測）
- `compute_saliency_maps` 和 `smooth_grad` 都沒有呼叫 `model.zero_grad()`，所以**模型參數的 .grad 會一直累加**：連續呼叫兩次 compute_saliency_maps，`fc[3].weight.grad` 的絕對值總和從 2.150e-02 變成 4.299e-02（剛好 2 倍）。這不影響熱圖（熱圖用的是輸入 x 的 .grad，每次都是新的 tensor），只是多佔記憶體。
- 執行時間：本次 hw09_ch03_grad.py 裡每個 SmoothGrad 變體 10 張約 52–58 秒，比 explain_cnn.py 內量到的 24.1 秒慢（同一個 process 裡先跑了其他計算，GPU 狀態不同）。教材的耗時請引用 ch00 的逐段計時（24.1 秒），不要引用這裡的數字。

## ch03 審稿補測（2026-10-04，本機；PR #13）
工具：`docs/tools/hw09_ch03_review.py`（在 HW09/ 裡跑；產生 `docs/HW09/img/ch03_smoothgrad_small.png`）。填掉 ch03 的 5 個 TODO。

- **saliency 的最大值落在哪個通道**（10 張一起 backward，和 compute_saliency_maps 相同）：10 張共 163,840 個像素，R 36.0%、G 41.7%、B 22.4%，**沒有平手的像素**，也沒有梯度為 0 的像素。逐張都是 G 最多、B 最少（G 0.384–0.459、B 0.179–0.285）。「CNN 實測 → Saliency」原本寫的「大約 R 35%、G 41%、B 22%」加起來不到 100%，以這裡為準。
- **圖 1、6、8 的 p(標籤) 存成 1，其他 p_j 不是 0**（批次 logits 的 softmax，float32）：p(標籤) == 1.0 為 True；其他 10 類加總：圖 1 2.759e-11、圖 6 3.381e-17、圖 8 4.913e-09；最大的單一 p_j：圖 1 2.757e-11、圖 6 3.367e-17、圖 8 2.782e-09。排序和梯度最大值（圖 1 3.2e-12、圖 6 1.5e-17、圖 8 1.1e-09）相同。支持 ch03 的推測；PyTorch 內部算法沒有讀原始碼查證。
- **雜訊樣本上 Vegetable/Fruit 拿走幾乎全部機率**（程式的 std、每張 500 個樣本、每張前 `torch.manual_seed(0)`、`torch.randn(500, …) * std`）：p(Vegetable/Fruit) 平均：圖 0、3–9 都是 1.0000，圖 1 0.9996、圖 2 0.9990；最低的單一樣本：圖 2 0.9693、圖 1 0.9974，其他 ≥ 0.9998。標籤的平均機率：圖 0 1.09e-11、1 2.11e-06、2 3.06e-07、3 2.06e-06、4 2.10e-06、5 1.10e-08、6 3.91e-10、7 9.18e-12、8 9.91e-09、9 3.15e-09。
- **SmoothGrad 正規化後的通道平均**（程式原樣、`torch.manual_seed(0)`、500 次；和 smoothgrad.png 不是同一批雜訊，smoothgrad.png 的數值沒存）：10 張合計 R 0.243、G 0.259、B 0.190；逐張都是 G ≥ R > B（圖 1 R 0.338 ≈ G 0.336 例外，R 略大）。整張最大值落在 G 的有 7 張、R 3 張（圖 0、2、8）。→ 三通道接近（灰）、G 略多 B 略少（偏綠），平均只有 0.1–0.34（暗）。
  - 更正：圖 1 是 R 0.338、G 0.336，所以「每一張都是 G 最高」不完全對；ch03 已照此寫成「除了圖 1（R、G 幾乎相同）」。
- **雜訊 std 0.01 的 SmoothGrad**（500 次，`torch.manual_seed(0)`，其餘同程式）：通道平均 10 張合計 R 0.044、G 0.050、B 0.030，比程式原樣暗很多。看得到的（ch03_smoothgrad_small.png 第四列）：很暗、帶紅綠藍細碎雜點；圖 1 奶油塊方形輪廓偏綠較亮、圖 2 牛奶壺弧線、圖 5 荷包蛋圓圈可辨，形狀接近第二列 saliency。
  - 和 saliency map（repo）的像素相關係數（SmoothGrad 先對 RGB 取 max、各自 normalize）：std 0.01：0.937、0.446、0.932、0.736、0.959、0.925、0.953、0.939、0.857、0.909（圖 0–9）；程式原樣：0.176、0.363、0.377、0.357、0.502、0.485、0.379、0.300、0.429、0.298。
- 圖 0 單一樣本（接續 hw09_ch03_grad.py 的同樣抽法）：std 0 → p 0.999997、loss 2.503e-06、梯度 max 8.565e-06；std 0.01 → p 0.999996、loss **3.576e-06**、梯度 max 1.095e-05（補上 ch03 表裡原本缺的 loss）。
- 圖：`img/ch03_smoothgrad_small.png`，四列：image、saliency (repo)、repo: std (0.4/range)^2、std 0.01。ch03 用作圖 3.9（新增 3.12 節）。

## ch04 實測（2026-10-04，本機；指令都在 HW09/ 裡執行）
工具：`docs/tools/hw09_ch04_filter.py`（在 HW09/ 裡跑；印出下面的數字，並產生 5 張 `docs/HW09/img/ch04_*.png`）。工具裡的 `ascend()` 是 filter_explanation 優化迴圈的複製（同樣 Adam、lr 0.1、100 步、目標 −filter 0 總和），另外加了「限制在 [0,1]」與記錄每一步的選項；hook 用區域 dict，不用全域變數。

### 行號（以本檔為準）
- explain_cnn.py：docstring **157–162**；`layer_activations = None` 164；`filter_explanation` 165–205（hook 定義 171–173、註冊 175、第一次 forward 180、取 activation 183、`x = x.cuda()` 186、`requires_grad_` 187、Adam 189、迴圈 191–199、取結果 200、`hook_handle.remove()` 203）；`filter_explain` 208–220（呼叫參數 209、三列畫圖 211–219、存檔 220）；主程式呼叫 310–311。
- outline 的「explain_cnn.py:157-220」正確（含 docstring）；FACTS「CNN 實測」標題的 164–221 和 outline 對照表的 164-220 是不含 docstring 的寫法。
- model.py：`stack_blocks(3, 128, 3)` 在第 27 行、`stack_blocks(128, 256, 3)` 第 29 行。cnn 索引：stage 1 是 0–9（Conv 0、3、6；BN 1、4、7；ReLU 2、5、8；MaxPool 9），stage 2 是 10–19，stage 3 是 20–29（Conv 20、23、26）。印出的層：cnn[6] `Conv2d(128, 128, 3×3, stride 1, padding 1)`、cnn[7] `BatchNorm2d(128)`、cnn[8] `ReLU()`、cnn[23] `Conv2d(256, 256, …)`、cnn[24] `BatchNorm2d(256)`、cnn[25] `ReLU()`。

### 投影片 p.10（用 pypdf 抽出的文字，逐字）
```
Filter Visualization
Question 14 to 17
● Use Gradient Ascent method to ﬁnd the image that activates the selected 
ﬁlter the most and plot them (start from white noise).
Ref: https://reurl.cc/mGZNbA
```
（只有這幾行文字；短網址指到哪裡沒有查。）

### hook 抓到的是什麼（10 張原圖，`torch.no_grad()` 下一次 forward）
- filter 0 的 activation，Conv 本身 vs 接在後面的 BN vs ReLU：
  | 層 | 形狀 | min | max | mean | <0 的比例 | =0 的比例 |
  |---|---|---|---|---|---|---|
  | cnn[6] Conv2d | (10,128,128,128) | −53.050 | 48.331 | −2.172 | 0.693 | 0 |
  | cnn[7] BN | 同上 | −6.949 | 5.848 | −0.527 | 0.778 | 0 |
  | cnn[8] ReLU | 同上 | 0 | 5.848 | 0.182 | 0 | 0.778 |
  | cnn[23] Conv2d | (10,256,32,32) | −175.011 | 33.615 | −30.689 | 0.911 | 0 |
  | cnn[24] BN | 同上 | −3.823 | 2.277 | 0.397 | 0.286 | 0 |
  | cnn[25] ReLU | 同上 | 0 | 2.277 | 0.592 | 0 | 0.286 |
- 每張圖 filter 0 的 min/max：cnn[6] 圖 0 (−49.31, 40.72)、1 (−22.24, 22.90)、2 (−49.11, 40.17)、3 (−47.81, 38.77)、4 (−40.70, 40.56)、5 (−52.29, 48.33)、6 (−49.59, 46.76)、7 (−45.92, 45.04)、8 (−39.54, 28.80)、9 (−53.05, 44.43)；cnn[23] 圖 0 (−123.86, 22.00)、1 (−53.21, 18.53)、2 (−161.82, 13.30)、3 (−126.82, 29.92)、4 (−128.22, 18.31)、5 (−118.58, 17.28)、6 (−142.34, 33.62)、7 (−175.01, 21.86)、8 (−125.20, 26.57)、9 (−159.03, 20.19)。每張總和和「CNN 實測 → Filter explanation」相同。
- 權重形狀：cnn[6].weight (128, 128, 3, 3)，filter 0 是 (128, 3, 3)，bias[0] −0.0057；cnn[23].weight (256, 256, 3, 3)，bias[0] 0.0370。
- 通道 0 的 BN 參數（checkpoint）：cnn[7] γ 1.2184、β −0.5347、running_mean −2.2347、running_var 93.1679；cnn[24] γ 0.9297、β 0.0824、running_mean −41.4389、running_var 1011.2743。兩個 γ 都 > 0，所以 eval 模式下 BN 對這個通道是遞增的一次函數：最大化 Conv 輸出的總和，等同最大化 BN 輸出的總和；但**不等同**最大化 ReLU 之後的總和（ReLU 會把負的截成 0）。（由公式推得，沒有另外跑「對 BN／ReLU 輸出做 ascent」的對照。）
- 「第二列 filter activation」：`filter_explain` 第 216 行 `imshow(normalize(img))`，img 是 2 維 (H, W)，**沒有指定 cmap**，用 matplotlib 預設色表 viridis（深紫 → 藍綠 → 黃）。cnn[6] 是 128×128、cnn[23] 是 32×32（畫成同樣大小，所以看起來是一格一格的）。
- 看得到的（filter_cnn6.png 第二列）：藍綠色底，物體邊緣像浮雕一樣，一側亮（黃綠）一側暗（深藍），例如奶油塊的稜線、鬆餅的格子、荷包蛋的圓圈、叉子。**推測**（沒有驗證）：filter 0 對某個方向的亮度變化有反應，像邊緣偵測器。
- 看得到的（filter_cnn23.png 第二列）：32×32 的方格，黃綠色為主，物體輪廓處有深藍的線條或區塊（披薩三角形的邊、奶油塊的方框、荷包蛋的同心圓、湯碗的圓）。

### hook 的細節（實測）
- `filter_explanation` 結束後，cnn[6] 上的 forward hook 數 `len(model.cnn[6]._forward_hooks)` 是 **0**（第 203 行移除了）。不 remove 就連續註冊兩次：hook 數是 2（每次 forward 兩個都會被呼叫）；remove 之後回到 0。
- 函式結束後全域變數 `layer_activations` **仍然留著最後一次 forward 的輸出**：形狀 (10, 128, 128, 128)、`requires_grad=True`、在 cuda:0 上（10×128×128×128 個 float32 ≈ 84 MB，連同它的計算圖一起，直到下一次被覆蓋）。
- 第 180 行 `model(x.cuda())` 沒有包 `torch.no_grad()`，所以這次 forward 也建了計算圖（只是沒人 backward）。第 183 行 `.detach().cpu()` 把結果切開搬回 CPU。
- `x.cuda()`：x 在 CPU 時每次回傳新張量（`images.cuda() is not images.cuda()` 為 True）；x 已經在 GPU 時回傳**同一個物件**（True）。所以第 186 行之後優化的是 GPU 上的新複本，主程式的 `images` 不會被改；第二次呼叫（cnnid=23）拿到的仍是原圖。

### 優化的軌跡（repo 的設定：從原圖、lr 0.1、100 步；圖 `img/ch04_trajectory.png`、`img/ch04_xrange.png`）
- 每一步 forward 時（step k 是第 k 次迴圈、更新前）的 filter 0 總和（10 張合計）與 x 的值域：
  | step | cnn[6] 總和 | cnn[6] x 值域 | cnn[23] 總和 | cnn[23] x 值域 |
  |---|---|---|---|---|
  | 1 | −355,931 | 0.00..1.00 | −314,252 | 0.00..1.00 |
  | 2 | −364,330 | −0.10..1.10 | −532,294 | −0.10..1.10 |
  | 3 | −170,234 | −0.20..1.20 | −222,235 | −0.20..1.20 |
  | 6 | 244,823 | −0.50..1.50 | −76,504 | −0.50..1.50 |
  | 10 | 598,093 | −0.90..1.90 | −5,259 | −0.90..1.87 |
  | 20 | 1,331,296 | −1.94..2.91 | 88,859 | −1.87..2.73 |
  | 50 | 3,828,020 | −5.37..6.15 | 239,616 | −4.18..5.93 |
  | 100 | 9,102,480 | −11.77..12.17 | 371,812 | −6.71..7.58 |
  100 步之後再 forward 一次（程式沒有做，工具多做的）：cnn[6] 9,216,263（每張 874,798–976,690）；cnn[23] 373,688（每張 28,914–50,499）。
- 和「CNN 實測」記錄的數字（598,087、3,828,560、9,108,426；x −11.87..12.32）在第 4–5 位不同：GPU 非確定性（ch00 0.6 節），見下面「每次跑都不一樣」。
- 看得到的（ch04_trajectory.png）：cnn[6] 總和在前 2 步略降，之後一路上升、到 100 步還在加速（曲線往上彎），沒有收斂；cnn[23] 第 2 步跌到最低（−53 萬），之後快速回升、約第 10–11 步過 0，然後越升越慢。
- 看得到的（ch04_xrange.png）：前 ~20 步 x 的最大值、最小值都以每步約 0.1 的速度直線往外走（就是 lr）；cnn[6] 一直直線走到 ±12；cnn[23] 在 40 步後變慢、甚至一度往回。灰色帶是 [0, 1]。
- 第 1 步 Adam 讓**每個像素都剛好移動 lr**：cnn[6] 第 1 步梯度的絕對值從 9.06e-06 到 171.1（中位數 11.33），差了 7 個數量級，但更新量 |Δx| 最小 0.0999、中位數 0.1000、最大 0.1000，100% 的像素在 0.1 ± 0.001 之內。（Adam 第 1 步的更新 = lr × g / (|g| + ε)，就是 lr × 梯度的正負號；公式推得，數字實測。）所以 20 步之內值域每步擴張 0.1。
- 100 步後：cnn[6] x 有 90.3% 的值在 [0,1] 外，和原圖的平均絕對差 3.175；cnn[23] 57.5% 在外，平均差 0.576。
- **模型對優化後的圖的預測**：cnn[6] 的 10 張 → Vegetable/Fruit 3 張（圖 0、4、5）、Dessert 7 張；cnn[23] 的 10 張全部 → Dessert。（原圖 10 張全對，ch01。）

### 每次跑都不一樣、batch vs 單張
- 同樣從原圖、同樣 100 步，跑兩次（同一 process）：圖 0 的 x 差異 cnn[6] 最大 5.24、平均 0.199、56% 的值差超過 0.1；cnn[23] 最大 2.06、平均 0.045。normalize 之後差異：cnn[6] 最大 0.228、平均 0.0086；cnn[23] 最大 0.199、平均 0.0107。（ch00 0.6 節「filter_cnn6.png 肉眼看不出差異」是存成 PNG、normalize 之後的比較，兩者一致。）
- 圖 0 單獨優化 vs 在 10 張的批次裡：cnn[6] 最大差 17.85、平均 1.446、94% 差超過 0.1；cnn[23] 最大 4.83、平均 0.296。eval 模式下 BN 用固定的 running 統計，10 張圖的目標互不影響（總和對 x_i 的梯度只來自第 i 張），數學上應該完全相同。**推測**（沒有驗證）：GPU 依 batch 大小選不同的卷積演算法，梯度有極小的差異；Adam 依梯度大小正規化，梯度接近 0 的像素正負號一翻，就整步 lr 往反方向走，100 步後差異被放大。（「批次 vs 單張 logit 在小數第 3 位不同」見 ch01。）

### 每個 activation 看得到多大一塊（receptive field，實測）
- 對 filter 0 中心那一格的 activation 取輸入的梯度，看哪些輸入像素的梯度不是 0：cnn[6] 的 (64, 64) 只依賴輸入的 rows 61..67、cols 61..67，也就是 **7×7**；cnn[23] 的 (16, 16) 依賴 rows 47..84、cols 47..84，**38×38**。
- 手算對照：3 個 3×3 卷積 → 7×7；再經 MaxPool、stage 2 三個卷積、MaxPool、stage 3 兩個卷積 → 38×38，和實測一致（計算方法：每個 3×3 卷積加 2×目前步距，每個 MaxPool 加 1×目前步距、步距乘 2）。

### 變體：從白雜訊出發、限制在 [0,1]、lr=1（圖 `img/ch04_variants_cnn6.png`、`img/ch04_variants_cnn23.png`）
- 白雜訊：`torch.manual_seed(0)` 後 `torch.rand(10, 3, 128, 128)`（0–1 均勻分佈）。「限制」= 每步 `optimizer.step()` 之後 `x.clamp_(0, 1)`。其他同 repo（lr 0.1、100 步）。
- 100 步後再 forward 的 filter 0 總和（10 張合計）與 x 值域：
  | 層 | repo（從原圖） | 從原圖＋限制 | 從雜訊 | 從雜訊＋限制 |
  |---|---|---|---|---|
  | cnn[6] | 9,216,263（−11.90..12.31） | 926,080（0..1） | 10,611,874（−12.21..12.99） | 956,999（0..1） |
  | cnn[23] | 373,688（−6.75..7.64） | 354,622（0..1） | 402,245（−6.37..7.57） | 367,707（0..1） |
  白雜訊本身在 cnn[6] filter 0 的總和是 −1,210,640（10 張合計，優化前）。
- 和起點的平均絕對差：cnn[6] 從原圖＋限制 0.352、從雜訊 3.561、從雜訊＋限制 0.371；cnn[23] 0.287、0.703、0.353。
- 從雜訊出發的 10 張結果彼此的平均絕對差：cnn[6] 3.931、cnn[23] 0.924（起點雜訊彼此是 0.333）。
- cnn[6] 限制在 [0,1] 時，總和只有不限制的約 1/10（926,080 vs 9,216,263）；cnn[23] 差不多（354,622 vs 373,688）。
- 看得到的（ch04_variants_cnn6.png，六列：image、repo: from image、from image, clamp [0,1]、white noise (start)、from noise、from noise, clamp [0,1]）：
  - repo 列：洋紅／綠的細短橫線、直線交錯的「迷宮」紋理鋪滿整張，底下還隱約看得到原圖輪廓（披薩三角、鬆餅格子、荷包蛋圓）；最右邊一行有一條橘紅色的直條（邊界）。
  - 限制列：同樣的迷宮紋理但更鮮豔（亮粉紅），原圖輪廓更清楚，有些圖角落有深紫色塊。
  - 從雜訊：迷宮紋理，**完全看不出原圖**，10 張彼此看起來幾乎一樣。
  - 從雜訊＋限制：同樣紋理、鮮豔的粉紅。
- 看得到的（ch04_variants_cnn23.png）：repo 列偏灰，有細小的彩色點狀紋理，原圖輪廓隱約可見（比 cnn[6] 清楚）；限制列是密密麻麻的小圓圈／眼睛狀圖案（白、藍、紅、黑），食物的輪廓（披薩、鬆餅、荷包蛋、湯碗）很清楚；從雜訊是一片灰，有很淡的點狀紋理，10 張相近；從雜訊＋限制是滿版的小圓圈圖案。
- 看得到的（`img/ch04_crop.png`，圖 0 中間 64×64 放大：原圖、cnn[6] repo、cnn[23] repo、cnn[6] from noise）：cnn[6] 的紋理是幾個像素寬的洋紅／綠短線段，橫、直交錯；cnn[23] 是灰底上更細碎的彩色點；從雜訊出發的 cnn[6] 和從原圖出發的紋理是同一種。
- lr=1（函式的預設值，呼叫端傳的是 0.1）：cnn[6] 總和 103,450,840、x −119.74..120.75；cnn[23] 323,961、x −35.40..68.60。

### normalize 對 filter visualization 的影響
- 第 219 行 `normalize(img.permute(1, 2, 0))` 對一張圖的三個通道一起做 min-max。cnn[6] 圖 0 的 x 是 −11.07..12.05：原本的 0 對應到 0.479、原本的 1 對應到 0.522，原圖所有的亮度變化被壓進 0.48–0.52 這 4% 的範圍；只有 10.2% 的值還在 [0,1] 裡。圖 5（−11.48..11.03）：0 → 0.510、1 → 0.554、10.1%。cnn[23] 圖 0（−4.10..6.80）：0 → 0.376、1 → 0.468，52.6%；圖 5（−6.75..6.87）：0 → 0.496、1 → 0.569，44.6%。這就是第三列「偏灰、原圖淡淡的」的原因。
- 註：上面 cnn[6] 圖 0 的值域（−11.07..12.05）和 10 張的整體值域（−11.90..12.31）是不同範圍。

### 其他程式細節
- `filter_explain` 的 `filterid=0` 寫死，只看第 0 個 filter；`iteration=100`、`lr=0.1` 由呼叫端給（函式預設 lr=1）。
- 第 200 行 `.squeeze()`：x 是 (10, 3, 128, 128)，沒有長度 1 的維度，squeeze 不做任何事。（若只傳 1 張圖會把批次維度去掉。由 PyTorch 語意推得。）
- 第 213 行 `imshow(img.permute(1, 2, 0))` 直接給 torch 張量（沒有 `.numpy()`），可以畫。
- 第 192 行 `optimizer.zero_grad()` 清的是 x 的梯度；模型參數的 `.grad` 一樣會累加（同 ch03 的觀察，這裡沒有另外量）。
- 計時引用 ch00 0.5 節：cnn[6] 2.3 s、cnn[23] 2.9 s。

## ch04 審稿補測（2026-10-04，本機；PR #14）
工具：`docs/tools/hw09_ch04_review.py`（在 HW09/ 裡跑；產生 `docs/HW09/img/ch04_hook_bn_relu.png`）。對 explain_cnn.py 的修改是把原始碼文字改掉後 exec 到新的命名空間，檔案本身沒動。填掉 ch04 的 4 個 TODO。

- **hook 掛在 cnn[6] Conv／cnn[7] BN／cnn[8] ReLU**（同樣從原圖、Adam lr 0.1、100 步、目標 −filter 0 總和；100 次更新後再 forward 量）：
  | hook | 自己那層 filter 0 總和 | 同時 cnn[6] Conv 總和 | 掛的那層輸出 =0 的比例 | x 值域 |
  |---|---|---|---|---|
  | cnn[6] Conv（repo） | 9,216,958 | 9,216,958 | 0 | −11.95..12.30 |
  | cnn[7] BN | 1,122,571 | 9,221,400 | 0 | −11.79..12.46 |
  | cnn[8] ReLU | 7,199,764.5 | **−26,682,812** | 0.569 | −12.38..13.43 |
  - 優化結果兩兩比較（原始值平均絕對差／normalize 後平均絕對差）：Conv vs BN 0.704／0.0315；Conv vs ReLU 6.619／0.2681；BN vs ReLU 6.618／0.2677。（對照：同一做法跑兩次，normalize 後 0.0086，ch04 實測。）
  - 看得到的（ch04_hook_bn_relu.png，四列：image、hook cnn[6] Conv (repo)、hook cnn[7] BN、hook cnn[8] ReLU）：Conv 與 BN 兩列肉眼看不出差別（洋紅迷宮紋理）；ReLU 列換成黃綠色、串珠狀的斜紋，比較亮，原圖輪廓（披薩、奶油塊、鬆餅、荷包蛋）看得到。
  - 結論：證實 ch04 由公式推得的「Conv ≡ BN、≠ ReLU」。BN 總和小是因為除以 √93.17 ≈ 9.65 再乘 γ 1.2184（手算）。
- **拿掉第 172 行 `global layer_activations`**，跑 `filter_explain(model, images, cnnid=6)`：`TypeError: 'NoneType' object is not subscriptable`，發生在 `filter_activations = layer_activations[:, filterid, :, :].detach().cpu()`（拿掉一行後是第 182 行，原本第 183 行）。
- **第 189 行改成 `Adam(model.parameters(), lr=lr)`**：
  - cnn[6] 那次：`filter_visualizations` 和 `images` **逐位元相同**（`torch.equal` True）。
  - 被改的 state_dict 項目只有 cnn.0／1／3／4／6 的 weight 和 bias（hook 之前、影響 cnn[6] 的層；cnn.7 以後沒有梯度，Adam 跳過；BN 的 running_mean／var 不變）。改變量：cnn.0.weight 平均 11.16、最大 15.57（原本平均 |w| 0.179）；cnn.3.weight 平均 5.99；cnn.6.weight 平均 0.064、最大 15.56；cnn.6.bias 最大剛好 10.0000（= 0.1 × 100）。
  - 接著 cnn[23] 那次：再改到 cnn.7、10、11、13、14、16、17、20、21、23 的 weight／bias。
  - 兩次之後，模型對 10 張原圖的預測**全部是類別 2（Dessert）**；p(標籤) 只有圖 3、4（本來就是 Dessert）是 1.0，其他是 0.0。
  - Integrated Gradients（用改過的模型 vs 原模型，每張 10 步）：最大值從 0.90–3.03 變成 2.9e15–9.3e15；兩者 normalize 後的相關係數 −0.006..0.010（完全無關）。
- **刪掉第 203 行 `hook_handle.remove()`**，依序呼叫 cnn[6]、cnn[23]：第一次後 cnn[6] 有 1 個 hook；第二次後 cnn[6]、cnn[23] 各 1 個；全域 `layer_activations` 是 (10, 256, 32, 32)（cnn[23] 的）。cnn[23] 的第二列（activation）和正常版逐位元相同；第三列和正常版差：原始值平均 0.052、normalize 後 0.0109；正常版自己跑兩次：0.054／0.0116。→ filter_cnn23.png 看不出差別，證實 ch04 的推論。
- **cnn[6] 圖 5 的值域**：ch04 實測那次是 −11.48..11.03（工具有印、FACTS 漏記）。這次（審稿）repo 設定再跑：圖 0 −11.01..12.04（0 → 0.478、1 → 0.521）、圖 5 −11.45..11.04（0 → 0.509、1 → 0.554）。ch04 表格用前者（同一次執行）。
- 圖：`img/ch04_hook_bn_relu.png`，ch04 用作圖 4.10（4.9 節最後的「hook 掛在 BN 或 ReLU 之後會怎樣」）。

## ch05 實測（2026-10-04，本機；指令都在 HW09/ 裡執行）
工具：`docs/tools/hw09_ch05_ig.py`（在 HW09/ 裡跑；印出下面的數字，並產生 4 張 `docs/HW09/img/ch05_*.png`）。工具裡的 `ig(x, c, steps, rule, baseline)` 是「直線路徑上梯度的平均」（不乘 x − baseline），`rule='left'` 和程式一樣取 α = k/steps（k = 0..steps−1），`'mid'` 取 (k + 0.5)/steps；梯度對象是標籤 logit（`model(x)[:, c].sum().backward()`）。

### 行號（以本檔為準）
- explain_cnn.py：docstring 223；`class IntegratedGradients` 225–265（`__init__` 226–230 含 `model.eval()` 230；`generate_images_on_linear_path` 232–235，list comprehension 在 234；`generate_gradients` 237–252：`requires_grad = True` 239、forward 241、`model.zero_grad()` 243、one-hot 245–246、`backward(gradient=one_hot_output)` 248、取 `.grad` 249、`.numpy()[0]` 251；`generate_integrated_gradients` 254–265：`np.zeros(input_image.size())` 258、累加 263、`[0]` 265）；`integrated_gradients` 268–281（`images.cuda()` 269、每張 `unsqueeze(0)` 274、steps=10 在 275、畫圖 276–280、存檔 281）；主程式呼叫 312。
- outline 的「explain_cnn.py:223-281」正確（含 docstring）。
- 投影片 p.11（pypdf 抽出，逐字）：
  ```
  Integrated Gradients
  Question 18 to 20
  ● Flexible baseline
  Ref: https://arxiv.org/pdf/1703.01365.pdf
  ```

### 程式細節（讀碼＋實測）
- 第 250、264 行註解寫「[0] to get rid of the first channel」，實際去掉的是**批次維度**（形狀 (1,3,128,128) → (3,128,128)），不是 channel。
- 第 258 行累加器是 (1,3,128,128) 的 float64；第 263 行加上 (3,128,128) 的陣列，靠 NumPy 廣播變回 (1,3,128,128)；第 265 行再 `[0]`。
- 第 234 行 step = 0 時 xbar = `input_image * 0`，就是全黑圖（baseline）；step 最大 9 → α 最大 0.9。
- `images.cuda()` 不需要梯度，所以 xbar 是沒有 grad_fn 的葉節點，第 239 行可以直接設 `requires_grad = True`；每個 xbar 是新張量，輸入的梯度不會累加。第 243 行 `model.zero_grad()` 清的是**模型參數**的梯度（第 3、4 章累加下來的也一併清掉）。
- `labels[i]` 是 0 維張量，第 246 行 `one_hot_output[0][target_class] = 1` 直接拿它當索引可以用。
- 檢查：工具的 `ig(..., 10, 'left')` 和程式的類別對圖 0 的結果，最大差 3.0e-03（程式結果最大值 2.01）；兩者數學相同（one-hot backward ≡ 對那一格 logit backward），差異是 GPU 浮點。

### 程式的輸出與 completeness（重算，和「CNN 實測 → IG」表逐位相同）
| 圖 | 類別 | f(x) | f(0) | f(x) − f(0) | Σ 程式輸出 | Σ 程式輸出 × x | 程式輸出 max | 負值比例 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 10.331 | −7.580 | 17.911 | 0.273 | 19.119 | 2.010 | 0.499 |
| 1 | 1 | 19.787 | −3.327 | 23.114 | 9.356 | 24.851 | 0.898 | 0.499 |
| 2 | 1 | 7.808 | −3.327 | 11.135 | 8.418 | 6.439 | 1.654 | 0.504 |
| 3 | 2 | 9.024 | −3.658 | 12.682 | −7.373 | 10.975 | 1.893 | 0.504 |
| 4 | 2 | 12.216 | −3.658 | 15.873 | 20.791 | 17.960 | 1.747 | 0.498 |
| 5 | 3 | 12.389 | −1.921 | 14.310 | 9.317 | 12.767 | 1.105 | 0.496 |
| 6 | 5 | 22.948 | −3.364 | 26.311 | 1.506 | 23.017 | 2.468 | 0.504 |
| 7 | 6 | 11.446 | −5.953 | 17.399 | −7.293 | 16.557 | 1.935 | 0.498 |
| 8 | 8 | 13.004 | −3.524 | 16.528 | −7.785 | 12.057 | 1.757 | 0.515 |
| 9 | 9 | 14.110 | −0.412 | 14.522 | 14.499 | 16.327 | 3.025 | 0.498 |
（「max」是 |程式輸出| 的最大值。f(x) 是單張算的 logit；ch01 已說明它和 10 張批次算的值在小數第 2–3 位不同。）
- 程式輸出（路徑平均梯度）的總和和 f(x) − f(0) **毫無關係**：從 −7.785 到 20.791，正負都有。乘上 x 之後 10 張都落在差值附近，但 10 步的誤差可以很大（圖 2：6.439 vs 11.135；圖 8：12.057 vs 16.528）。**更正**：「CNN 實測 → IG」那句「乘上 x 之後，10 步就已經接近差值」說得太滿，以這裡為準。
- 程式輸出的值大約一半正、一半負（負值比例 0.496–0.515）。

### 全黑圖（baseline）本身
- 全黑圖的 11 個 logit：[−7.580, −3.327, −3.658, −1.921, −7.684, −3.364, −5.953, −6.349, −3.524, −0.412, −0.420]；模型把它判成類別 9（Soup），p = 0.4147；類別 10（Vegetable/Fruit）−0.420 緊跟在後。
- 所以「同類別的 f(0) 相同」（圖 1、2 都是 −3.327）只是因為 baseline 對每張圖都是同一張全黑圖、f(0) 只看類別，沒有別的意義。

### 收斂：Σ(路徑平均梯度 × x) 隨步數與取點方式（目標是 f(x) − f(0)）
| 圖 | 目標 | 左 10 | 左 50 | 左 200 | 中點 10 | 中點 50 | 中點 200 |
|---|---|---|---|---|---|---|---|
| 0 | 17.911 | 19.126 | 16.715 | 17.973 | 15.750 | 18.146 | 17.851 |
| 1 | 23.114 | 24.856 | 23.715 | 23.272 | 23.412 | 23.216 | 23.065 |
| 2 | 11.135 | 6.448 | 11.145 | 11.296 | 15.623 | 11.569 | 10.932 |
| 3 | 12.682 | 10.974 | 13.068 | 12.751 | 12.978 | 12.478 | 12.627 |
| 4 | 15.873 | 17.966 | 15.619 | 15.662 | 11.824 | 16.117 | 15.922 |
| 5 | 14.310 | 12.770 | 14.333 | 14.307 | 13.735 | 14.245 | 14.216 |
| 6 | 26.311 | 23.023 | 26.605 | 26.361 | 23.739 | 26.301 | 26.101 |
| 7 | 17.399 | 16.577 | 16.831 | 16.957 | 16.268 | 16.183 | 17.565 |
| 8 | 16.528 | 12.078 | 17.046 | 16.508 | 18.500 | 15.885 | 16.534 |
| 9 | 14.522 | 16.341 | 15.205 | 14.563 | 10.629 | 14.299 | 14.681 |
- 「左 10」就是程式的取點乘上 x（和上表 Σ 程式輸出 × x 在小數第 2 位不同，GPU 浮點）。
- 200 步時 10 張的誤差都在 3% 以內（最大：圖 7 左 16.957 vs 17.399 ≈ 2.5%；中點 200 步最大是圖 2 10.932 vs 11.135 ≈ 1.8%）。10 步時中點不一定比左端好（圖 0、4、7、9 中點 10 步反而更差；ch05 審稿更正，原本誤寫成「圖 2、4、9」）。
- 「CNN 實測」記錄的圖 0 中點 10 步 15.704 和這次 15.750 不同（不同次執行、GPU 浮點）；50 步 18.141 vs 18.146、200 步 17.851 相同。

### 圖 0 的路徑（圖 `img/ch05_path.png`、`img/ch05_alpha_images.png`）
- α 從 0 到 1 取 101 點：f(αx)（Bread logit）、p(Bread)、Σ(∂f/∂x × x)（在 α 處的梯度乘 x 的總和；積分 ∫ 這條曲線 dα 就是 completeness 的值）。程式用的 10 個 α：
  | α | f(αx) | p(Bread) | argmax | max\|grad\| | Σ grad × x |
  |---|---|---|---|---|---|
  | 0.0 | −7.580 | 0.0003 | 9 | 0.164 | −1.569 |
  | 0.1 | −6.563 | 0.0010 | 9 | 1.627 | 15.961 |
  | 0.2 | −6.240 | 0.0007 | 5 | 2.529 | 37.113 |
  | 0.3 | 1.240 | 0.4540 | 0 | 3.971 | 85.861 |
  | 0.4 | 6.508 | 0.9944 | 0 | 4.302 | 31.470 |
  | 0.5 | 9.083 | 1.0000 | 0 | 3.965 | 18.761 |
  | 0.6 | 10.351 | 1.0000 | 0 | 3.369 | 5.274 |
  | 0.7 | 10.649 | 1.0000 | 0 | 3.100 | 1.025 |
  | 0.8 | 10.612 | 1.0000 | 0 | 2.404 | −0.595 |
  | 0.9 | 10.485 | 1.0000 | 0 | 2.620 | −2.039 |
  | 1.0 | 10.331 | 1.0000 | 0 | 2.020 | −0.523 |
  （α = 1.0 那一列程式不會算到。）
- 圖 0 在 α = 0.30 第一次判成 Bread，α = 0.39 時 p(Bread) ≥ 0.99。
- 看得到的（ch05_path.png 三格）：左，logit 從 −7.6 在 α 0.2–0.4 之間急升，0.6 之後持平在 10.3–10.6（0.7 附近最高，之後略降）；中，p(Bread) 在 α ≈ 0.25–0.4 從 0 跳到 1，之後一直是 1；右，Σ grad × x 在 α ≈ 0.2–0.35 有一個很高、鋸齒狀的峰（最高約 97），α > 0.6 後接近 0、略為負；紅點是程式用的 10 個 α，峰附近只有 0.2、0.3 兩個點，這是 10 步誤差大的原因（推論）。
- 看得到的（ch05_alpha_images.png）：α 0.0 全黑，往右逐漸變亮，α 0.3 已經看得清楚披薩與叉子。
- **意義（推論，由上表）**：f 在 α ≥ 0.6 已經飽和（圖再亮，logit 幾乎不變），那一段的梯度乘 x 接近 0；IG 把 α 0.2–0.4 那段「logit 真正在變」的梯度也算進來，這就是 IG 論文說的解決 saturation（飽和）的方式。只看 α = 1 的梯度（就是 saliency 的對象），Σ grad × x 是 −0.523。
- 10 張圖的路徑（p(標籤) 第一次 ≥ 0.5 的 α）：圖 0 0.31、1 0.13、2 0.17、3 0.14、4 0.09、5 0.42、6 0.12、7 0.41、8 0.25、9 0.02。α = 0（全黑圖）時 p(標籤)：圖 9（Soup）0.4147，其他 0.0003–0.0917。路徑不一定單調：圖 2 在 α 0.5 時 p 0.0036（之前在 0.17 已過 0.5）、圖 4 在 α 0.5 時 p 0.0000（之前在 0.09 已過 0.5），α 0.9 時都回到 ≥ 0.9997。

### 畫圖：normalize 保留正負號
- 第 280 行 `normalize(img)` 對 (3,128,128) 三通道一起 min-max，**沒有取絕對值**。圖 0：最小 −2.010、最大 0.983，**值 0 被對到 0.672**；圖 3：−1.893..1.293，0 → 0.594。程式輸出約一半是負的，所以畫出來的底色是 0 對到的那個灰（不是黑），正的偏亮、負的偏暗，R/G/B 三通道各自正負不同就出現彩色細紋。
- 看得到的（integrated_gradients.png 下列）：每張都是灰底，物體邊緣有細的粉紅／綠色線條：奶油塊的稜線、牛奶壺的輪廓與把手、荷包蛋的同心圓、鬆餅的格子都看得出來；10 張的灰底深淺不同（圖 8 最暗，圖 0、2 偏淺），因為每張的 0 對到的位置不同（推論，由上面 0 → 0.672／0.594 的計算）。

### 乘上 x 之後的圖（圖 `img/ch05_ig_variants.png`）
- 五列：image、repo: avg grad (10 steps)、avg grad * x (10 steps)、avg grad * x (midpoint, 200)、|IG| max over RGB (hot)（中點 200 步、乘 x、取絕對值、RGB 取 max、各自 normalize、hot 色表）。
- normalize 後「程式輸出」和「程式輸出 × x」的像素相關係數：圖 0–9 0.940、0.980、0.948、0.932、0.905、0.936、0.896、0.895、0.918、0.842。「× x（左 10 步）」和「× x（中點 200 步）」：0.983、0.995、0.994、0.995、0.994、0.992、0.990、0.986、0.994、0.994。
- 看得到的：第二、三、四列幾乎看不出差別（灰底、彩色細紋）；第五列（hot）黑底，物體輪廓以紅黃亮點呈現：奶油塊方框、牛奶壺與草莓、荷包蛋圓、鬆餅整片、湯碗圓圈。
- **意義**：少乘 x 對**總和**（completeness）影響很大，對**圖的樣子**影響很小（相關 0.84–0.98）。原因（推論）：x 在 0–1 之間、大多數像素的值差不多，乘上去主要是整體縮放；暗的像素（x ≈ 0）會被壓下去。

### Flexible baseline（投影片 p.11；圖 `img/ch05_baselines.png`）
- 四種 baseline，全部用中點 200 步、乘上 (x − baseline)：black（全 0，程式用的）、gray 0.5（全 0.5）、blur（31×31 平均模糊，邊界 replicate）、noise（`torch.manual_seed(0)` 後 `torch.rand_like`，0–1 均勻）。noise 每張圖抽一次。
- f(baseline)、f(x) − f(baseline) 與 Σ 歸因：
  | 圖 | black f(b)／差／Σ | gray 0.5 | blur | noise |
  |---|---|---|---|---|
  | 0 | −7.580／17.911／17.851 | −4.437／14.768／14.797 | −5.787／16.119／16.150 | −17.982／28.314／28.217 |
  | 1 | −3.327／23.114／23.065 | −2.459／22.246／22.243 | 6.014／13.773／13.845 | −15.152／34.939／34.588 |
  | 2 | −3.327／11.135／10.932 | −2.459／10.267／10.251 | 4.192／3.616／3.627 | −16.697／24.504／24.243 |
  | 3 | −3.658／12.682／12.627 | 2.108／6.916／6.825 | 0.190／8.834／8.759 | −4.599／13.623／13.659 |
  | 4 | −3.658／15.873／15.922 | 2.108／10.108／9.843 | −0.926／13.142／13.265 | −4.075／16.290／16.364 |
  | 5 | −1.921／14.310／14.216 | −7.876／20.265／20.250 | −4.505／16.894／16.806 | −9.472／21.861／21.824 |
  | 6 | −3.364／26.311／26.101 | −6.137／29.085／29.001 | −14.139／37.086／37.058 | −10.125／33.073／32.933 |
  | 7 | −5.953／17.399／17.565 | −10.020／21.467／21.385 | −18.690／30.136／29.945 | −20.525／31.971／32.088 |
  | 8 | −3.524／16.528／16.534 | −0.940／13.944／13.962 | −4.236／17.241／17.271 | −10.350／23.354／23.265 |
  | 9 | −0.412／14.522／14.681 | −4.934／19.044／19.042 | 0.797／13.313／13.331 | −10.752／24.862／24.669 |
  四種 baseline 都滿足 completeness（Σ 和差值差 < 3%；最大是圖 4 gray 0.5：9.843 vs 10.108 ≈ 2.6%），但「差值」本身隨 baseline 變：同一張圖 2，對 blur 只有 3.616、對 noise 有 24.504。
- |歸因|（RGB 取 max）和 black 版的像素相關係數：gray 0.5 0.306–0.894；blur 0.201–0.627；noise 0.103–0.316（逐張：圖 0 0.507／0.333／0.249、1 0.894／0.627／0.134、2 0.698／0.236／0.137、3 0.553／0.435／0.304、4 0.485／0.362／0.235、5 0.550／0.407／0.238、6 0.397／0.402／0.265、7 0.601／0.201／0.103、8 0.793／0.506／0.316、9 0.306／0.452／0.215）。
- 看得到的（ch05_baselines.png，九列：image，然後每種 baseline 兩列「baseline 本身」「|IG|（hot）」）：black 與 gray 0.5 的 |IG| 相似（物體輪廓）；blur 的 |IG| 集中在邊緣與細紋（鬆餅格子很清楚、奶油塊只剩細邊），因為模糊圖和原圖只差在細節（推論）；noise 的 |IG| 散成滿版細點，形狀最不清楚。
- **意義**：baseline 決定了「相對於什麼」來分功勞；換 baseline，歸因圖會變。投影片的 Flexible baseline 只有這三個字，題目問什麼本 repo 不知道。

### 計時
- 引用 ch00 0.5 節：IG 整段 0.7 s（10 張 × 10 步 = 100 次 forward + backward）。

## ch05 審稿補測（2026-10-04，本機；PR #15）
工具：`docs/tools/hw09_ch05_review.py`（在 HW09/ 裡跑）。把 explain_cnn.py 第 265 行照 ch05 自我測驗第 1 題字面換成三行（`result = integrated_grads[0] * input_image[0].detach().cpu().numpy()`、`print('sum of IG:', result.sum())`、`return result`；改原始碼文字後 exec，檔案沒動），跑 `integrated_gradients(model, images, labels)`。
- stdout 逐字：
  ```
  sum of IG: 19.11851266922134
  sum of IG: 24.85052742367112
  sum of IG: 6.439128675088346
  sum of IG: 10.975121340375775
  sum of IG: 17.960180575904484
  sum of IG: 12.76702126532808
  sum of IG: 23.017462406626954
  sum of IG: 16.556605084804794
  sum of IG: 12.056517770682136
  sum of IG: 16.326940268601284
  saved ./output/ch05_quiz1.png
  ```
  （工具把存檔名換成 ch05_quiz1.png，照字面改的話是 integrated_gradients.png。）四捨五入到 3 位和「ch05 實測」表的「Σ 程式輸出 × x」完全相同。
- 看得到的（改後的圖，沒有放進書裡）：第二列仍是灰底、粉紅／綠細紋，輪廓位置和原本的 integrated_gradients.png 相同，肉眼分不太出差別。
- 雲端在 ch05 指出的 FACTS 疑點已核對並更正：(1) 中點 10 步比左 10 差的是圖 0、4、7、9（不是 2、4、9）；(2) 工具產生 4 張 ch05_*.png，不是 5 張。另：ch05_path.png 右圖 α ≈ 0.15 附近跌到約 −25，看圖確認屬實。

## ch06 實測（2026-10-04，本機；指令都在 HW09/ 裡執行，CPU）
工具：`docs/tools/hw09_ch06_bert.py`（在 HW09/ 裡跑，`HF_HUB_OFFLINE=1` 用快取的模型；import `bert_hidden_states` 取三組問答與模型名稱；印出下面的數字，並產生 4 張 `docs/HW09/img/ch06_*.png`：`ch06_q{1,2,3}_layers.png` 是把現有的 `bert_q{N}_layer{1,4,8,12}.png` 拼成 2×2，`ch06_pca_variance.png` 是新畫的折線圖）。

### 行號（以本檔為準）
- bert_hidden_states.py：import 1–9；docstring 12–32（Part 2 標題 12、Attention 14–15、Embedding 17–25 含四個步驟 19–23、本 repo 的說明 27–31）；`output_dir` 34、`hw9_bert_dir` 35、`qa_model_name` 36；`same_seeds` 40–48；三組問答 51–72（第 1 組 53–58、第 2 組 60–66、第 3 組 68–72）；`visualize` 75–122（tokenize 77、question／context 範圍 80–81、讀預存或現算 83–87、逐層迴圈 91、PCA 95、figure 97、逐 token 畫點 99–111、判斷答案 103、圖例 114–117、標題 118、存檔 119–121、print 122）；main 125–138。
- outline 寫的「docstring 12–32、三題資料 51–72、visualize 75–122、main 125–138」都正確。

### 投影片 p.12–15（pypdf 抽出，逐字）
```
=== p.12
T opic II: BERT explanation
=== p.13
Task
● Run the sample code and ﬁnish 10 questions (all multiple choice form)
● We’ll cover 3 explanation approaches
○ Attention Visualization
○ Embedding Visualization
○ Embedding analysis
● You need to:
○ Know the basic idea of each method
○ Run the code and observe the results
○ For some cases, you may need to modify a small part of the code
=== p.14
Attention Visualization
Question 21 to 24
● Visualize attention mechanism of 
bert using 
https://exbert.net/exBERT.html
Alternative link: 
https://huggingface.co/exbert/
Ref: https://arxiv.org/pdf/1910.05276.pdf
Tutorial: https://youtu.be/e31oyfo_thY
=== p.15
Embedding Visualization
Question 25 to 27
● Visualize embedding across 
layers of BERT using PCA 
(Principal Component Analysis)
● Fine-tuned for Question 
Answering
```
（「T opic」的空格是 PDF 抽字的結果。）

### exBERT 現況（2026-10-04 重測）
- `https://exbert.net/exBERT.html`：15 秒逾時，curl 回 000（和 10-03 相同）。
- `https://huggingface.co/exbert/` → 302 到 `https://huggingface.co/exbert` → 最後是 `https://huggingface.co/spaces/exbert-project/exbert`（HTTP 200）。
- HF API `api/spaces/exbert-project/exbert`：runtime stage **RUNNING**、sdk docker、lastModified 2023-05-12。Space 的 app 網址 `https://exbert-project-exbert.hf.space/` 302 到 `/client/exBERT.html`，HTTP 200，回傳 `<title>exBERT</title>` 的頁面。**頁面載得到；互動功能（選模型、輸入句子、看 attention）能不能用沒有在瀏覽器裡試**。
- 原版 notebook（cell 38–39）：Markdown「You are highly recommended to visualize on this website directly: https://exbert.net/exBERT.html」，程式 `display.IFrame("https://exbert.net/exBERT.html", width=1600, height=1600)`。本 repo 只在 docstring 第 15 行留一句。

### 原版 notebook 的 Part 2a（cell 41–53，`~/poyi/GitHubPublic/ML2022-Spring/HW09/HW09.ipynb`）
- cell 41：`!pip install transformers==4.5.0`；import `BertModel, BertTokenizerFast`（沒有 BertForQuestionAnswering）；`plt.rcParams['figure.figsize'] = [12, 10]`；`same_seeds` 和本 repo 第 40–48 行相同，`same_seeds(0)`。
- cell 43：四個步驟的中英文說明（中文：1. 將類似的文字分羣（根據文字在文章中的關係）2. 提取答案 3. 將類似的文字分羣（根據文字的意思）4. 從文章中尋找與問題有關的資訊；「這些步驟並**不**按照順序排列」；「你可以在只看見模型 hidden states embedding 的情況下，找出各個layer的功能嗎?」）。本 repo docstring 19–25 只留英文。
- cell 45：`!gdown --id '1h3akaNdouiIGItOqEs6kUZE-hAF0QeDk' --output hw9_bert.zip`、`!unzip`；cell 47：`BertTokenizerFast.from_pretrained("hw9_bert/Tokenizer")`。
- cell 49：三組問答，和本 repo 51–72 逐字相同。
- cell 51：`QUESTION = 1`（「Choose from 1, 2, 3」，TODO 區）。
- cell 53：visualize 的本體，和本 repo 77–118 的邏輯相同，差別：原版讀 `torch.load(f"hw9_bert/output/model_q{QUESTION}")`（不跑模型）、沒有 `plt.figure`（用 rcParams 的 12×10）、最後 `plt.show()`；本 repo 包成函式、多了 `model is None` 分支（83–87）、`fig.savefig` 與 `plt.close`。

### 模型與 tokenizer
- `BertForQuestionAnswering`：12 層、hidden 768、12 個 attention head、intermediate 3072、max_position 512、vocab 28,996；參數 107,721,218；`qa_outputs = Linear(768, 2)`（每個 token 兩個分數：起點、終點）。
- 特殊 token id：[CLS] 101、[SEP] 102、[PAD] 0、[UNK] 100；`do_lower_case` False（cased）。所以第 80 行的 `index(102)` 是找第一個 [SEP]。
- `hidden_states[0]` 和 `model.bert.embeddings(input_ids, token_type_ids)` 的輸出相同（`allclose` True）：第 0 個是 embedding 層的輸出（還沒經過任何 attention）。`qa_outputs(hidden_states[12])` 等於模型的 start/end logits（True）：最後一層就是拿來找答案的那一層。
- `token_type_ids`：問題那段（含 [CLS]、第一個 [SEP]）是 0，文章那段（含最後的 [SEP]）是 1。Q1 11／67、Q2 13／107、Q3 8／48。
- 每層 token 向量的平均長度（L2 norm），layer 0→12：Q1 19.31 22.75 23.28 23.15 23.74 24.76 25.43 24.91 24.48 25.43 25.83 25.89 **20.79**；Q2、Q3 同樣形狀（layer 1–11 約 22–26，layer 12 掉回約 20.3–20.8）。

### 三組問答的 token 與標色
- context 字串用 `\` 續行，續行的 12 個空白會留在字串裡（Q1 2 處、Q2 3 處、Q3 1 處；例如 Q1 `'al engineer,             mec'`）；tokenizer 把空白當分隔，不影響 token。
- Q1：78 tokens，question 1..9、context 11..76，[SEP] 在 10、77。藍點 [32] `1856`。
- Q2：120 tokens，question 1..11、context 13..118，[SEP] 在 12、119。tokens 全部：`['[CLS]', 'What', 'is', 'a', 'common', 'punishment', 'in', 'the', 'UK', 'and', 'Ireland', '?', '[SEP]', 'Currently', 'detention', 'is', 'one', 'of', 'the', 'most', 'common', 'punishment', '##s', 'in', 'schools', 'in', 'the', 'United', 'States', ',', 'the', 'UK', ',', 'Ireland', ',', 'Singapore', 'and', 'other', 'countries', '.', 'It', 'requires', 'the', 'pupil', 'to', 'remain', 'in', 'school', 'at', 'a', 'given', 'time', 'in', 'the', 'school', 'day', '(', 'such', 'as', 'lunch', ',', 're', '##cess', 'or', 'after', 'school', ')', ';', 'or', 'even', 'to', 'attend', 'school', 'on', 'a', 'non', '-', 'school', 'day', ',', 'e', '.', 'g', '.', '"', 'Saturday', 'detention', '"', 'held', 'at', 'some', 'schools', '.', 'During', 'detention', ',', 'students', 'normally', 'have', 'to', 'sit', 'in', 'a', 'classroom', 'and', 'do', 'work', ',', 'write', 'lines', 'or', 'a', 'punishment', 'essay', ',', 'or', 'sit', 'quietly', '.', '[SEP]']`。藍點 [14, 86, 94] 三個 `detention`。
- Q3：56 tokens，question 1..6、context 8..54，[SEP] 在 7、55。藍點 [12] `cats`。
- 三組都檢查過：`Tokenizer.decode(單一 id)` 和 `convert_ids_to_tokens` 結果完全一樣（`##sla` 這類片段 decode 出來仍帶 `##`）。

### 這個模型自己的答案（start/end logits）
| 組 | argmax 起點／終點 | [CLS]（無答案）分數 | 最好的非空答案 | 分數 | 起點分數前 3 |
|---|---|---|---|---|---|
| 1 | 32／32 `1856` | −0.114 | `1856`（32..32） | 7.576 | `1856` 3.47、`10` 0.16、`[CLS]` −0.12 |
| 2 | 0／0 `[CLS]` | 11.720 | `detention`（14..14） | −0.487 | `[CLS]` 6.26、`detention` 0.91、`requires` −3.26 |
| 3 | 0／0 `[CLS]` | 4.580 | `sheep`（26..26） | −1.322 | `[CLS]` 1.69、`sheep` −1.53、`cats` −1.84 |
（分數 = 起點 logit + 終點 logit；非空答案限制在 context 內、長度 ≤ 30 token。）
- 第 2 組：模型認為「沒有答案」的分數遠高於任何片段，但非空答案裡最好的就是第一個 `detention`。
- 第 3 組：正確答案是 cats（Emily 是狼，狼怕貓），模型選「無答案」；非空裡最好的是 `sheep`（錯），`cats` 在起點分數排第 3。
- 這只說明這個替代模型的行為，和作業題目無關。

### PCA 保留的變異比例（含程式沒畫的 layer 0；圖 `img/ch06_pca_variance.png`）
- Q1：0.144 0.145 0.154 0.136 0.139 0.142 0.148 0.150 0.169 0.188 0.223 0.243 **0.376**
- Q2：0.139 0.139 0.127 0.122 0.119 0.123 0.129 0.130 0.151 0.168 0.187 0.221 **0.485**
- Q3：0.240 0.261 0.260 0.239 0.230 0.235 0.228 0.223 0.222 0.225 0.227 0.264 **0.491**
（layer 1–12 和「BERT 實測」記錄的兩位小數一致。）看得到的：第 1、2 組從 layer 8 起逐步上升，第 3 組到 layer 10 大致持平（0.22–0.26），layer 11 起上升；三組在 layer 12 跳到 0.38–0.49。（ch06 審稿更正：原本寫「三條線在 layer 0–10 大致平」，和表不符。）
- PCA 座標範圍（決定圖的刻度）：Q1 layer 1 x −8.0..13.5、y −7.6..12.1；layer 12 x −28.3..4.6、y −6.6..13.9。Q2 layer 12 x −21.2..6.5、y −24.6..4.6（註：這是全部 token 的範圍，含不畫的 [CLS]/[SEP]；圖上畫出的點最低到 detention 的 −9.6，y 軸刻度依 matplotlib 自動決定）。Q3 layer 12 x −18.4..7.9、y −7.4..14.5。第 111 行把字寫在點的右上方 (+0.1, +0.2)（資料座標），刻度範圍大時字和點幾乎重疊。

### 答案 token 在 PCA 平面上的位置
- Q1 `1856`：layer 1 (2.4, 1.3)；layer 12 (−28.3, 7.9)，就是 x 範圍的最左端，**整張圖最遠的離群點**。
- Q2 三個 `detention`：layer 1 (11.4, 8.1)、(11.8, 7.1)、(11.9, 7.3)（三個疊在一起）；layer 12 位置 14 (−18.6, −9.6)、86 (−9.1, 1.0)、94 (−12.2, −0.1)。第一個 `detention`（「Currently detention is…」，也是模型非空答案的那一個）在 layer 12 離群最遠。
- Q3 `cats`：layer 1 (7.8, 9.4)；layer 12 (−15.8, 0.0)。

### 看得到的（拼圖 `img/ch06_q{1,2,3}_layers.png`，2×2：左上 Layer 1、右上 Layer 4、左下 Layer 8、右下 Layer 12）
- **第 3 組**：
  - Layer 1：同一個字的點疊在一起，問題裡的字（紅）緊貼文章裡同一個字（綠）：所有 `afraid`（含問題的）在右下角自成一群；`is`/`are`/`of`/`a`/`.` 在左邊一群；動物名詞（sheep、wolf、wolves、Wolves、cats、Cats、mouse）在右上；人名與拆開的片段（Gertrude、Emily、Jessica、Win、##ona、She、Mi、##ep）在中間一條斜線上。答案 `cats`（藍）在動物群裡，和 `Wolves`、`Cats` 貼在一起。
  - Layer 4：`afraid` 仍在下方一群；左邊是虛詞的一大團；右上是動物；中間是人名。（只在拼圖的縮小版上看過，細節以原圖 bert_q3_layer4.png 為準。）
  - Layer 8：句號 `.` 在下方排成一直列；問題的 6 個紅點（What、?、is、of、afraid、Emily）在中間排開，和文章裡的同一個字分開了（文章的 `afraid`、`is`、`of` 在左上，問題的在中間）；動物在右上，`cats` 在其中；人名在右側中段。
  - Layer 12：問題的紅點聚在右側，和大部分文章 token 擠成一團；動物（sheep、wolves、wolf）在左邊；句號在上方；`cats`（藍）在 (−15.8, 0.0)，靠近左邊的動物群。（「BERT 實測」對這張圖的描述：句號一群、問題 tokens 與 Emily/is/afraid 聚在右側、動物名詞聚在左側。）
- **第 1 組**：
  - Layer 1：西里爾字母片段（Н、##и、##к… 與 Т、##е…）在右下角一群；`electrical`、`engineer`、`mechanical`、`physicist` 在上方一群；`1856`（藍）在中間偏右，附近有 `born`、`known`、`Te`；數字 `10`、`7`、`1943`、`January`、`July` 在中下方。
  - Layer 4：西里爾字母仍在右邊一群；問題的紅點（In、what、year、born、was、?）在左下聚在一起；`1856` 在中下方，靠近 `1943`、`10`、`January`、`July`。
  - Layer 8：紅點（what、year、In、was、born、?）在上方一群；`1856` 在中間，旁邊是 `Nikola`、`10`、`July`。
  - Layer 12：`1856` 獨自在左上角（x −28.3），離其他點都很遠；其次往左的是 `10`、`)`、`;`、`1943`；問題的紅點和大部分文章 token 擠在右側。
- **第 2 組**：
  - Layer 1：三個 `detention` 疊在右上角，和三個 `punishment`（含問題裡的紅色 `punishment`）在一起；`school`/`schools` 在右下角一群；虛詞在左邊一團。
  - Layer 4：國家名（UK、Ireland、United、States、Singapore、countries）在上方一群，問題裡的紅色 `UK`/`Ireland` 也在其中；`punishment`、`detention`、`pupil` 在右側中段。
  - Layer 8：紅點（What、is、a、in、the、common、punishment…）多數在右下一群；紅色 `Ireland`、`UK`、`and` 在右上，靠近文章的國家名；`detention` 在左側中段。
  - Layer 12：紅點全部擠在右側那一大團；三個 `detention` 都在左半邊，位置 14 那個在左下角最遠處 (−18.6, −9.6)；上方是 school、lunch、work、essay 等。
- 這些描述是逐張看圖得到的；「哪一層對應作業四個步驟的哪一步」本 repo 沒有標準答案（換了模型）。

### 768 維的量（不經過 PCA）
- 每層 cosine 平均（q-q：問題 token 兩兩；c-c：文章 token 兩兩；q-c：問題對文章）；「answer→question」是答案 token 對所有問題 token 的平均 cosine，括號是它在文章所有 token 裡的排名（1 = 最像問題）；NN 是答案 token 在全部 token 裡最像的 5 個（不含自己）。
  - Q1（`1856`，66 個文章 token）：layer 0 q-q 0.059 c-c 0.074 q-c 0.014，answer→q −0.005（第 47）；layer 6 0.270／0.290／0.219，0.215（第 36）；layer 9 0.380／0.305／0.221，0.226（第 30）；layer 11 0.505／0.520／0.376，0.209（第 66，最後）；layer 12 **0.861／0.755／0.775**，−0.358（第 66）。NN：layer 0–6 都是 `1943` 開頭（1943、–、July、January、10、7）；layer 7–11 `10` 開頭；layer 12 `[CLS]`、`10`、`)`、`1943`、`;`。
  - Q2（第一個 `detention`，106 個文章 token）：layer 0 0.126／0.105／0.063，−0.003（第 83）；layer 9 0.372／0.407／0.279，0.280（第 57）；layer 11 0.561／0.625／0.512，0.298（第 106，最後）；layer 12 **0.948／0.798／0.827**，0.033（第 106）。NN：layer 0–5 是另外兩個 `detention` 與三個 `punishment`；layer 6–11 是兩個 `detention` 加 `Currently`、`is` 等；layer 12 `detention`、`pupil`、`requires`、`lunch`、`detention`。
  - Q3（`cats`，47 個文章 token）：layer 0 0.105／0.134／0.070，−0.025（第 41）；layer 9 0.429／0.504／0.349，0.347（第 25）；layer 10 0.501／0.582／0.423，**0.448（第 13）**；layer 11 0.525／0.646／0.488，0.485（第 27）；layer 12 **0.893／0.689／0.698**，0.216（第 45）。NN：layer 0–6 `Cats` 第一、其次 wolves、sheep；layer 7–12 `wolves` 第一，都是動物名詞。
  - 完整 13 層的數字在工具輸出（每層一行）；上面只摘幾層。
- 觀察：(1) 三組的 q-q、c-c、q-c 都隨層變大，layer 12 一下跳到 0.69–0.95（所有 token 的方向變得很像，和 PCA 在 layer 12 保留的變異大增一致，推論）。(2) 答案 token「像不像問題」的排名在 layer 11–12 掉到最後或接近最後：答案在最後幾層反而和問題最不像（和 PCA 圖上 layer 12 答案離群一致）。(3) 答案 token 的最近鄰在前幾層是「同一個字或同類字」（另外的 detention、Cats、1943），後面幾層仍是同類（動物、數字）。這些都是描述，不是「哪層做哪一步」的答案。
- Q3 三組 token 的平均 cosine（動物名詞 14 個含拆開的 She/##ep/Mi/##ce；句號 `.`；人名 Emily/Gertrude/Jessica/Win/##ona）：layer 1 動物-動物 0.231、句號-句號 0.857、人名-人名 0.255、動物-句號 0.123；layer 6 0.523／0.849／0.514／0.414；layer 12 0.662／0.903／0.763／0.586。

### 程式細節（讀碼＋實測）
- 第 91 行 `enumerate(outputs_hidden_states[1:])`：layer_index 0..11，標題與檔名用 `layer_index + 1`，所以檔名的 layer1 是 hidden_states[1]（第 1 層 attention 之後），layer 0（embedding）沒有畫。
- 第 95 行 `PCA(...).fit_transform(embeddings[0])`：直接傳 torch 張量（模型分支在 `torch.no_grad()` 裡算，不需要梯度），sklearn 會轉成 NumPy。每一層各自 fit 一次 PCA，所以**不同層的圖座標軸不能互相比**（每層的 2 個主成分方向不同；主成分的正負號也是任意的）。
- 第 103 行先判斷答案，所以答案字即使在問題裡出現也會畫成藍色（這三組的答案字都不在問題裡）。
- 第 109–110 行：[CLS] 與兩個 [SEP] 不畫（它們不在 question/context 範圍、也不是答案）；但 PCA 是用**全部** token（含 [CLS]、[SEP]）fit 的。
- `same_seeds(0)`（第 126 行）：這支腳本在 CPU 上、模型在 eval 模式、PCA 有 `random_state=0`，本身沒有用到亂數；ch00 0.6 節實測 36 張圖重跑逐位元相同。
- 計時引用 ch00：real time 12.3 秒（模型已在快取）。LOAD REPORT 的 UNEXPECTED（pooler）ch00 已講。

## ch06 審稿補測（2026-10-04，本機；PR #16）
工具：`docs/tools/hw09_ch06_review.py`（在 HW09/ 裡跑，CPU）。填掉 ch06 的 3 個 TODO。
- **PCA 實際用的 solver**：三組問答的 (78／120／56, 768) 都是 `randomized`（sklearn 的 `svd_solver='auto'` 在這個大小改用 randomized SVD）。`random_state` 0 換成 1，layer 12 座標最大差：第 1 組 8.322e-05、第 2 組 2.616e-07、第 3 組 5.344e-06。
- **第 2 組 layer 12 的特殊 token**：位置 0 `[CLS]` (−21.2, −24.6)，x、y 都是全部 token 的最小值；位置 12 與 119 兩個 `[SEP]` 都在 (5.4, −2.1)（重疊）。畫出來的點範圍 x −18.6..6.5、y −9.6..4.6。其他層的 [CLS]／[SEP]：layer 1 (−2.9, −0.1)／(−2.9, 0.2)／(−3.3, 0.0)；layer 4 (−3.0, −0.5)／(0.6, 2.3)／(0.6, 2.3)；layer 8 (6.8, −2.7)／(6.4, −0.2)／(6.4, −0.2)。→ 只有 layer 12 的 [CLS] 跑到遠處；第 2 組的模型答案正是 [CLS]（無答案）。
- **自我測驗第 6 題**：第 103 行照字面改成 `if word.lower() in answers[QUESTION-1].split():`，跑 `visualize` 三組（改原始碼文字後 exec，檔案沒動；圖存到 `HW09/output/ch06_quiz6/`）。每張圖的藍點數：第 1 組 1、第 2 組 3、第 3 組 2（攔截 `plt.scatter(color='blue')` 計數，12 層合計 12／36／24）。第 1、2 組 24 張 PNG 和 `docs/HW09/img/` 的原圖**逐位元相同**；第 3 組 12 張全部不同。看得到的（第 3 組 layer 1）：原本綠色的 `Cats` 變成藍色菱形，在 `cats`、`Wolves` 旁邊。
- **工具 stdout 節錄**：ch06 6.10 節後的說明框補了 `hw09_ch06_bert.py` 的 9 行逐字輸出（三組的標題、argmax、PCA 變異比例），取自「ch06 實測」那次執行。
- 雲端在 ch06 指出的 FACTS 疑點（變異比例「layer 0–10 大致平」）已核對並更正，見上面「ch06 實測 → PCA」。

## ch07 實測（2026-10-04，本機；指令都在 HW09/ 裡執行，CPU）
工具：`docs/tools/hw09_ch07_embedding.py`（在 HW09/ 裡跑，`HF_HUB_OFFLINE=1` 用快取的 bert-base-chinese；import `bert_embedding` 取句子與函式；印出下面的數字，並產生 8 張 `docs/HW09/img/ch07_*.png`）。「變體」都是把 bert_embedding.py 的原始碼文字改掉幾行後 exec（跑的是同一份主程式，只把存檔路徑換成 `img/ch07_*.png`），檔案本身沒動。

### 行號（以本檔為準）
- bert_embedding.py：import 1–10；docstring 13–18；`output_dir` 20；字型註解 22–23、`FONT_PATH` 24；`same_seeds` 28–36；句子 39–50（句 0–9 依序在第 41–50 行）；TODO 區 docstring 53–55；`select_word_index` 註解 57–58、蘋 59、果（註解掉）60；距離函式註解 62、`euclidean_distance` 63–65、`cosine_similarity` 67–69；`METRIC` 71–72；`get_select_embedding` 74–82（`LAYER = 12` 在 76、取 hidden state 78、`word_to_tokens` 80、回傳 82）；main 85–120（字型 88–91、載入模型 93–94、tokenize 97、forward 100–101、取向量 104、`pairwise_distances` 107、畫圖 110–116、存檔 117–120）。
- outline 寫的「字型 20–24、句子 39–50、TODO 區 53–82、main 85–120」：字型實際是 22–24（20 是 output_dir），其餘正確。
- 投影片 p.16（pypdf，逐字）：
  ```
  Embedding Analysis
  Question 28 to 30
  ● Compare output embedding of 
  BERT using:
  ○ Euclidean distance
  ○ Cosine similarity
  You only need to change code in 
  the section “TODO” !
  ```

### 原版 notebook 的 Part 2b（cell 41、54–61）
- cell 41：`!gdown --id '1JWHUSlcPwoEzmr0VE6J71jcnwinH10G6' --output taipei_sans_tc_beta.ttf`，`myfont = FontProperties(fname=r'taipei_sans_tc_beta.ttf')`，註解「後續在相關函式中增加 fontproperties=myfont 屬性即可」，來源連結 willismax/matplotlib_show_chinese_in_colab。
- cell 55：載入 `BertModel`、`BertTokenizerFast`（'bert-base-chinese'），和本 repo 93–94 相同。
- cell 57：10 個句子，和本 repo 逐字相同（含「（富士)」全形半形混用、「發振」）。
- cell 59（TODO 區）：`select_word_index` 兩行、`euclidean_distance` 與 `cosine_similarity` 都是 `return 0`、`METRIC = euclidean_distance`、`get_select_embedding`（`LAYER = 12`）。本 repo 只把兩個 `return 0` 換成實作，第 62 行加註「The notebook leaves these two as `return 0` for students to fill in」。
- cell 61：主體和本 repo 97–116 相同；差別：原版 `plt.rcParams['figure.figsize'] = [12, 10]`（本 repo 用 `plt.figure(figsize=(12, 10))`）、`plt.yticks(..., fontproperties=myfont)`（只有 y 軸標籤用中文字型，本 repo 改成全域 `font.family`）、最後 `plt.show()`。原版註解有錯字「Pairwse comparsion … metirc」，本 repo 改成「Pairwise comparison … metric」。

### 模型與斷詞
- `BertModel`（bert-base-chinese）：12 層、hidden 768、vocab 21,128、參數 102,267,648。每句 `hidden_states` 13 個元素，例如句 0 每個 (1, 10, 768)。所有句子 `token_type_ids` 全部是 0（只有一段）。
- 每句的 token（[CLS] 與 [SEP] 各佔一格）：
  - 句 0 `['[CLS]', '今', '天', '買', '了', '蘋', '果', '來', '吃', '[SEP]']`
  - 句 1 `['[CLS]', '進', '口', '蘋', '果', '（', '富', '士', ')', '平', '均', '每', '公', '斤', '下', '跌', '12', '.', '3', '%', '[SEP]']`
  - 句 2 `['[CLS]', '蘋', '果', '茶', '真', '難', '喝', '[SEP]']`
  - 句 3 `['[CLS]', '老', '饕', '都', '知', '道', '智', '利', '的', '蘋', '果', '季', '節', '即', '將', '到', '來', '[SEP]']`
  - 句 4 `['[CLS]', '進', '口', '蘋', '果', '因', '防', '止', '水', '分', '流', '失', '故', '添', '加', '人', '工', '果', '糖', '[SEP]']`
  - 句 5 `['[CLS]', '蘋', '果', '即', '將', '於', '下', '月', '發', '振', '新', '款', '[UNK]', '[SEP]']`
  - 句 6 `['[CLS]', '蘋', '果', '獲', '新', '[UNK]', '[UNK]', '專', '利', '[SEP]']`
  - 句 7 `['[CLS]', '今', '天', '買', '了', '蘋', '果', '手', '機', '[SEP]']`
  - 句 8 `['[CLS]', '蘋', '果', '的', '股', '價', '又', '跌', '了', '[SEP]']`
  - 句 9 `['[CLS]', '蘋', '果', '押', '寶', '指', '紋', '辨', '識', '技', '術', '[SEP]']`
- 字數／word 數／token 數：句 1 是 20 字、19 個 word、21 個 token（`12` 兩個字是一個 word、一個 token）；句 5 17 字、12 word、14 token（`iPhone` 一個 word → `[UNK]`）；句 6 13 字、8 word、10 token（`Face`、`ID` 各一個 word，中間的空白不算）；其他句子字數 = word 數，token 數 = 字數 + 2。

### `word_to_tokens` 的「word」不是「字」（實測句 1）
- 句 1 的 `word_ids()`：`[None, 0, 1, …, 18, None]`（19 個 word）。
- 傳同一個整數 k，`char_to_token(k)`（第 k 個**字元**）與 `word_to_tokens(k)`（第 k 個 **word**）在 `12` 之前一樣，之後就錯開：
  | k | 第 k 個字元 | char_to_token | word_to_tokens → token |
  |---|---|---|---|
  | 2 | 蘋 | 3 | (3, 4) `蘋` |
  | 13 | 下 | 14 | (14, 15) `下` |
  | 15 | 1 | 16 | (16, 17) `12` |
  | 16 | 2 | 16 | (17, 18) `.` |
  | 17 | . | 17 | (18, 19) `3` |
  | 18 | 3 | 18 | (19, 20) `%` |
  | 19 | % | 19 | **None** |
  `word_to_tokens(19)` 回傳 None，程式第 80 行的 `.start` 會出錯（`AttributeError: 'NoneType' object has no attribute 'start'`，本書第一次寫工具時就踩到）。
- 程式的兩行 index 都指在英數字之前，字元位置 = word 位置，所以沒出事：蘋 `[4, 2, 0, 8, 2, 0, 0, 4, 0, 0]` → 10 句都取到 `蘋`，token 位置 `[5, 3, 1, 9, 3, 1, 1, 5, 1, 1]`；果 `[5, 3, 1, 9, 3, 1, 1, 5, 1, 1]` → 10 句都取到 `果`。
- 句 4 有兩個「果」（字元 3 的「蘋果」與 16 的「果糖」），果的 index 3 取的是前者。

### Layer 0：位置相同 → 距離剛好 0
- 「蘋」在 layer 0（embedding 層輸出）的歐氏距離矩陣（圖 `img/ch07_layer0.png`）：
  ```
  [[0.   5.83 7.66 7.26 5.83 7.66 7.66 0.   7.66 7.66]
   [5.83 0.   6.37 7.08 0.   6.37 6.37 5.83 6.37 6.37]
   [7.66 6.37 0.   7.93 6.37 0.   0.   7.66 0.   0.  ]
   [7.26 7.08 7.93 0.   7.08 7.93 7.93 7.26 7.93 7.93]
   [5.83 0.   6.37 7.08 0.   6.37 6.37 5.83 6.37 6.37]
   [7.66 6.37 0.   7.93 6.37 0.   0.   7.66 0.   0.  ]
   [7.66 6.37 0.   7.93 6.37 0.   0.   7.66 0.   0.  ]
   [0.   5.83 7.66 7.26 5.83 7.66 7.66 0.   7.66 7.66]
   [7.66 6.37 0.   7.93 6.37 0.   0.   7.66 0.   0.  ]
   [7.66 6.37 0.   7.93 6.37 0.   0.   7.66 0.   0.  ]]
  ```
  「蘋」的 token 位置 `[5, 3, 1, 9, 3, 1, 1, 5, 1, 1]`。距離剛好 0 的 12 對：(0,7)、(1,4)，以及 {2, 5, 6, 8, 9} 兩兩（10 對）——**正是「蘋」在同一個位置的句子**。layer 0 的向量只由「哪個字（蘋）＋第幾個位置＋token_type（都是 0）」決定（再經過 LayerNorm），同字同位置就完全相同；非 0 的距離只有 6 種值（5.83、6.37、7.08、7.26、7.66、7.93），只取決於兩句「蘋」各在第幾個位置。
  - 這說明了「BERT 實測」的「layer 0 公司組內只有 3.064」：公司句 5、6、8、9 的「蘋」都在位置 1，彼此距離 0；只有句 7（位置 5）不同。
- 看得到的（ch07_layer0.png）：整張只有深紫（0）和黃綠（5.8–7.9）兩種色塊，排成格子狀；左上（句 0）和句 7 的列一模一樣。

### Layer 12（程式預設）
- 歐氏距離矩陣（= `img/bert_embedding.png`）與 cosine 矩陣：見「BERT 實測 → Q28–30」，本次重算逐位相同（dtype float32，對角線 0／1，`max |M − Mᵀ| = 0`，完全對稱）。
- 每句最近的 3 句（歐氏；cosine 只有句 3 的順序不同：歐氏 [0, 1, 4]、cosine [1, 0, 4]）：句 0 → 4、3、1；句 1 → 4、3、0；**句 2 → 6、8、9**（三句公司）；句 3 → 0、1、4；句 4 → 1、0、3；句 5 → 6、2、9；句 6 → 9、8、5；**句 7 → 0、3、1**（三句水果）；句 8 → 9、6、2；句 9 → 8、6、2。
- 幾對的兩種量：0-7 歐氏 14.29／cosine 0.79；2-6 12.85／0.82；2-1 18.85／0.62；7-5 25.77／0.33。
- 看得到的（bert_embedding.png）：viridis 色表，對角線深紫（0），句 5 那一列／行在水果句（0、1、3、4）和句 7 的位置是最亮的黃（24–26）；左上 0、1、3、4 彼此偏藍綠（10–12）；右下 6、8、9 彼此偏藍（9.6–11.5）。每格寫著兩位小數。x 軸只有 0、2、4、6、8 五個刻度（程式只設了 y 軸的句子標籤，x 軸是 matplotlib 自動刻度）。
- 看得到的（`img/ch07_cosine.png`，METRIC 換成 cosine_similarity）：對角線是黃（1.00），顏色意義和歐氏相反：亮 = 像。句 5 對水果句是最暗的紫（0.31–0.39）；6、8、9 彼此 0.85–0.90 偏黃綠。

### 第 116 行 `plt.text(i, j, …)` 是轉置的
- `np.ndenumerate` 給的 (i, j) 是 (列, 行)；`plt.text(x, y)` 的第一個參數是**水平**座標（行），第二個是垂直（列）。所以 M[i, j] 的數字被寫在第 j 列、第 i 行的格子上，也就是 M[j, i] 的位置。矩陣完全對稱（max |M − Mᵀ| = 0），所以畫出來看不出錯；如果換成不對稱的量就會寫錯格。（由 matplotlib 參數順序推得，沒有另外畫不對稱的例子。）

### 各層：組內 vs 組間（圖 `img/ch07_layer_sweep.png`）
- 水果 = 句 0–4、公司 = 句 5–9。每層的「水果組內／公司組內／組間」平均（歐氏、cosine）：
  - 蘋 歐氏：L0 6.142/3.064/5.596 · L1 10.386/9.454/11.121 · L2 11.427/10.138/12.326 · L3 11.149/9.928/12.571 · L4 12.384/11.772/15.230 · L5 11.591/11.726/15.200 · L6 11.438/11.754/15.643 · L7 12.169/12.299/16.007 · L8 11.563/11.447/15.804 · L9 11.450/10.945/15.867 · L10 10.745/10.695/14.956 · L11 10.618/10.910/14.818 · L12 13.776/15.652/18.481（和「BERT 實測」表相同）
  - 蘋 cosine：L0 0.956/0.976/0.959 · L1 0.912/0.924/0.901 · L2 0.892/0.912/0.875 · L3 0.883/0.905/0.852 · L4 0.874/0.885/0.811 · L5 0.885/0.884/0.806 · L6 0.895/0.891/0.807 · L7 0.885/0.882/0.803 · L8 0.885/0.886/0.786 · L9 0.886/0.892/0.781 · L10 0.901/0.897/0.806 · L11 0.908/0.902/0.824 · L12 0.788/0.719/0.625
  - 果 歐氏：L0 6.602/3.282/5.944 · L1 10.573/8.200/10.068 · L2 11.058/8.957/10.602 · L3 11.343/9.839/11.525 · L4 13.333/11.807/14.812 · L5 12.065/11.728/15.238 · L6 11.327/11.269/15.295 · L7 12.302/12.284/16.289 · L8 12.646/12.342/17.445 · L9 12.557/11.966/17.459 · L10 11.506/10.774/16.451 · L11 11.387/10.819/16.261 · L12 12.375/11.570/17.130
  - 果 cosine：L0 0.940/0.967/0.946 · L1 0.908/0.940/0.915 · L2 0.893/0.928/0.901 · L3 0.874/0.903/0.869 · L4 0.850/0.880/0.815 · L5 0.872/0.878/0.797 · L6 0.895/0.895/0.811 · L7 0.883/0.880/0.795 · L8 0.864/0.865/0.739 · L9 0.860/0.869/0.730 · L10 0.878/0.890/0.751 · L11 0.890/0.900/0.777 · L12 0.854/0.874/0.726
- 看得到的（ch07_layer_sweep.png，蘋，左歐氏、右 cosine）：歐氏的「組間」線從 layer 1 起都在兩條組內線上方，layer 4 起拉開；cosine 的「組間」線從 layer 1 起都在兩條組內線下方，layer 4–9 差距最大，layer 12 三條一起往下掉。
- **最近鄰檢查**（每句在 cosine 下最像的另一句，是否同組；10 句裡對幾句）：
  - 蘋：layer 0→12 = 3, 8, 8, 9, 10, 10, 9, 9, 10, 10, 10, 10, **8**
  - 果：layer 0→12 = 3, 6, 6, 7, 10, 9, 9, 10, 10, 10, 10, 10, **10**
  - （layer 0 有大量距離 0 的平手，argmax 取第一個，所以 3 只是參考。）「蘋」在 layer 8–11 全對，layer 12 掉回 8（句 2、句 7 錯組，見上面的最近 3 句）；「果」在 layer 7–12 全對。

### 「果」的 layer 12 矩陣（圖 `img/ch07_guo.png`，把第 59、60 行對調）
- 看得到的：清楚的兩個方塊：左上 0–4 彼此 9.62–14.70（藍綠）、右下 5–9 彼此 9.43–14.65；兩塊之間 15.18–19.55（黃綠到黃）。句 2（蘋果茶）的「果」和水果句 12.14–14.70、和公司句 16.30–18.82；句 7（蘋果手機）的「果」和公司句 12.24–14.65、和水果句 16.46–17.51：**用「果」時這兩句都回到自己的組**。（和「蘋」的 layer 12 不同：那裡句 2 靠近公司、句 7 靠近水果。）
- 原因不明；本書只記錄現象。

### Layer 8 的「蘋」（圖 `img/ch07_layer8.png`，`LAYER = 8`）
- 看得到的：句 1–4 對句 5–9 多在 15–18（黃綠）；句 5、6、8、9 彼此 8.98–10.66；句 1-4 是 8.17（全圖最小的非對角值）；句 2 對水果句 11.11–13.24、對公司句 14.73–17.21（回到水果那邊）；句 7 對公司句 12.86–13.80、對水果句 13.32–16.93（最近的是句 8 的 12.86，但和句 0 的 13.32 差不多）。

### 沒實作距離函式（原版的 `return 0`；圖 `img/ch07_unimplemented.png`）
- 把第 65 行換回 `return 0`：`similarity_matrix` 100 格全部是 0，圖是一整片同色（viridis 中間的藍綠），colorbar 刻度 −0.100 到 0.100，每格寫 0.00。不會報錯。

### 字型（圖 `img/ch07_font_droid_only.png`、`img/ch07_font_none.png`）
- `FONT_PATH` 的檔案存在，family 名稱 `Droid Sans Fallback`。
- 本 repo 的設定 `['DejaVu Sans', 'Droid Sans Fallback']`：沒有任何缺字警告（這次所有變體都是 0 個 Glyph 警告）。
- 只設 Droid Sans Fallback（第 91 行拿掉 'DejaVu Sans'）：Glyph 警告 568 次、38 種字，全部是英數與符號：a–i、m–p、r、s、B–F、I、P、R、T、W、0–9、`%`、`)`、`.`。例如 `Glyph 73 (I) missing from font(s) Droid Sans Fallback.`。看得到的：中文句子正常，但所有數字（格內的兩位小數、colorbar、x 軸刻度）、英文標題、句子裡的英數（12.3%、iPhone、Face ID）和半形 `)` 都變成方框。→ 這個字型**沒有拉丁字母和數字**，不只缺「Face ID」的 I、D（「原版 vs 本 repo」那條只舉了 I）。
- 找不到字型（`FONT_PATH` 指到不存在的檔案，第 88 行的 if 不成立，只用預設 DejaVu Sans）：Glyph 警告 210 次，例如 `Glyph 20170 (\N{CJK UNIFIED IDEOGRAPH-4ECA}) missing from font(s) DejaVu Sans.`（U+4ECA 是「今」）。看得到的：數字、標題、英數都正常，y 軸句子的中文全變方框，只剩 `(`、`)`、`12.3%`、`iPhone`、`Face ID` 這些英數。

### 其他
- 執行時間引用 ch00：real time 6.7 秒。LOAD REPORT（cls.predictions／seq_relationship）ch00 已講。
- `same_seeds(0)`：CPU、eval、沒有亂數操作；ch00 0.6 節實測 bert_embedding.png 重跑逐位元相同。
- `pairwise_distances(embeddings, metric=函式)`：sklearn 對每一對呼叫一次函式，對角線也呼叫（所以 cosine 版對角線是 1、不是 0）；結果完全對稱。
