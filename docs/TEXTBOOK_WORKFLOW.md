# HW 教材的製作流程

docs/HWxx/ 底下每一本 HTML 教材的做法。HW01（docs/HW01，2026-10-03 完成）照這個流程完成，HW02、HW09 等之後的作業也沿用。本機 session 和雲端 session 都應該先讀這份。

使用的 skill 是 `completed-repo-to-html-textbook`，與 `cold-read` 一起放在公開的 [poyilee1030/mySkills](https://github.com/poyilee1030/mySkills) 裡，放到 repo 的 `.claude/skills/` 底下。`.claude/` 不 commit。

## 分工

**本機 session（有 GPU）**
- Phase 0：讀程式、跑 baseline，寫出大綱，等使用者核可。
- Phase 1：建立 docs/HWxx/（樣式、目錄、大綱）、`FACTS.md`，以及實驗工具。
- 每一章動筆前：把那一章會引用的數字與逐字輸出全部量好，寫進 `docs/HWxx/FACTS.md` 的「chNN 實測」，commit 並 push 到 master，然後給使用者一份雲端 prompt。
  - 另一種做法（HW09 用過）：一開始就把全書的事實一次量完，雲端可以連續寫好幾章，不必每章等本機。

**雲端 session（沒有 GPU，不跑任何程式）**
- clone mySkills 到 `.claude/skills/`。
- 只寫指定的那一章，只引用 FACTS 裡的數字；缺的數字標 `<!-- TODO(本機實測): 要量什麼 -->`，不估、不編。
- verify → 冷讀（2 支 subagent）→ inline_assets.py → 把新名詞補進 FACTS → 開 branch、開 PR。branch 名稱是雲端自己取的（例如 `claude/hw01-ch05-px9qd3`）。

**本機審 PR**
1. 用 `git merge-base` 確認 PR 是從哪個 commit 開始的；如果不是最新的 master，先把 master merge 進來。
2. 每一頁跑 `python3 docs/tools/verify_book.py . docs/HWxx/<page>.html`。
3. 檢查每一個 `href="x.html#id"` 的錨點都存在。
4. 用 grep 核對交叉引用與程式行號。
5. 在本機量出 TODO 的數字填回去，寫進 FACTS 的「chNN 審稿補測」。
6. commit 並 push 到 PR 的 branch；使用者 merge。

## 教訓（HW01 學到的，新書一開始就做）

- **Phase 0 就檢查驗證指標有沒有偏差**：比較訓練時印出的值，和最佳 checkpoint 在整個驗證集上一次算完的真實值。HW01 印出 1.661、真實是 2.069，這件事到 ch03 才發現，ch00、ch02 和目錄頁都得回頭改。
- **Phase 1 就寫好實驗工具**，並先驗證它能逐位元重現 train.py。HW01 的工具是 `docs/tools/hw01_exp.py` 和 `hw01_run_grid.sh`。之後每一章的實測、題庫核對都用它。
- **「改 X 會怎樣」這類題目，一律實際跑一次**：在 HW 目錄的複本裡照題目字面修改，跑真的 train.py。HW01 ch06 的自我測驗第 1 題就是推論錯了。
- **重建亂數順序時**：要額外載入的模型，先在 `same_seed` 之前建立；在它之後建立模型會消耗全域亂數，順序就對不上。
- **避免 FACTS 衝突**：上一章的 PR merge 之後，才追加下一章的「實測」。
- **每次 commit 前用 `git status -sb` 確認目前的 branch**。
- **stdout 和 stderr 混在一起的輸出**，要用 `script -qc "<指令>" /dev/null` 模擬終端機來抓；接到管線時順序會改變。
- **實驗用的額外套件裝在專案的 .venv 之外**，例如 `uv pip install --target <暫存目錄>`，不要改 requirements.txt。
- **不要 `cp -r .venv`**：它有 7G。
- **樣式版本**：HW01 的 docs/HW01/style.css 是舊版 skill 的複製，所以指令區塊用 `<pre class="shell">`，不用 `cmd`。新書可以用新版共用樣式（有 Python 上色和 `pre.shell.cmd`），HW09 選了新版。同一本書內要一致，寫進 prompt。
- **每本書的固定結構**：ch00 要有「模型總覽」一節（架構圖、各層形狀、參數量、定義在哪個檔案）；目錄頁要有「2022 vs 現在」的導論；各章遇到過時的寫法時加「現在的做法」框。

## 雲端 prompt 的寫法

HW01 每一份 prompt 都包含以下幾段：
- repo 是 `lordleeber/ML2022-Spring-pytorch`，base 是 master（沒有 main），以及預期的 HEAD hash。
- 「不需要環境、不要跑程式」：只引用 FACTS，缺的標 TODO。
- clone mySkills 的指令；`figure.listing` 的 `data-hot` 寫原始碼行號；指令區塊用哪一種 class。
- 「只寫 chNN，寫完就停」，以及範圍（檔名:行號）。
- **前面章節答應本章要講的事**（用 grep 搜「第 N 章」整理出來），以及哪些內容已經講過、只要回指不要重講。
- 本章的主線，以及容易寫錯的地方（例如「不能宣稱過了 Kaggle 基準線」）。
- 流程：verify → 冷讀 → inline_assets.py → 補 FACTS 名詞 → 開 PR → 回報（改了哪些檔、TODO 清單、需要本機核對的題目、前面章節要修的地方）。

## HW09 的調整

HW09 用同一套流程，但有幾處不同：
- 沒有訓練，所以不需要檢查驗證指標的偏差。這本書的陷阱是程式和論文、投影片不一致的地方（FACTS 的「repo 問題清單」列了六條）。
- 全部事實用 `docs/tools/hw09_facts.py` 一次量完（commit `5785d1e`），雲端可以連續寫好幾章。
- 44 張真實輸出的 PNG 放在 docs/HW09/img/，章節用 `<img>` 引用。
- 用新版共用樣式（使用者的決定）。
- `verify_book.py` 已經不再寫死 HW01。
- Phase 0：雲端先寫目錄頁和大綱就停，等使用者核可。

## HW02 的調整（2026-10-04～05）

HW02 沒有走雲端分工，整本由同一個本機 session 依序寫完（使用者決定）：
- **事實一次量完**，但訓練很慢（一次完整訓練約 2 小時 20 分），所以改良實驗用「同一套縮小規格」並排比（batch 64 × 5 epoch），另附一次原規格的完整紀錄。實驗工具先驗證和 train.py 逐位一致；batch 相同時，縮小規格的結果就是原規格的前幾個 epoch。
- **GPU 由使用者放行**：一次跑一組，跑完停下等指示；使用者授權「自己安排」時，也只在授權的時間內排。長時間的工作每 30 分鐘回報一次。
- **計時要乾淨**：計時那次 GPU 上不能有任何其他程式（包括自己順手跑的檢查），開跑前後用 `nvidia-smi --query-compute-apps` 記錄。HW02 第一次計時被自己的檢查干擾，重量了一次。
- **本機一章一停**：每章寫完 → verify_book.py → 回報 → 使用者說「yes」才推上 master、寫下一章。HW02 在使用者要求下暫停了冷讀（只限那個 session），所以 HW02 各章沒有冷讀。
- **圖表**：資料圖照 dataviz skill（配色先用 validate_palette.js 驗證，深色底 `#161c24`），附 hover 與同數字的表格。SVG 用 PyMuPDF 渲染成 PNG 目視檢查；MuPDF 不吃 CSS class、`marker-end` 與 inline `style`，渲染前要把 `s-lbl`／`s-sm`／`s-mono`／`s-box` 換成屬性。HW02 靠這一步抓到兩處標籤重疊、一處被切掉。
- **「改 X 會怎樣」的題目照樣實測**，能用現成 checkpoint 推論的就不用重訓（例如驗證集 shuffle 對 acc／loss 的影響）。
- 單純推論出來、沒有實測的答案，在題目裡註明「本書沒有實際跑」。

## HW03 的調整（2026-10-05）

HW03 也是同一個本機 session 依序寫完（使用者決定），沿用 HW02 的做法，另外學到這些：
- **實驗工具要「連進度條一起」複製**：`from tqdm.auto import tqdm` 在一般腳本裡是 `tqdm_asyncio`，建構時多呼叫一次 `iter(loader)`，每個迴圈多抽一個亂數；工具第一版拿掉 tqdm，第一個 batch 就不同。工具要用 `tqdm(..., disable=True)` 包住同樣的迴圈，並在動用 GPU 跑大量實驗前，先用 `diff` 比對 stdout、`torch.equal` 比對 checkpoint，確認逐位元一致。驗證不一致時立刻停掉排程，不要讓後面幾組白跑。
- **`num_workers>0` 也會改變亂數消耗**（多行程迭代器建立時就預取），所以「只改 worker 數」不是中性的修改。
- **計時要看 CPU 執行緒**：PyTorch 預設開滿核心，讀圖的小運算反而變慢；HW03 用 `OMP_NUM_THREADS=1` 快了 18% 且結果不變。計時前後記 `nvidia-smi --query-compute-apps`，量的時候不要同時跑別的 CPU 量測。
- **推送**：只在使用者明講「推」時推，一章一個 commit、一次推一個；「繼續」只代表寫下一章。使用者說「不用再等我決定」時，可以自己做完並推上去。
- **GPU**：使用者核可實驗規格後，可以一組一組依序自己排；規格之外的加跑（例如 HW03 的 res0_40）先問。長時間訓練每 30 分鐘回報一次（session 內的 cron）。
- **不在規格裡、但能從現成結果算出來的**（holdout 重算、TTA、ensemble），用 `--dump` 存下的 logits 與 checkpoint 推論即可，不必重訓。checkpoint 只放在 scratchpad，session 結束就沒了，數字一定要寫進 FACTS 與 `docs/tools/hwNN_*runs.jsonl`。
- **回填**：後面的章推翻前面章節的說法時（HW03 ch03 解開 ch00、ch01 的「第 1 個 epoch」），先問使用者，再做點狀修改並重驗那幾章。
- **圖片**：根目錄 `.gitignore` 排除 `*.jpg`；要放真實圖片時在 `.gitignore` 加 `!docs/HWxx/img/*.jpg`。
- 冷讀：HW03 使用者在 session 中途停用，各章沒有冷讀；預設仍是要跑。

## HW04 的調整（2026-10-05～06）

HW04 也是同一個本機 session 依序寫完，沿用 HW02、HW03 的做法（這本沒有冷讀，使用者決定），另外學到這些：
- **原規格便宜時，全部用原規格**：一次 70,000 步不到 4 分鐘，ch08 每組都跑原規格，不必縮小。
- **量法本身要先查**：`valid()` 量的是隨機片段，`test.py` 用整句，兩者差 17 個百分點；每組實驗都同時報「印出的」與「整句」（再加固定片段），否則像「切段 256」這種只改量法的實驗會被誤判成有效。
- **對照組要控制參數量**：「換成 X 變好了」之前，先跑一組參數量相同的對照。HW04 的 Conformer 輸給同參數的 pre-norm Transformer，ch06／ch07 因此改寫結論；消融（拿掉一個元件）比只看總表可靠。
- **facts 腳本的亂數陷阱**：建模型或任何用到全域亂數的動作會改變後面 `random_split` 的結果；腳本在切分前一律 `set_seed` 重設，並 assert 驗證集前幾個 index，HW04 靠這個抓到兩次。
- **引用實驗工具的行號後就不要改那幾行**：章節引用了 `docs/tools/*.py` 的某段，之後要加功能就加在別處（HW04 的 `--kernel 0` 改成在 `Net` 裡替換模組），否則已推的章節驗證會失敗。
- **scratchpad 可能在 session 中途被清空**（HW04 發生一次）：checkpoint 與暫存工具都會消失，所以工具要能重跑出相同結果，數字一律即時寫進 FACTS 與 jsonl。
- **授權範圍照字面**：使用者說「之後不用等我、一路做完並推」時，每章寫完驗證就推、繼續下一章；規格外的加跑在授權裡明講過才跑。


## HW06 的調整（2026-10-07～08）

HW06 同樣由本機 session 依序寫完（不冷讀，使用者決定），另外學到這些：
- **大綱前先對照目前版本的 skill**：HW06 原本照抄 HW04 的結構（含 appendix）與舊共用樣式，寫完才發現 completed-repo-to-html-textbook 已改成「不寫附錄」「不用舊的 style.css／enhance.js／inline_assets.py，每本書自己設計樣板、寫在 index.html」。後來把附錄併入最後一章、全書換樣板。下一本書在 Phase 0 就照現行 skill 做。
- **渲染可以在本機看**：Windows 的 Chrome 可以 headless 截 WSL 裡的頁面（`chrome.exe --headless=new --user-data-dir=<scratch> --window-size=1200,N --screenshot=<path> file://wsl.localhost/...`）；HW06 靠它抓到兩個版面錯誤。
- **評估工具也要先驗證**：FID 用的 pytorch-fid 0.3.0 在 SciPy 1.18 上會壞、OpenCV 5 沒有 CascadeClassifier；量法本身（樣本數、JPEG、種子）要在寫章之前先量清楚雜訊與偏差（HW06 第 4 章）。
- **checkpoint 之間也要看**：GAN 的崩潰可能發生在兩個存檔之間；每個 epoch 的樣本格可以算出逐 epoch 的多樣性（HW06 第 5 章）。
- **長任務每 30 分鐘回報**（session cron），使用者 2026-10-07 起的通用規則。

## HW14 的調整（2026-10-08～09）

HW14 由同一個本機 session 寫完（不冷讀；使用者授權「每章寫完驗證就推、接著寫下一章，需要 GPU 實驗就直接做，一組一組依序跑」），另外學到這些：
- **notebook 沒有 repo 時，先做「參照版」**：把 notebook 的程式格原樣串成腳本（只拿掉 `!` 指令、補上跑不起來的 import），拆檔後的 train.py 與實驗工具都對它做逐位元比對。notebook 在一般 Python 跑不起來的地方（隱性的 `tqdm.auto`、新版套件不收的參數）本身就是教材。
- **只為了 print 的程式也可能改變結果**：notebook 的 `example = Model()` 會用掉亂數，拆檔時要保留。多個方法共用一條亂數流時，「只跑一種」和「照順序跑到它」結果不同，多種子實驗要每種方法獨立跑。
- **實驗工具的額外量測要不碰亂數**：HW14 的逐任務評估用事先轉好的張量、`no_grad`，不經過 DataLoader（DataLoader 每建一次 iterator 就抽一次全域亂數，shuffle 再抽一次）。
- **「重要度」這類量先量級、再形狀**：EWC 的 batch Fisher 比逐樣本小 111 倍、SCP 先平均再平方小 100 倍，但排序相關都很高；調 λ 就能補回。量了量級才看得出「作業的成績表其實在排懲罰強度」。
- **多種子是基本盤**：baseline 換種子 ACC 就差 3 個百分點；一個種子的成績表排名要配五個種子的範圍一起看。
- **圖表產生器與 listing 填入工具**放在 `docs/tools/hw14_book/`（`charts.py` 直接從 jsonl 畫 SVG、附 hover；`fill_listings.py` 依 figcaption 的「檔名:行號」從原始碼重填 code，避免行尾空白造成 verify 失敗）。
- **正文字數**：skill 要求每章 3,000+ 中文字，寫完用 verify_book.py 的 cjk 計數檢查，不足就補一節實質內容（HW14 ch04–ch07 都補過）。

## HW13 的調整（2026-10-09）

HW13 由同一個本機 session 在一天內寫完（不冷讀；授權同 HW14：每章寫完驗證就推、需要 GPU 實驗就直接做）。另外學到這些：
- **範例程式找不到時，看投影片的連結**：官方 GitHub 只有資料，範例 notebook 是投影片 p.2 的 Colab 連結，`drive.google.com/uc?export=download&id=<id>` 可以匿名下載（先問使用者）。
- **任何會跑 forward 的檢查工具，先 `eval()`**：torchsummary 在訓練模式下用 `torch.rand` 跑一次 forward，會改掉 BatchNorm 的 running 統計量。HW13 Phase 0 因此把老師準確率量成 0.86093（正確 0.87143），已推送的三頁要回頭更正。量模型之前，確認它剛載入、或已經 `eval()`。
- **建模型、torchsummary 都會用掉全域亂數**：拆檔時保留原位置；「改 X 會怎樣」照字面在 scratchpad 複本跑（HW13 拿掉建老師那一行，結果從 0.504 變 0.517）。
- **對照組要事先想、事後補**：同參數的 `plain` 推翻了「depthwise 比較好」；範例學生在同訓練方式下的 B2 把「架構的貢獻」從 15 點拆成 5–7 點。結果出來後發現分不開的，問使用者加跑。
- **計時排在所有訓練之後**，前後記 `nvidia-smi --query-compute-apps`；章節順序因此要配合（HW13 把推論時間從 ch02 移到 ch06）。
- **`os._exit(0)` 前要先 `sys.stdout.flush()`**；persistent DataLoader workers 在 Python 3.12 結束時會卡住，工具用 `os._exit` 收尾。
- **等待條件別寫錯**：`pgrep -f`／`pkill -f` 會比對到自己那條命令列；舊記錄檔裡的 FAILED 會讓 `grep -q FAILED` 立刻成立。
- **JetBrains Mono 有連字**：程式裡的 `!=` 會被畫成 ≠，樣板要 `font-variant-ligatures: none`。
- 樣板與工具在 `docs/tools/hw13_book/`（由 HW14 複製改色、改字型；`charts.py` 的 `task_lines` 預設改成 False）；`fill_listings.py` 也能引用 `.venv` 裡第三方套件的原始碼（說明文字寫完整路徑）。

## HW07 的調整（2026-10-09～10）

HW07 由同一個本機 session 寫完（不冷讀；授權同 HW13：每章寫完驗證就推、規格內的 GPU 實驗直接一組一組跑）。另外學到這些：
- **repo 裡的程式已被改過時，先問主線要用哪一版**：HW07 的 train.py 已換成 luhua large、3 epoch；使用者決定還原成官方範例當主線，舊結果（Kaggle 分數）當史料。被刪掉的 API（`transformers.AdamW`）原樣抄成 `legacy_adamw.py`，不要直接換成新 API——預設值不同就會改變結果。
- **GPU 上固定種子不等於可重現**：BERT 訓練跑兩次 100 步就分岔（SDPA 注意力的反向傳播，以及開關打開時被默默替換、不會警告的運算）。拆檔比對、工具驗證都在 `torch.use_deterministic_algorithms(True)`（加 `CUBLAS_WORKSPACE_CONFIG=:4096:8`）下做，用一個不改原檔的 wrapper（`docs/tools/hw07_det.py`）。代價約 2%。
- **先量種子雜訊**：範例設定下 3 個種子差 6.6 個百分點，比很多改良還大；之後每個比較都 3 個種子，差距小於雜訊不下結論。學習率調小後雜訊也變小。
- **「同一個 checkpoint、只改評估」的實驗很便宜，先做**：HW07 的後處理與評估 stride 不必重訓，就把範例從 0.416 拉到 0.63，而且揭露了訓練資料的偏差（模型偏愛視窗中央）。評估工具要批次化到 GPU 上（每個視窗在 Python 迴圈裡建 193×193 的遮罩，慢到不能用）。
- **用預測驗證解釋**：「小 stride 有效是因為中央偏好」這個解釋，事先寫下可檢驗的預測（隨機視窗的模型從小 stride 得到的好處應該很小），再跑實驗驗證。
- **對照組要和被比較的設定同一套訓練方式**：轉小寫的對照用了範例設定（不衰減），D 組用線性衰減，結果只能和不衰減的比，章節裡要明講這個缺陷。
- **換成已經微調過的模型時，先量零樣本**：ckiplab、luhua 不訓練就有 0.72、0.67；它們的 `[CLS]`（「沒有答案」）會讓範例的後處理大量回答「[CLS]」。
- **背景佇列只起一個，並確認**：HW07 有一個以為已死的 `bash -c` 等待程序其實活著，queue2 被兩個 runner 同時執行了 5 個多小時（結果因決定性模式兩份完全相同，但計時全部作廢、速度慢一倍）。起背景工作一律寫成腳本檔、用 `setsid nohup` 起，起完立刻 `ps -eo pid,ppid,cmd | grep run_grid` 確認只有一個；不要用 `pkill -f` 殺自己同名的程序（會殺到自己的 shell）。
- **計時排在最後、GPU 上不能有其他工作**：推論分析和訓練並行會讓訓練慢 2–4 倍；jsonl 裡被干擾的 train_s 要在 FACTS 註明不可用，乾淨的計時另外排一組。
- 樣板與工具在 `docs/tools/hw07_book/`（由 HW13 複製、改沙金色）；`make_charts.py` 依 `<!-- CHART:名稱 -->` 標記把 SVG 放進頁面，可重跑。
