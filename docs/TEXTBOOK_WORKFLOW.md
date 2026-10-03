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
