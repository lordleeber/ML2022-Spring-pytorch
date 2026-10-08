# HW06 教材計畫

HW06（Anime Face Generation，用 GAN 從 100 維雜訊產生 64×64 的動漫臉）的 HTML 教材。流程照 [docs/TEXTBOOK_WORKFLOW.md](../TEXTBOOK_WORKFLOW.md)，特別是「HW02／HW03／HW04 的調整」三節（本機一章一章寫、事實一次量完、只在使用者明講「推」時才推）；這份檔案只記 HW06 特有的事。量到的數字放在同一個目錄的 `FACTS.md`。

## 狀態（2026-10-07）

- **實驗全部跑完**（2026-10-07 11:00–19:12）：baseline 計時、工具逐位元驗證、grid 四組（gan／wgan／wgangp／wgan_sigmoid）、StyleGAN2 50,000 步，全部評估完，數字都在 FACTS.md。下一步：寫大綱（outline.html）、問使用者大綱與樣式，等核可。
- 疑點全部有結論（見 FACTS）：1 TypeError 實測、2 FID／AFD 已量、3 detach 快 15% 且結果相同、5 只在 n_critic 不整除 10 時發生（推論）、7 確認、8 test.py 兩次輸出不同、10 workers 2 以上不再變快。另量了：只換 loss_G（會恢復，epoch 20 FID 122.9）、clip KeyError、FID 種子／樣本數敏感度、1000 張 tgz 約 1.1–1.2 MB。
- checkpoint（grid、sg2）只在 scratchpad，session 結束就沒了；重跑：grid 每組 20–45 分、sg2 4 時 18 分。
- 使用者決定（2026-10-07）：
  - FID 用 **pytorch-fid 0.3.0**（Inception 權重 `pt_inception-2015-12-05-6726825d.pth`，sha256 `6726825d0af5…`），裝在 .venv 之外。
  - AFD 用 **nagadomi/lbpcascade_animeface**（OpenCV cascade，sha256 `9376d30a…`）近似；JudgeBoi 的偵測器沒有公開，教材要註明只能比相對高低。
  - 實驗：DCGAN／WGAN／WGAN-GP 各跑原規格 100 epoch、報告第 2 題的「各層梯度範數」（clipping vs GP），**再加 StyleGAN2**（lucidrains `stylegan2-pytorch` 1.9.0）。
- **大綱核可**（2026-10-08）：index、ch00–ch08（`outline.html`）。原本照 HW04 列了 appendix，2026-10-08 依 completed-repo-to-html-textbook 的規定（「不寫附錄」）改成：名詞對照與速查併入 ch08（8.8、8.9），不產出 appendix.html；要介紹 Crypko 是什麼（ch01）；**不冷讀**；Crypko 原圖也放一些（PNG，`docs/HW06/img/`）。下一步：寫 index、ch00，一章一停。
- 工具：`docs/tools/hw06_exp.py`（train.py 的複製＋變體）、`hw06_eval.py`（FID／AFD）、`hw06_run_grid.sh`。

## 投影片重點（`HW06/Machine Learning HW6.pdf`，26 頁，只讀了文字）

- p.5：輸入亂數、輸出動漫臉；實作要求 DCGAN & WGAN & WGAN-GP；產生 1000 張。
- p.6–7：評估指標 FID（Inception 特徵的 Fréchet 距離）、AFD（動漫臉偵測率，越高越好）。
- p.10：Crypko，71,314 張；**允許額外資料**（但 p.21 又說 Do NOT use additional data or pre-trained models，前後矛盾）。
- p.12、p.14：1000 張 `<number>.jpg`、tar 成 .tgz、**小於 2MB**；只有 public、自己選一次提交。
- p.16 基準線：Simple FID ≤ 30000／AFD ≥ 0；Medium ≤ 12000／≥ 0.4；Strong ≤ 10000／≥ 0.5；Boss ≤ 9000／≥ 0.6。**FID 的數量級（上萬）和一般 FID（幾十）不同**，JudgeBoi 的算法未公開，不能拿本書的 FID 去對基準線。
- p.17 建議：Simple 跑範例（<1 小時）、Medium 多跑 epoch（1–1.5 小時）、Strong WGAN 或 WGAN-GP（2–3 小時）、Boss StyleGAN（<5 小時）。
- p.19 報告：1. 列出 WGAN 與 GAN 至少兩點差異；2. 畫「Gradient norm」：用訓練資料、判別器至少 4 層，兩種設定（weight clipping、gradient penalty），Y 軸梯度範數（log）、X 軸判別器第幾層（低到高）。
- p.24：WGAN = 拿掉 Sigmoid、loss 不取 log、權重 clip 到常數（投影片寫「1 ~ -1」，論文是 0.01）、RMSProp 或 SGD；WGAN-GP = 用 GP 取代 clipping、在內插影像上算梯度。
- p.25 StyleGAN：z → w，w 用在不同解析度。

## 原版 notebook vs 本 repo

- notebook 一格一段；本 repo 拆成 config.py、dataset.py、utils.py（get_dataset、same_seeds、weights_init）、generator.py、discriminator.py、trainer_gan.py、train.py、test.py。
- `config.py`：`n_epoch` **1 → 100**（notebook 是 1）。
- `trainer_gan.py` 的 `inference`：`n_generate` **1000 → 100**（作業要交 1000 張；照跑 test.py 只會產生 100 張）。
- 訓練中每個 epoch 的 matplotlib 顯示被註解掉；train.py 的「Show the image」也註解掉。
- test.py：從寫死的 2022 checkpoint 路徑改成「最新的 `checkpoints/*_GAN/G_*.pth`」（`381aad0`）。
- 其餘（seed 2022、batch 64、lr 1e-4、Adam betas (0.5, 0.999)、z_dim 100、模型）相同。

## 已經看到的疑點（Phase 0 要實測確認）

1. **`gp()` 沒有實作**（`trainer_gan.py:84`，`def gp(self): pass`），而註解要的是 `self.gp(r_imgs, f_imgs)`：簽名對不上，照註解改會直接 TypeError。使用者筆記「WGAN, WGAN-GP 只替換 loss_G 是不能使用的」——WGAN 還要拿掉 Sigmoid、clip、換 optimizer。
2. **沒有任何評估指標**：訓練只印 loss_D／loss_G，GAN 的 loss 不代表圖的好壞；要用 FID／AFD 對每個存檔評估。
3. **判別器把 f_imgs 的梯度也算回 G**：D 的那一步 `f_imgs = self.G(z)` 沒有 `detach()`，`loss_D.backward()` 會順便算 G 的梯度（之後被 `G.zero_grad()` 清掉），白算。
4. **Sigmoid + BCELoss**：數值上不如 `BCEWithLogitsLoss`；D 太強時 loss_G 會飆高（baseline 中途 loss_G 到 15–17、loss_D 0.03）。
5. **n_critic > 1 時印的 loss_G 是舊的**（只有 G 更新的那一步才重算）。
6. **形狀註解錯**：discriminator.py 的 `(batch, 3, 32, 32)` 等應是 64/128/256/512 個 channel；generator.py 的 `feature_dim * 16` 也不對。
7. **checkpoint 編號**：`G_{e}.pth` 用 0 起算（G_0 = 第 1 個 epoch 後、G_99 = 第 100 個），sample 圖 `Epoch_{epoch+1:03d}.jpg` 用 1 起算。
8. **test.py 沒有設種子**：每次推論出來的 100 張不同；`TrainerGAN.__init__` 又會先建一次 G、D、z_samples（消耗亂數）。
9. **glob 沒排序**：`get_dataset` 的檔名順序是檔案系統的順序，不同機器 shuffle 出來的 batch 會不同（同一台可重現，要實測）。
10. **`num_workers=2`**：每 epoch 1,115 步約 18 秒，資料讀取可能是瓶頸（read_image → ToPILImage → Resize，兩個 worker 各吃 70–80% CPU）。
11. `Variable`、`.data` 是舊寫法；`device` 用 `.cuda()` 寫死是刻意的，不列為問題。
12. 投影片的 FID 基準線數量級和 pytorch-fid 不同（見上），AFD 偵測器不同。

## HW06 特有的注意事項

- 資料：`HW06/faces/` 71,314 張 96×96 RGB JPEG（560 MB），檔名 0.jpg–71313.jpg；被 `.gitignore` 排除。來源 Crypko（Arvin Liu 收集），授權沒寫；使用者決定原圖放少量。
- `HW06/` 裡已有 2026-10-03 的一次 100 epoch 訓練（logs、checkpoints、output），2026-10-07 又跑一次計時。
- 生成圖都是 JPEG；根目錄 `.gitignore` 排除 `*.jpg`，要放進書裡時加 `!docs/HW06/img/*.jpg` 或轉 PNG。
