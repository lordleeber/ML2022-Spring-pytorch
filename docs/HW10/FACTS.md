# HW10 事實清單（不進教材；寫每一章前先讀）

所有數字都在本機（RTX PRO 4000 Blackwell、torch 2.11.0+cu128、pytorchcv 0.0.74）以決定性模式量得（`docs/tools/hw10_det.py`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`），除非另外註明。

## 環境與拆檔（Phase 0）
- notebook `HW10/HW10.ipynb` 拆成 `config.py`（cells 3、5、6）、`dataset.py`（8）、`attack.py`（10、12、14；mifgsm 的 TODO 已補）、`ensemble.py`（24；TODO 補成 logits 相加）、`hw10.py`（8、16、18、20、22）、`report.py`（28、30、32；JPEG TODO 已補）。
- 參照版：`docs/tools/hw10_make_ref.py`（跳過 cell 24、26：ensembleNet 是語法錯誤）。`%cd` → `os.chdir`、`!tar` → `subprocess.run`；`!pip`、`!wget`、`!unzip`、`!rm` 註解掉。
- **預設模式不可重現**：參照版跑兩次，fgsm/ 有 28 張 PNG 不同（cuDNN 卷積反向）；兩次印出的 fgsm_loss 2.48256／2.48250，ifgsm_acc 0.01000／0.00500、ifgsm_loss 17.48673／17.61542。另一次（第一次跑參照版）fgsm_acc 0.59000、ifgsm 0.00500／17.61329。決定性模式下兩次完全相同。
- 兩次一般模式（ref1、ref2）的差異：fgsm/ 28 張圖、共 32 個像素不同（每個差 16，即 +8 變 −8）；ifgsm/ 195 張圖、402,938 個像素不同（最大差 16）。
- `.tgz`：fgsm.tgz 508,434 bytes、ifgsm.tgz 483,326 bytes（決定性參照版）；tar 列出 210 項（10 個目錄 + 200 張），順序是檔案系統順序（不是排序）。
- 決定性模式：hw10.py 與參照版的 fgsm/、ifgsm/ 逐位元相同；`hw10_exp.py`（`--attack fgsm|ifgsm`）與 hw10.py 逐位元相同。
- 參照版（含 imgaug 失敗前）整支約 15 秒（非決定性、非乾淨計時）。
- pytorchcv 0.0.74 裝進 .venv；權重在 `~/.torch/models`（下載自 github.com/osmr/imgclsmob releases），全部 70 個共 2.5 GB。
- imgaug 0.4.0：NumPy 2.5.3 下 `import imgaug` 失敗（`np.sctypes` was removed in the NumPy 2.0 release）；需要 cv2（opencv 5 與 imgaug 0.4.0 搭配的 uv 解析結果是 opencv-python 5.0.0.93；本書用 opencv-python-headless<5）與 shapely。shim：`docs/tools/hw10_npshim_sitecustomize.py`。

## 主線結果（決定性模式，resnet110_cifar10 當代理 = 白箱）
- `number of images = 200`
- `benign_acc = 0.95000, benign_loss = 0.22679`
- `fgsm_acc = 0.59500, fgsm_loss = 2.48333`
- `ifgsm_acc = 0.00500, ifgsm_loss = 17.49289`
- 存成 PNG 再讀回（`hw10_eval_dir.py`）：data 0.95000／0.22679；fgsm 0.59000／2.49183；ifgsm 0.00500／17.28956；L∞ 都是 8。
- report.py：`benign: dog2.png  dog: 99.64%`、`adversarial: dog2.png  cat: 72.20%`、`JPEG adversarial: dog2.png  dog: 99.24%`。

## 模型（hw10_facts.txt）
- resnet110_cifar10（`pytorchcv/models/resnet_cifar.py` 的 `CIFARResNet`）：參數 1,730,714（手算相同），buffers 8,399。
- init_block 464；stage1 84,096（18 個 unit）；stage2 330,048（18）；stage3 1,315,456（18）；final_pool 0；output（Linear 64→10）650。
- 111 個 Conv2d（109 個 3×3、2 個 1×1 shortcut：stage2、stage3 的 unit1 `identity_conv`，stride 2 + BN）、111 個 BN、1 個 Linear。
- 形狀（batch 8）：init (8,16,32,32) → stage1 (8,16,32,32) → stage2 (8,32,16,16) → stage3 (8,64,8,8) → AvgPool2d(8) (8,64,1,1) → output (8,10)。

## 資料與 ε
- 200 張，10 類各 20；類別順序 = 資料夾字母排序 = CIFAR-10 的類別順序（airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck）。
- 檔名字串排序：airplane1、airplane10、airplane11、airplane12…；dog/dog2.png 在第 111 個（0 起算）。
- ε（正規化後）= 8/255/std = [0.155310, 0.157651, 0.156082]（std 0.202、0.199、0.201）；8/255 = 0.031373。
- 正規化後 200 張圖的範圍：每通道最小 [-2.4307, -2.4221, -2.2239]、最大 [2.5198, 2.6030, 2.7512]。
- 四捨五入與 clamp（resnet110）：FGSM 614,400 個像素中 13,754 個落在 [0,255] 外；浮點準確率 0.595、clamp＋四捨五入後 0.590。I-FGSM 7,628 個在範圍外；0.005／0.005。
- PNG 的 |adv − benign| 分布：FGSM 600,646 個是 8、4,305 個是 0（其餘 1–7 是被 clamp 的像素）；I-FGSM：0 有 89,045、1 有 1,575、2 有 154,965、3 有 124,722、4 有 982、5 有 91,500、6 有 65,068、7 有 18,636、8 有 67,907（步長 0.8 的倍數四捨五入後集中在 2、3、5、6）。

## 模型庫（hw10_zoo.jsonl）
- pytorchcv 85 個 `*_cifar10`；70 個可載入，benign 準確率 0.915（nin）到 0.98（preresnet542bn、wrn40_8）。
- 載不到 15 個：13 個下載失敗（resnext20_16x4d、resnext20_32x2d、resnext20_32x4d、seresnet1001、seresnet1202、sepreresnet1001、sepreresnet1202、msdnet22、fractalnet、diaresnet1001、diaresnet1202、diapreresnet1001、diapreresnet1202），2 個沒有預訓練權重（resdropresnet20、shakedropresnet20）。
- 代理池 40 個（nin、resnet*、preresnet*、seresnet*、sepreresnet*、densenet*、xdensenet*）；受害者 8 個：wrn28_10、wrn40_8、pyramidnet110_a48、resnext29_32x4d、ror3_110、rir、shakeshakeresnet26_2x32d、diaresnet56；其餘 22 個受害者家族的模型不使用。

## 試跑（hw10_pilot.jsonl，非決定性模式）
- resnet110 單一代理，8 受害者平均（無防禦／JPEG70）：FGSM 0.627／0.641；I-FGSM 0.492／0.633；MI-FGSM 0.341／0.647；DIM-MI 0.259／0.596。
- 6 模型 ensemble（resnet20、preresnet56、seresnet56、densenet40_k12_bc、nin、resnet1001）：I-FGSM 0.057／0.583；DIM-MI 0.061／0.507。攻擊時間約 81 秒。

## JPEG
- imgaug `JpegCompression(compression=70)` → PIL quality = round(1 + 99·(1 − 70/101)) = 31。`hw10_exp.py` 的 `jpeg()` 與 imgaug 在 fgsm/ 的 200 張 × 壓縮率 10、50、70、90 共 800 組逐位元相同。
- resnet110 白箱，受害者端加 JPEG70：FGSM 0.66、I-FGSM 0.665（verify 時量的）。

## 實驗 A（hw10_runs.jsonl；8 受害者平均，V＝無防禦、Vj＝JPEG70、ens＝受害者 logits 相加）
- **JPEG70 對乾淨圖的代價很大**：clean（不攻擊）V 0.962 → Vj 0.664；受害者 ensemble 0.985 → 0.735。resnet110 白箱 clean 0.95。
- 見 docs/tools/hw10_runs.jsonl；第一批摘要：fgsm ε=1/2/4/8/16 → V 0.891/0.842/0.778/0.626/0.267；I-FGSM 步數 1/2/5/10/20/50/100 → V 0.891/0.814/0.759/0.589/0.485/0.446/0.441（白箱 0.735/0.585/0.270/0.030/0.005/0/0）；步長 0.2/0.4/0.8/1.6/3.2/8 → V 0.757/0.619/0.485/0.452/0.442/0.465。

## CIFAR-10 與作業圖片（hw10_overlap.py，2026-10-10 下載 cifar-10-python.tar.gz，torchvision 驗證 md5）
- 作業的 200 張全部出自 **CIFAR-10 測試集**，逐像素相同、標籤全部一致；訓練集 0 張。正好是測試集裡每一類的前 20 張（索引 0–254 之間）。
- 所以自己在 CIFAR-10 訓練集上訓練代理不會「看過」這 200 張；pytorchcv 的模型也一樣（它們在訓練集上訓練，在測試集上報準確率）。
- 印出 vs PNG（hw10_ch01.txt，實驗 A 全部 97 組單一代理攻擊）：41 組不同，最多差 0.02；PNG 較高 25 組、較低 16 組。
- FGSM 白箱的浮點像素分類（hw10_ch01.txt）：梯度剛好為 0 的像素 0 個；被剪成完全沒改 4,305；部分剪掉（1–7）9,449；滿 8 有 600,646。原圖像素 = 0 有 2,633、= 255 有 5,920。transform 來回最大誤差 1.53e-5，四捨五入後完全還原。忘了 /std 的 ε 換回像素 = 1.616、1.592、1.608。
- ch02（hw10_ch02.txt，resnet110 白箱、浮點圖）：t=0/1/2/4/8/16/32 → FGSM 方向 acc 0.950/0.735/0.665/0.660/0.595/0.305/0.135、loss 0.227/1.631/2.029/2.292/2.483/3.793/5.246、一次近似預測 loss 0.227/2.306/4.385/8.543/16.858/33.489/66.752；隨機 ±1 方向 acc 0.950/0.950/0.945/0.935/0.845/0.560/0.195。FGSM8：190 張答對 → 119 張仍對（騙倒 71），錯→對 0。騙成：狗→貓 11、飛機→鳥 6、飛機→貓 4、鹿→鳥 4、汽車→青蛙 3、貓→狗 3、馬→貓 3。
- 實驗 A 單一代理（40 個）：FGSM 黑箱（8 受害者平均）0.574–0.718，平均 0.652；I-FGSM 0.362（nin）–0.681（densenet40_k12_bc），平均 0.540。I-FGSM 黑箱與 log(參數量) 的相關 −0.16、與原圖準確率 −0.10；FGSM 黑箱與 I-FGSM 黑箱的相關 0.30；FGSM 白箱與 FGSM 黑箱 0.82。
- ch04（hw10_ch04.txt，resnet110 I-FGSM 步長 0.8，PNG 等價圖）：步數 0/1/2/3/5/10/15/20/30/40/50/70/100 → 白箱 0.95/0.735/0.585/0.445/0.27/0.03/0.015/0.005/0.005/0.005/0/0/0；8 受害者 0.9625/0.8912/0.8137/0.8306/0.7594/0.5894/0.5187/0.4856/0.46/0.4475/0.4462/0.4431/0.4406；受害者 ensemble 0.985/0.905/0.825/0.845/0.755/0.625/0.52/0.46/0.44/0.42/0.445/0.435/0.445；受害者平均 loss 0.184→3.277；在 ±8 邊界上的像素比例 0（≤5 步）/0.024/0.051/0.111/0.159/0.189/0.210/0.236/0.258。每步剪回合法範圍（20 步）：受害者 0.4881、ensemble 0.465。隨機起點（PGD，3 種子，20 步）：受害者 0.5400/0.5356/0.5350，白箱 0.02/0.01/0.01。
- 實驗 A 步長（20 步）：0.2/0.4/0.8/1.6/3.2/8 → 白箱 0.10/0.015/0.005/0.005/0.01/0.06；受害者 0.757/0.619/0.485/0.452/0.442/0.465；ensemble 0.80/0.635/0.46/0.445/0.415/0.475；L∞ 4/8/8/8/8/8。步數 1/2/5 的 L∞ 是 1/2/4。
- JPEG 強度掃描（hw10_jpeg.jsonl；8 受害者平均／受害者 ensemble；rate→quality：10→90、20→80、30→71、40→61、50→51、60→41、70→31、80→22、90→12）：
  - clean：無 0.962/0.985；q90 0.926/0.945；q80 0.901/0.930；q71 0.876/0.920；q61 0.819/0.875；q51 0.803/0.870；q41 0.747/0.805；q31 0.664/0.735；q22 0.567/0.605；q12 0.403/0.455。
  - fgsm_eps8：0.626 → q90 0.682、q80 0.731、q71 0.769、q61 0.768、q51 0.762、q41 0.726、q31 0.641、q22 0.566、q12 0.418。
  - ifgsm_it20：0.485 → q90 0.792、q80 0.841、q71 0.806、q61 0.769、q51 0.772、q41 0.718、q31 0.651、q22 0.551、q12 0.401（ensemble q80 0.865）。
  - ifgsm_it100：0.441 → q90 0.780、q80 0.812、q31 0.639。single_ifgsm_nin：0.362 → q90 0.506、q80 0.603、q51 0.670、q31 0.563。fgsm_eps16：0.267 → q90 0.427、q80 0.407、q71 0.437、q51 0.614、q31 0.588。
- BPDA 試跑（scratch，非正式）：jpeg70+resnet110 當代理的 I-FGSM，resnet110+JPEG70 剩 0.255（不知道 JPEG 的範例 I-FGSM 是 0.665）；同批圖直接給 resnet110（無 JPEG）0.70。
- ensembleNet（hw10.py --models nin,resnet20,preresnet20 --attacks ifgsm）與 hw10_exp.py 的 ens_trio_logits_sum 逐位元相同；印出 benign 0.95000／0.22468、ifgsm 0.00000／40.18109。

## 實驗 B（ensemble，I-FGSM；V＝8 受害者平均）
- trio（nin、resnet20、preresnet20）：logits_sum V 0.174／ens 0.120；logits_mean 0.164／0.100；prob_mean 0.174／0.145；loss_sum 0.196／0.160；FGSM（logits_sum）0.474／0.505。三個成員單獨 I-FGSM 的 V 平均 0.471、最好 0.362（nin）。
- 隨機抽 k 個（seed 10）：k2 → 0.224/0.362/0.201；k4 → 0.169/0.098/0.134；k8 → 0.039/0.033/0.039；k16（logits_sum）→ 0.805/0.834/0.888 **失效**。
- **k16 logits_sum 失效**：159／165／178 張圖完全沒改動（其他組 0 張）。hw10_ch05_zero.txt：ens_k16_d0 的 16 個模型 logits 相加，原圖上 159 張的輸入梯度全為 0、正確類別與第二名的 logit 差距中位數 93.7、197 張 max softmax 在 float32 剛好 = 1.0；改成 logits 平均：0 張梯度為 0、差距中位數 5.9。改用 mean／prob／loss_sum 重跑（hw10_grid_B2.txt，排在 U 之後）。
- 家族：resnet20/56/110/164bn → 0.096／ens 0.060；跨家族 resnet56、preresnet56、seresnet56、densenet40_k12_bc → 0.116／0.060。
- 這些組的 attack_s 與代理訓練並行，不乾淨（k8 約 415–500 秒、k16 700–1,230 秒）。

## 實驗 U（paper B；hw10_train.jsonl、hw10_ch05_u.txt）
- 訓練：pytorchcv 的 resnet20／resnet56 架構從頭訓練，CIFAR-10 訓練集 50,000 張（無需排除，作業圖片都在測試集），SGD lr 0.1、momentum 0.9、wd 5e-4、batch 128、RandomCrop 4 + flip，60 epoch，第 30、45 epoch 學習率 ×0.1；3 個種子。最終測試準確率（10,000 張）：resnet20 0.9081/0.9114/0.9100，resnet56 0.9241/0.9243/0.9238。單次 resnet20 約 9.5 分、resnet56 約 18–21 分（與攻擊並行，非乾淨計時）。
- 每個 checkpoint 當代理跑 I-FGSM（範例設定），8 受害者平均（3 種子平均 [範圍]）：
  - resnet20 e1/2/3/5/10/15/20/30/40/45/60：0.879/0.841/0.794/0.706/0.650/0.631/0.610/0.539/**0.435** [0.424–0.450]/0.463/0.485；測試準確率 0.511/0.572/0.620/0.718/0.746/0.755/0.750/0.738/0.897/0.898/0.910。
  - resnet56：0.924/0.873/0.817/0.744/0.663/0.613/0.532/0.522/**0.438** [0.422–0.463]/0.499/0.588；測試準確率 0.371/0.536/0.607/0.685/0.734/0.749/0.796/0.763/0.911/0.908/0.924。
  - 對照 pytorchcv 預訓練：resnet20 0.535、resnet56 0.502。
- 最好的是第 40 epoch（第一次降學習率之後 10 個 epoch、第二次之前）；繼續訓練到 60 反而變差（resnet56 0.438 → 0.588）。很早期（e1–10）的 checkpoint 很差。
- 實驗 C（MI／DIM，代理 resnet110，V／ens）：mi decay 0/0.5/1/2 → 0.485/0.415/0.331/0.513（decay 0 與 I-FGSM 逐位元相同）；mi 10/50 步 → 0.419/0.359。DIM-I-FGSM p 0.25/0.5/0.75/1.0（3 種子平均 [範圍]）→ 0.385 [0.372–0.406]/0.336 [0.327–0.342]/0.367 [0.363–0.371]/0.427 [0.413–0.445]。DIM-MI p0.5 → 0.266 [0.256–0.278]（ens 0.240）；p1.0 → 0.333 [0.319–0.342]。DIM-MI dim_max 34/36/40/48 → 0.277/0.266/0.261/0.270。fgsm_eps1.6 與 fgsm_eps2 的 PNG 完全相同。
- B2（16 個代理重跑，乾淨 GPU）：logits_mean d0/d1/d2 → V 0.035/0.032/0.041（ens 0.035/0.030/0.040；白箱成員平均 0.025/0.023/0.034），attack_s 572/317/686；d0 prob_mean 0.059（ens 0.050）、loss_sum 0.071（ens 0.065）。k8 平均 0.037 → k16 0.036：平台。
- ens_k16_d0（logits_sum 失效）逐圖：159 張未改動的圖每個受害者都答對（wrn28_10、resnet1001 皆 159/159），41 張改動的只有 2 張仍答對 → 161/200 = 0.805。
- ch07 雜訊在 JPEG 之後（hw10_ch07.txt；0–255 單位；高頻＝32×32 FFT 中央 8×8 以外的能量比例）：
  - ifgsm_it20：delta RMS 4.354、高頻 0.923；q90 → RMS 5.325、高頻 0.876、cos 0.367；q80 → 5.543、0.856、0.223；q31 → 5.701、0.761、0.080。
  - fgsm_eps8：RMS 7.928、高頻 0.943；q90 cos 0.548；q80 0.349；q31 0.103。
  - single_ifgsm_nin：RMS 5.822、高頻 0.882；q90 cos 0.541；q80 0.385；q31 0.157。
  - dim_mi_p0.5_s0：RMS 6.760、高頻 0.909；q90 0.472；q80 0.304；q31 0.111。
  - JPEG 對原圖本身的誤差 RMS：q90 5.813、q80 7.581、q31 12.344。
  - 結論：JPEG 後的差異沒有變小（甚至變大），但和原本雜訊的方向相關很低；投影片的選項 (b)「把雜訊減少」只對了一半。
- 報告題與 eval（hw10_ch07_eval.txt，每次新載入模型）：pytorchcv get_model 回傳 training=True。原圖 dog2：train 模式 dog 40.45%、eval dog 99.64%；fgsm/dog2：train 模式 dog 30.17%（沒被騙！）、eval cat 72.20%。一次 train 模式前向（no_grad 下）就讓 init_block BN 的 running_mean 最多變 0.0849，之後 eval 的原圖 dog2 變成 dog 99.06%。

## 實驗 C2／C3（組合，logits 平均；V＝8 受害者、q90/q80/q31＝受害者前加 JPEG，hw10_jpeg.jsonl）
- resnet110：I-FGSM V 0.485／q90 0.792／q80 0.841／q31 0.651；DIM-MI（3 種子）0.266／0.586／0.681／0.604。
- 自訓 6（resnet20、56 各 3 種子第 40 epoch）：I-FGSM 0.169／0.336／0.512／0.566；MI 0.204／0.288／0.430／0.546；DIM-MI 0.204 [0.194–0.209]／0.272／0.350 [0.342–0.361]／0.483。白箱都 0。
- 預訓練 8（ens_k8_d1 的成員）：I-FGSM 0.048／0.589／0.716／0.610；MI 0.079／0.359／0.545／0.601；DIM-MI 0.121 [0.117–0.129]／0.352／0.523／0.567。白箱 0.019／0.049／0.078–0.096。
- 自訓 6 ＋ 預訓練 8：I-FGSM 0.014／0.264／0.492／0.554；MI 0.013／0.175／0.349／0.549；DIM-MI 0.016 [0.014–0.019]／0.158／0.274 [0.268–0.284]／0.462。attack_s 約 325–327 秒（C2 未與其他工作並行）。
- 原圖：0.962／0.926／0.901／0.664。

## 實驗 D（BPDA；hw10_bpda.py）
- jpeg20（q80）+resnet110 當代理 I-FGSM：代理自己（含 JPEG）0.225；受害者 V 0.664／q90 0.683／q80 **0.582**／q31 0.607（不知道 JPEG 的 I-FGSM：0.485／0.792／0.841／0.651）。
- jpeg70（q31）+resnet110：代理自己 0.255；受害者 V 0.755／q90 0.754／q80 0.799／q31 **0.519**。
- jpeg20+自訓 6，MI：V 0.520／q90 0.452／q80 **0.362**／q31 0.502（無 BPDA 的 MI：0.204／0.288／0.430／0.546）。DIM-MI s0：0.497／0.457／**0.351**／0.409（無 BPDA DIM-MI s0：0.194／0.266／0.342／0.502）。
