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
