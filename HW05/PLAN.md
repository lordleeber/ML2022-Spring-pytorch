# HW05 機器翻譯（英翻中 seq2seq）— 研究筆記與計畫

> 2026-10-08 由前一個 session 整理，尚未開始。總覽見 [docs/HW_STUDY_OVERVIEW.md](../docs/HW_STUDY_OVERVIEW.md)。價值很高，但**環境是最大障礙**。

## 題目（官方 `~/poyi/GitHubPublic/ML2022-Spring/HW05/HW05.pdf`，50 頁）
- 英翻繁中，資料 TED2020，已在 `DATA/rawdata/ted2020/`：`train_dev.raw.en/zh` 各 394,066 行（有中文，**可切驗證集量 BLEU**）；`test.raw.en` 4,000 行，`test.raw.zh` 是 `。` 佔位。
- 評分：BLEU。基準線：Simple 14.58（RNN seq2seq，1 小時）、Medium 18.04（Noam 學習率排程＋訓練更久，1 小時 40 分）、Strong 25.20（換 Transformer、調超參數，約 3 小時）、Boss 29.13（back-translation，> 12 小時）。
- 報告：1) 視覺化位置編碼兩兩的相似度並解釋；2) gradient clipping，畫各步的梯度範數，圈出兩處梯度爆炸。

## 本機狀態
- 13 個 .py 共 4,249 行：`hw05.py`、`hw5_ori.py`（notebook 轉出，各約 1,400 行）、`train.py` 419、`test.py` 209、`data_prepare.py`、`hw5_config.py`（RNN、max_epoch 60、max_tokens 8192、accum 2、Noam warmup 4000、beam 5…）、`rnn_encoder.py`、`rnn_decoder.py`、`attention_layer.py`、`seq_2_seq.py`、`noam_opt.py`、`labelsmooth_cross_entropy_criterion.py`、`hints.py`、`hw5_utils.py`。
- **大量依賴 fairseq**（`TranslationTask`、`iterators`、`FairseqEncoder/Decoder`、`MultiheadAttention`、beam search…）。共用 `.venv` 沒有 fairseq；memory 記錄課程指定的 fairseq @ `9a1c497` 在這台機器裝不起來，使用者先前決定跳過。commit `1bef93e`「Add HW05, it works on ubuntu」表示曾在別的環境跑過。
- fairseq 是 Meta FAIR 的序列建模工具包，已不活躍維護（後繼 fairseq2 與它不相容）；舊版依賴舊的 hydra／omegaconf／NumPy，且要編譯 C++/CUDA 擴充，和 Python 3.12＋torch 2.11（Blackwell 需要 cu128）衝突。

## 可能的路線（Phase 0 先花 1–2 小時試）
1. 另開獨立 venv，在新版 torch 上編譯舊 fairseq。
2. 去掉 fairseq，改寫成純 PyTorch（書就不再是讀原程式）。
3. 找一個能裝的 fairseq 版本並修相容問題。
→ 試不成功就放棄，改做 HW14／HW13。

## 為什麼值得
- Transformer encoder-decoder、自回歸生成、beam search、sentencepiece、位置編碼、warmup＋gradient clipping、back-translation（合成資料的前身），和 ML2025 HW3/HW4、ML2026 HW4（Training Transformer、位置編碼）直接相關。
