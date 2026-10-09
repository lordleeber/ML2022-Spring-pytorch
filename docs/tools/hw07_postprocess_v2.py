# A readable version of the improved post-processing in docs/tools/hw07_post.py (rule valid_len_off).
# Same inputs as HW07/postprocess.py evaluate, plus what is needed to cut the answer from the paragraph.
# usage: cd HW07 && PYTHONPATH=. python ../docs/tools/hw07_postprocess_v2.py <saved_model> [doc_stride]
import torch


def evaluate_v2(data, output, paragraph, encoding, question_len, doc_stride, max_answer_len=30):
    # paragraph: the paragraph string; encoding: its tokenizer output (has .offsets)
    # question_len: [CLS] + question tokens + [SEP], i.e. where the paragraph part of every window begins
    best_score, answer = float('-inf'), ''
    for k in range(data[0].shape[1]):
        n = int(data[1][0][k].sum()) - 1                 # paragraph tokens in window k (token_type 1, minus the last [SEP])
        lo, hi = question_len, question_len + n          # the paragraph part of the window is [lo, hi)
        s = output.start_logits[k][lo:hi]
        e = output.end_logits[k][lo:hi]
        pair = s[:, None] + e[None, :]                   # pair[i, j] = start score of i + end score of j
        ones = torch.ones_like(pair, dtype=torch.bool)
        ok = ones.triu() & ~ones.triu(max_answer_len)    # i <= j < i + max_answer_len
        pair = pair.masked_fill(~ok, float('-inf'))
        score, idx = pair.flatten().max(0)
        if score > best_score:
            best_score = score
            i, j = divmod(int(idx), n)
            first, last = k * doc_stride + i, k * doc_stride + j   # token positions in the whole paragraph
            answer = paragraph[encoding.offsets[first][0]:encoding.offsets[last][1]]
    return answer


if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from transformers import BertForQuestionAnswering, BertTokenizerFast
    from dataset import read_data, QA_Dataset
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = BertForQuestionAnswering.from_pretrained(sys.argv[1]).to("cuda").eval()
    dev_questions, dev_paragraphs = read_data("hw7_dev.json")
    dev_questions_tokenized = tokenizer([q["question_text"] for q in dev_questions], add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    dev_set.doc_stride = stride
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to("cuda"), token_type_ids=data[1].squeeze(dim=0).to("cuda"),
                           attention_mask=data[2].squeeze(dim=0).to("cuda"))
            q = dev_questions[i]
            qlen = 2 + min(dev_set.max_question_len, len(dev_questions_tokenized[i].ids))
            ans = evaluate_v2(data, output, dev_paragraphs[q["paragraph_id"]], dev_paragraphs_tokenized[q["paragraph_id"]], qlen, stride)
            correct += ans == q["answer_text"]
    print(f"stride {stride}: dev EM = {correct / len(dev_loader):.5f} ({correct}/{len(dev_loader)})")
