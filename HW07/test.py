import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertTokenizer, AlbertForQuestionAnswering
from tqdm.auto import tqdm
from dataset import QA_Dataset

"""## Testing"""

"""## Function for read_data"""
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


"""## Function for Evaluation"""
def evaluate(data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])

    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ', '')


if __name__ == "__main__":
    print("Evaluating Test Set ...")

    device = "cuda"
    # model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    model = AlbertForQuestionAnswering.from_pretrained("saved_model").to(device)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizerFast.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
    # tokenizer = BertTokenizerFast.from_pretrained('luhua/chinese_pretrain_mrc_macbert_large')
    # tokenizer = BertTokenizerFast.from_pretrained('wptoux/albert-chinese-large-qa')

    test_questions, test_paragraphs = read_data("hw7_test.json")

    """## Tokenize Data"""

    # Tokenize questions and paragraphs separately
    # 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                         add_special_tokens=False)

    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    # You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device),
                           token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output))

    result_file = "result.csv"
    with open(result_file, 'w') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',', '')}\n")

    print(f"Completed! Result is in {result_file}")
