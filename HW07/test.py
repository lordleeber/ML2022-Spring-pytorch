import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm

from dataset import read_data, QA_Dataset
from postprocess import evaluate

device = "cuda"

model = BertForQuestionAnswering.from_pretrained("saved_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

test_questions, test_paragraphs = read_data("hw7_test.json")
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output, tokenizer))

result_file = "result.csv"
with open(result_file, 'w') as f:
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")
