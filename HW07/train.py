import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, AutoModelForQuestionAnswering, BertTokenizer
from tqdm.auto import tqdm
from dataset import QA_Dataset


"""## Training"""

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)
fp16_training = False

# if fp16_training:
#     !pip install accelerate==0.2.0
#     from accelerate import Accelerator
#     accelerator = Accelerator(fp16=True)
#     device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/


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
    device = "cuda"
    num_epoch = 3
    validation = True
    logging_step = 100
    learning_rate = 1e-4
    train_batch_size = 32

    # original, score = 0.49495
    # model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # 1, ckiplab, score = 0.57160
    # model = AutoModelForQuestionAnswering.from_pretrained('ckiplab/bert-base-chinese-qa').to(device)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # 2, score = 0.60508
    model = AutoModelForQuestionAnswering.from_pretrained('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large').to(device)
    tokenizer = BertTokenizerFast.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")

    # 3, score = 0.53206
    # model = AutoModelForQuestionAnswering.from_pretrained('luhua/chinese_pretrain_mrc_macbert_large').to(device)
    # tokenizer = BertTokenizerFast.from_pretrained('luhua/chinese_pretrain_mrc_macbert_large')

    # 4, score = 0.55990
    # model = AutoModelForQuestionAnswering.from_pretrained('wptoux/albert-chinese-large-qa').to(device)
    # tokenizer = BertTokenizerFast.from_pretrained('wptoux/albert-chinese-large-qa')

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_questions, train_paragraphs = read_data("hw7_train.json")
    dev_questions, dev_paragraphs = read_data("hw7_dev.json")

    """## Tokenize Data"""

    # Tokenize questions and paragraphs separately
    # 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                          add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                        add_special_tokens=False)

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)

    # You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)

    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()

    print("Start Training ...")
    best_acc_rate = 0
    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####


            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                                   attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]

                dev_acc_rate = dev_acc / len(dev_loader)
                print(f"Validation | Epoch {epoch + 1} | dev_acc_rate = {dev_acc_rate:.3f} | best_acc_rate = {best_acc_rate:.3f}")
                if dev_acc_rate > best_acc_rate:
                    best_acc_rate = dev_acc_rate
                    # Save a model and its configuration file to the directory 「saved_model」
                    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
                    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
                    print(f"Saving Model ... at epoch {epoch + 1}")
                    model_save_dir = "saved_model"
                    model.save_pretrained(model_save_dir)

            model.train()




