import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from transformers import BertForQuestionAnswering, BertTokenizerFast


"""# Homework 9 - Explainable AI (Part 2 BERT)

## Attention Visualization (Q21~24)
Use https://exbert.net/exBERT.html directly in the browser.

## Embedding Visualization (Q25~27)
We have a pre-trained model which is fine-tuned for QA.
Solving QA requires 4 steps: (steps are NOT in order)
1. Clustering similar words together (based on relation of words in context)
2. Answer extraction
3. Clustering similar words together (based on meaning of words)
4. Matching questions with relevant information in context

Can you find out the functionalities of each layer just by looking into the embedding of hidden states?

The Colab notebook loads the TA's tokenizer and saved hidden states from hw9_bert.zip,
but that Google Drive file is gone (404). If hw9_bert/ exists it is still used;
otherwise the hidden states are computed with a public BERT fine-tuned on SQuAD,
so the figures will differ from the slides.
Figures are saved to output/bert_q{QUESTION}_layer{N}.png.
"""

output_dir = './output/'
hw9_bert_dir = './hw9_bert/'
qa_model_name = 'deepset/bert-base-cased-squad2'


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


contexts, questions, answers = [], [], []

# Question 1
contexts += ["Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a Serbian American inventor, electrical engineer, \
            mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current \
            (AC) electricity supply system."]
questions += ["In what year was Nikola Tesla born?"]
answers += ["1856"]

# Question 2
contexts += ['Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, Singapore and other countries. \
            It requires the pupil to remain in school at a given time in the school day (such as lunch, recess or after school); or even to attend \
            school on a non-school day, e.g. "Saturday detention" held at some schools. During detention, students normally have to sit in a classroom \
            and do work, write lines or a punishment essay, or sit quietly.']
questions += ['What is a common punishment in the UK and Ireland?']
answers += ['detention']

# Question 3
contexts += ['Wolves are afraid of cats. Sheep are afraid of wolves. Mice are afraid of sheep. Gertrude is a mouse. Jessica is a mouse. \
            Emily is a wolf. Cats are afraid of sheep. Winona is a wolf.']
questions += ['What is Emily afraid of?']
answers += ['cats']


def visualize(Tokenizer, model, QUESTION):
    # Tokenize and encode question and paragraph into model's input format
    inputs = Tokenizer(questions[QUESTION-1], contexts[QUESTION-1], return_tensors='pt')

    # Get the [start, end] positions of [question, context] in encoded sequence for plotting
    question_start, question_end = 1, inputs['input_ids'][0].tolist().index(102) - 1
    context_start, context_end = question_end + 2, len(inputs['input_ids'][0]) - 2

    if model is None:
        outputs_hidden_states = torch.load(f"{hw9_bert_dir}output/model_q{QUESTION}")
    else:
        with torch.no_grad():
            outputs_hidden_states = model(**inputs).hidden_states

    ##### Traverse hidden state of all layers #####
    # "outputs_hidden_state" is a tuple with 13 elements, the 1st element is embedding output, the other 12 elements are attention hidden states of layer 1 - 12
    for layer_index, embeddings in enumerate(outputs_hidden_states[1:]):  # 1st element is skipped

        # "embeddings" has shape [1, sequence_length, 768], where 768 is the dimension of BERT's hidden state
        # Dimension of "embeddings" is reduced from 768 to 2 using PCA (Principal Component Analysis)
        reduced_embeddings = PCA(n_components=2, random_state=0).fit_transform(embeddings[0])

        fig = plt.figure(figsize=(12, 10))
        ##### Draw embedding of each token #####
        for i, token_id in enumerate(inputs['input_ids'][0]):
            x, y = reduced_embeddings[i]  # Embedding has 2 dimensions, each corresponds to a point
            word = Tokenizer.decode(token_id)  # Decode token back to word
            # Scatter points of answer, question and context in different colors
            if word in answers[QUESTION-1].split():  # Check if word in answer
                plt.scatter(x, y, color='blue', marker='d')
            elif question_start <= i <= question_end:
                plt.scatter(x, y, color='red')
            elif context_start <= i <= context_end:
                plt.scatter(x, y, color='green')
            else:  # skip special tokens [CLS], [SEP]
                continue
            plt.text(x + 0.1, y + 0.2, word, fontsize=12)  # Plot word next to its point

        # Plot "empty" points to show labels
        plt.plot([], label='answer', color='blue', marker='d')
        plt.plot([], label='question', color='red', marker='o')
        plt.plot([], label='context', color='green', marker='o')
        plt.legend(loc='best')  # Display the area describing the elements in the plot
        plt.title('Layer ' + str(layer_index + 1))  # Add title to the plot
        path = os.path.join(output_dir, f'bert_q{QUESTION}_layer{layer_index + 1}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    print(f'saved {output_dir}bert_q{QUESTION}_layer{{1..12}}.png')


if __name__ == "__main__":
    same_seeds(0)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(hw9_bert_dir):
        Tokenizer = BertTokenizerFast.from_pretrained(f"{hw9_bert_dir}Tokenizer")
        model = None
    else:
        Tokenizer = BertTokenizerFast.from_pretrained(qa_model_name)
        model = BertForQuestionAnswering.from_pretrained(qa_model_name, output_hidden_states=True).eval()

    # The original notebook picks one QUESTION (1, 2 or 3); here all three are drawn
    for QUESTION in [1, 2, 3]:
        visualize(Tokenizer, model, QUESTION)
