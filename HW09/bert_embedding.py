import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from sklearn.metrics import pairwise_distances
from transformers import BertModel, BertTokenizerFast


"""# Homework 9 - Explainable AI (Part 2 BERT)

## Embedding Analysis (Q28~30)
Compare the embedding of 蘋 (or 果) across sentences where 蘋果 is the fruit or the company.
The figure is saved to output/bert_embedding.png.
"""

output_dir = './output/'

# The Colab notebook downloads taipei_sans_tc_beta.ttf to draw Traditional Chinese.
# Locally any CJK font works; set FONT_PATH to the one you have.
FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'


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


# Sentences for visualization
sentences = []
sentences += ["今天買了蘋果來吃"]
sentences += ["進口蘋果（富士)平均每公斤下跌12.3%"]
sentences += ["蘋果茶真難喝"]
sentences += ["老饕都知道智利的蘋果季節即將到來"]
sentences += ["進口蘋果因防止水分流失故添加人工果糖"]
sentences += ["蘋果即將於下月發振新款iPhone"]
sentences += ["蘋果獲新Face ID專利"]
sentences += ["今天買了蘋果手機"]
sentences += ["蘋果的股價又跌了"]
sentences += ["蘋果押寶指紋辨識技術"]


"""### TODO
This is the only part you need to modify to answer Q28~30.
"""

# Index of word selected for embedding comparison. E.g. For sentence "蘋果茶真難喝", if index is 0, "蘋 is selected"
# The first line is the indexes for 蘋; the second line is the indexes for 果
select_word_index = [4, 2, 0, 8, 2, 0, 0, 4, 0, 0]
# select_word_index = [5, 3, 1, 9, 3, 1, 1, 5, 1, 1]

# The notebook leaves these two as `return 0` for students to fill in
def euclidean_distance(a, b):
    # Compute euclidean distance (L2 norm) between two numpy vectors a and b
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    # Compute cosine similarity between two numpy vectors a and b
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Metric for comparison. Choose from euclidean_distance, cosine_similarity
METRIC = euclidean_distance

def get_select_embedding(output, tokenized_sentence, select_word_index):
    # The layer to visualize, choose from 0 to 12
    LAYER = 12
    # Get selected layer's hidden state
    hidden_state = output.hidden_states[LAYER][0]
    # Convert select_word_index in sentence to select_token_index in tokenized sentence
    select_token_index = tokenized_sentence.word_to_tokens(select_word_index).start
    # Return embedding of selected word
    return hidden_state[select_token_index].numpy()


if __name__ == "__main__":
    same_seeds(0)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(FONT_PATH):
        # Latin glyphs come from DejaVu Sans, the missing CJK glyphs fall back to FONT_PATH
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=FONT_PATH).get_name()]

    model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True).eval()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    # Tokenize and encode sentences into model's input format
    tokenized_sentences = [tokenizer(sentence, return_tensors='pt') for sentence in sentences]

    # Input encoded sentences into model and get outputs
    with torch.no_grad():
        outputs = [model(**tokenized_sentence) for tokenized_sentence in tokenized_sentences]

    # Get embedding of selected word(s) in sentences. "embeddings" has shape (len(sentences), 768), where 768 is the dimension of BERT's hidden state
    embeddings = [get_select_embedding(outputs[i], tokenized_sentences[i], select_word_index[i]) for i in range(len(outputs))]

    # Pairwise comparison of sentences' embeddings using the metric defined. "similarity_matrix" has shape [len(sentences), len(sentences)]
    similarity_matrix = pairwise_distances(embeddings, metric=METRIC)

    ##### Plot the similarity matrix #####
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix)  # Display an image in the plot
    plt.colorbar()  # Add colorbar to the plot
    plt.yticks(ticks=range(len(sentences)), labels=sentences)  # Set tick locations and labels (sentences) of y-axis
    plt.title('Comparison of BERT Word Embeddings')  # Add title to the plot
    for (i, j), label in np.ndenumerate(similarity_matrix):  # np.ndenumerate is 2D version of enumerate
        plt.text(i, j, '{:.2f}'.format(label), ha='center', va='center')  # Add values in similarity_matrix to the corresponding position in the plot
    path = os.path.join(output_dir, 'bert_embedding.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')
