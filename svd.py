from nltk.tokenize import RegexpTokenizer
import pandas as pd
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
import torch
import warnings
import scipy as sp

UNK_CUTOFF=3
UNKNOWN_TOKEN='<unk>'
WINDOW_SIZE=5
BATCH_SIZE=128
EMBEDDING_SIZE_SVD=300
PAD_TOKEN='<pad>'
NUM_LABELS=4
HIDDEN_SIZE=128
lrate=1e-3
EPOCHS=10

df=pd.read_csv('data/train.csv')
train_labels=df['Class Index'].tolist()
df=df['Description']
warnings.filterwarnings("ignore")
sentences=[]
for sent in df:
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sent)
    tokens=[token.lower() for token in tokens]
    sentences.append(tokens)

def replace_low_frequency_words(sentences, threshold=UNK_CUTOFF):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    replaced_sentences = [
        [UNKNOWN_TOKEN if word_counts[word] < threshold else word for word in sentence]
        for sentence in sentences
    ]
    return replaced_sentences
sentences=replace_low_frequency_words(sentences)
vocab_svd=build_vocab_from_iterator(sentences, specials=[UNKNOWN_TOKEN,PAD_TOKEN])
vocab_svd.set_default_index(vocab_svd[UNKNOWN_TOKEN])
co_occurrence_matrix=sp.sparse.lil_matrix((len(vocab_svd),len(vocab_svd)))

for word_list in sentences:
        for i, word in enumerate(word_list):
            center_index = vocab_svd[word]
            context_indices = [vocab_svd[word_list[j]] for j in range(max(0, i - WINDOW_SIZE), min(len(word_list), i + WINDOW_SIZE + 1)) if i!=j]
            for context_index in context_indices:
                co_occurrence_matrix[center_index,context_index] += 1
U,_,_=sp.sparse.linalg.svds(co_occurrence_matrix, EMBEDDING_SIZE_SVD, return_singular_vectors='u', which='LM')
embeddings_SVD=U

torch.save(embeddings_SVD,'svd-word-vectors.pt')