from nltk.tokenize import RegexpTokenizer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator,Vocab
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
import numpy as np
import torch
import scipy as sp
from torch.utils.data import Dataset
import random
import sys

UNK_CUTOFF=3
UNKNOWN_TOKEN='<unk>'
WINDOW_SIZE=5
BATCH_SIZE=128
EMBEDDING_SIZE=150
EMBEDDING_SIZE_SGNS=300
PAD_TOKEN='<pad>'
NUM_LABELS=4
HIDDEN_SIZE=128
lrate=1e-3
NEG_SAMPLES=4
EPOCHS=10
THRESHOLD=1e-5

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
vocab_sgns=build_vocab_from_iterator(sentences, specials=[UNKNOWN_TOKEN,PAD_TOKEN])
vocab_sgns.set_default_index(vocab_sgns[UNKNOWN_TOKEN])

def count_word_occurrences(list_of_lists):
    word_count = Counter()
    for inner_list in list_of_lists:
        word_count.update(inner_list)
    return word_count
word_counts=count_word_occurrences(sentences)
int_to_vocabword={}
for i,w in enumerate(vocab_sgns.get_itos()):
    if w!=PAD_TOKEN:
        int_to_vocabword[i]=w
words=[w for sen in sentences for w in sen]
word_counts = Counter(words)
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # descending freq order
int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
int_words = [vocab_to_int[word] for word in words]

device = "mps" if torch.cuda.is_available() else "cpu"
device = torch.device("mps")

def subsample_sentences(words, sentences, threshold = THRESHOLD):
    word_counts = Counter(words)
    total_n_words = len(words)
    freq_ratios = {word: count/total_n_words for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freq_ratios[word]) for word in word_counts}
    return [[word for word in sentence if random.random() < (1 - p_drop[word])] for sentence in sentences]
train_sens = subsample_sentences(words,sentences)

training_pairs = set()
for sentence in train_sens:
    for i, target_word in enumerate(sentence):
        context = [sentence[j] for j in range(max(0, i - WINDOW_SIZE), min(len(sentence), i + WINDOW_SIZE + 1)) if j != i]
        for context_word in context:
            training_pairs.add((torch.tensor(vocab_sgns[target_word]), torch.tensor(vocab_sgns[context_word])))
training_pairs=list(training_pairs)

word_counts = Counter(words)
word_freqs = np.array([word_counts[word] / sum(word_counts.values()) for word in word_counts.keys()])
neg_sampling_weights = torch.from_numpy(word_freqs ** 0.75 / np.sum(word_freqs ** 0.75))
before = neg_sampling_weights[:1]
after =neg_sampling_weights[1:]
new_tensor = torch.cat((before, torch.tensor(0).unsqueeze(0), after))
neg_sampling_weights=new_tensor

arranged_values = [0]*len(vocab_sgns)
temp=list(word_counts.keys())
temp.insert(1,PAD_TOKEN)
for a,b in zip(temp,neg_sampling_weights):
    arranged_values[vocab_sgns[a]]=b
neg_sampling_weights=arranged_values

embeddings_sgns=torch.load('skip-gram-word-vectors.pt')
embeddings_sgns = torch.tensor(embeddings_sgns, device='cpu')

df=pd.read_csv('data/test.csv')
test_labels=df['Class Index'].tolist()
df=df['Description']
test_sentences=[]
for sent in df:
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sent)
    tokens=[token.lower() for token in tokens]
    test_sentences.append(tokens)

class Dataset_LSTM(Dataset):
    def __init__(self, sent, labs, embeddings, vocabulary):
        self.sentences = sent
        self.labels = labs
        self.vocabulary = vocabulary
        self.embeddings=embeddings
    def __len__(self) -> int:
        return len(self.sentences)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        word_embeddings=[self.embeddings[self.vocabulary[j]] for j in self.sentences[index]]
        return torch.stack(word_embeddings), torch.tensor(torch.nn.functional.one_hot(torch.tensor(self.labels[index]-1), num_classes=NUM_LABELS)).float()
    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        sentences = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        padded_labels=padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(0))
        return padded_sentences, padded_labels

train_dataset=Dataset_LSTM(sentences,train_labels,embeddings_sgns,vocab_sgns)
test_dataset=Dataset_LSTM(test_sentences,test_labels,embeddings_sgns,vocab_sgns)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate)
test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=test_dataset.collate)

class LSTMModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = torch.nn.Linear(hidden_dim, num_classes)
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2label(lstm_out[-1])
        tag_scores = torch.softmax(tag_space, dim=1)
        return tag_scores
    
model = LSTMModel(EMBEDDING_SIZE_SGNS, HIDDEN_SIZE, NUM_LABELS)
model=model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for batch_sentences, batch_labels in train_dataloader:
        (batch_sentences, batch_labels) = (batch_sentences.to(device), batch_labels.to(device))
        outputs = model(batch_sentences.permute(1,0,2))
        loss = loss_fn(outputs, batch_labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

torch.save(model,'skip-gram-classification.pt')

model.eval()
predictions=[]
true_vals=[]
with torch.no_grad():
    for words, tags in train_dataloader:
        (words, tags) = (words.to(device), tags.to(device))
        pred = model(words.permute(1,0,2))
        pred_max_index = torch.argmax(pred, dim=1)+1
        true_vals.extend((torch.argmax(tags, dim=1)+1).cpu())
        predictions.extend(pred_max_index.cpu())
predictions=torch.stack(predictions).numpy()
true_vals=torch.stack(true_vals).numpy()
print('Evaluation Metrics for train set :')
print(f'Accuracy Score: {accuracy_score(true_vals,predictions)}')
print(f'F1_Score (Macro): {f1_score(true_vals,predictions, average='macro')}')
print(f'F1_Score (Micro): {f1_score(true_vals,predictions, average='micro')}')
print(f'Precision Score: {precision_score(true_vals,predictions, average='weighted')}')
print(f'Recall Score: {recall_score(true_vals,predictions, average='weighted')}')
print(f'Confusion Matrix:\n {confusion_matrix(true_vals,predictions)}')

model.eval()
predictions=[]
true_vals=[]
with torch.no_grad():
    for words, tags in test_dataloader:
        (words, tags) = (words.to(device), tags.to(device))
        pred = model(words.permute(1,0,2))
        pred_max_index = torch.argmax(pred, dim=1)+1
        true_vals.extend((torch.argmax(tags, dim=1)+1).cpu())
        predictions.extend(pred_max_index.cpu())
predictions=torch.stack(predictions).numpy()
true_vals=torch.stack(true_vals).numpy()
print('Evaluation Metrics for test set :')
print(f'Accuracy Score: {accuracy_score(true_vals,predictions)}')
print(f'F1_Score (Macro): {f1_score(true_vals,predictions, average='macro')}')
print(f'F1_Score (Micro): {f1_score(true_vals,predictions, average='micro')}')
print(f'Precision Score: {precision_score(true_vals,predictions, average='weighted')}')
print(f'Recall Score: {recall_score(true_vals,predictions, average='weighted')}')
print(f'Confusion Matrix:\n {confusion_matrix(true_vals,predictions)}')