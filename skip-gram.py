from nltk.tokenize import RegexpTokenizer
import pandas as pd
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator,Vocab
from torch.utils.data import DataLoader
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import random

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

class NegativeSamplingLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,input_vectors,output_vectors,noise_vectors):
    # losses=[]
    # for i in range(input_vector.size(0)):
    #   curr_loss=-(torch.log(torch.sigmoid(torch.dot(input_vector[i],output_vector[i])))+torch.sum(torch.log(torch.sigmoid(torch.mm(torch.neg(noise_vectors[i]), (input_vector[i]).unsqueeze(1))))))
    #   losses.append(curr_loss)
    # return sum(losses)/len(losses)
    batch_size, embed_size = input_vectors.shape
    input_vectors = input_vectors.view(batch_size, embed_size, 1)
    output_vectors = output_vectors.view(batch_size, 1, embed_size)
    out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
    noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
    noise_loss = noise_loss.squeeze().sum(1)
    return -(out_loss + noise_loss).mean()

class SkipGramNeg(torch.nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist, vocab):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        self.vocab=vocab

        self.in_embed = torch.nn.Embedding(n_vocab, n_embed)
        self.out_embed = torch.nn.Embedding(n_vocab, n_embed)
        
        # # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings
    
    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings
    
    def forward_noise(self, inp, target, n_samples):
        batch_size=inp.size(0)
        noise_words = torch.multinomial(input=self.noise_dist,num_samples=batch_size*n_samples,replacement = False)
        noise_words = noise_words.to(device)
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        return noise_vectors
    
class Dataset_SGNS(Dataset):
  def __init__(self, train_dt):
    self.training_pairs=train_dt
  def __len__(self) -> int:
    return len(self.training_pairs)
  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.training_pairs[index][0],self.training_pairs[index][1]

model = SkipGramNeg(len(vocab_sgns),EMBEDDING_SIZE,torch.tensor(neg_sampling_weights), vocab_sgns)
model=model.to(device)
criterion = NegativeSamplingLoss()
optimizer = torch.optim.Adam(model.parameters(),lrate)
train_dataset=Dataset_SGNS(training_pairs)
train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

for epoch in range(EPOCHS):
    total_loss=0
    model.train()
    for inputs, targets in train_dataloader:
      inputs = inputs.to(device)
      targets = targets.to(device)
      embedded_input_words = model.forward_input(inputs)
      embedded_target_words = model.forward_target(targets)
      embedded_noise_words = model.forward_noise(inputs,targets,n_samples=NEG_SAMPLES)
      loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
      total_loss+=loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {total_loss}')

word_embeddings=model.in_embed.weight.data
context_embeddings=model.out_embed.weight.data
embeddings_sgns=torch.cat((word_embeddings,context_embeddings),dim=1)
torch.save(embeddings_sgns,'skip-gram-word-vectors.pt')