## Introduction to NLP

Files included in the submission: 
- Source code files: svd.py, skip-gram.py, svd-classification.py, skip-gram-classification.py

- Pretrained models and embedding files: svd-word-vectors.pt, skip-gram-word-vectors.pt, svd-classification-model.pt, skip-gram-classification-model.pt

- Jupyter notebooks containing code for generating embeddings and evaluating them on downstream task for both methods: SVD.ipynb, skip-gram.ipynb

Execute any of the .py files by the command:
> python3 filename.py

Load the pretrained models or word embeddings in code using:
> embeddings/model=torch.load('filename.pt')

Hyper-parameter tuning was done by changing the value of variable named 'WINDOW_SIZE' in the .py files and running them to generate embeddings and perform the downstream task.

I have performed subsampling using the method given in the paper: <href>https://arxiv.org/pdf/1310.4546.pdf</href>. The value of parameter t that I chose is 1e-5. Also, I have performed negative sampling (4 negative samples per positive sample) by creating a probability distribution based on frequency of occurence of words.

Link to One drive for assignment: </href>https://iiitaphyd-my.sharepoint.com/:u:/g/personal/akshit_sharma_students_iiit_ac_in/ETruU87RVIRMo_aTcM6vp58B8Pe1MfCFbJAM6sRU5RLuuA?e=AHBpbe </href>
