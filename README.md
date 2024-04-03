# Vector_Semantics
This project studies the comparison of vector semantic methods with gold standard dataset SimLex-999 and reports multiple evaluation results


### Project Setup

```
# create the virtualenv vsm, python3.10 recommended
python3.10 -m virtualenv vsm

# on windows
vsm\Scripts\activate

# on linux
source vsm/bin/activate

# install from requirements.txt
pip install -r requirements.txt

# download the required corpora
python -m nltk.downloader brown

# run vectorize.py to calculate ndcg for all combinations of baselines at one go
cd src
python -u vectorize.py

```

### Experiments and Evaluation

1. Preproces SimLex-999 (The gold standard corpus G) to prepare topk (k = 10) ranked words for each word in the corpus 
2. Prepare Brown corpus (C) (initially genre = 'news' and 'romance') for vectorizing
3. We vectorize the corpora using TF-TDF and Word2Vec baselines (v)
4. We compare the performance of these vectorizers using the gold standard (G) SimLex-999
5. For each baseline v, we compute the topk similar words of word w based on the cosine similarity of vectors for w and the other word (the vectors given by baseline v)
6. We evaluate using NDCG metric made available by eval() function


Link : https://github.com/jamil2388/Vector_Semantics