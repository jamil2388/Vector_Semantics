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
3. Filter SimLex vocabulary to match the words present in the target corpus C
4. We vectorize the corpora using TF-TDF and Word2Vec baselines (v)
5. We compare the performance of these vectorizers using the gold standard (G) SimLex-999
6. For each baseline v, we compute the topk similar words of word w based on the cosine similarity of vectors for w and the other word (the vectors given by baseline v)
7. We evaluate using NDCG metric made available by eval() function
8. To calculate NDCG, for each w in G, we rank the Gold results from SimLex as Descending order of the position of the similar words to w (e.g : [10 9 8 7 6 5 4 3 2 1]). This is the relevance score of the Gold standard
9. Then for each w in the similar words in C, we compare and find the words that are in the gold similar list and put the relevance score of that word in the required position. If a similar word is absent in the similar list of G, we put 0 as predicted relevance score
10. We feed pairwise relevance scores to Pytrec_Eval (https://pypi.org/project/pytrec-eval-terrier/) to calculate the NDCG metric for us


Link : https://github.com/jamil2388/Vector_Semantics