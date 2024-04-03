'''

Create vectors of words from given corpus
using methods like 'tfidf', 'word2vec' etc.

Initially we are using brown corpus (news and romance genre)

We will evaluate in the same file

'''



from nltk.corpus import brown
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from simlex import prepare_simlex
import pytrec_eval
import pandas as pd
import numpy as np
import re
import time

# Preprocess the corpus
def preprocess_text(text):
    text = ' '.join(text) # Join the tokens into a single string
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove unnecessary characters
    return text.lower()

# break each sentence of the corpus into tokenized list of words
def tokenize(c):
    # each sentence can be a single string
    for i, sent in enumerate(c):
        sent = sent.replace('  ', ' ')
        c[i] = sent.split(' ')
    return c

def extract_corpus(corpus, category):
    c = []
    word_count = 0
    words = set()
    for i, sent in enumerate(corpus.sents(categories=[category])):
        # if i > 10 : break
        c.append((preprocess_text(sent)))
        word_count += len(c[i].replace('  ', '').split())
        words.update(c[i].replace('  ', '').split())

    print(f'Total number of sentences in {category} : {len(c)}')
    print(f'Total number of words in {category} : {word_count}')
    print(f'Total number of unique words in {category} : {len(words)}')

    return c


def tfidf_train(c):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(c) # Transform the corpus into TF-IDF vectors
    print("Shape of TF-IDF matrix:", tfidf_vectors.shape)
    return vectorizer, tfidf_vectors.transpose() # now each row would represent each word vector

# Function to get top k similar words for a given word
def get_top_k_similar_words(word_index, sims_matrix, vocab, k):
    # Get the row of cosine similarities for the given word
    word_sims = sims_matrix[word_index]
    # Sort indices of similarities in descending order
    sorted_indices = np.argsort(word_sims)[::-1]
    # Get top k similar words
    top_k_words = [vocab[i] for i in sorted_indices[1:k+1]]  # Exclude the word itself
    return top_k_words

# extract vocab, index and the vectors from the model and store it in a dataframe
def index_vectors(model, vectors, tfidf = 1):
    df = pd.DataFrame()

    if tfidf:
        vocab_dict = model.vocabulary_
        df['vocab'] = vocab_dict.keys()
        df['index'] = vocab_dict.values()
        df['vector'] = None # Create a new column 'vector' in df
        # Iterate over each row in df
        for index, row in df.iterrows():
            # Get the index from the row
            index_val = row['index']
            # Check if the index exists in the tfidf_vectors matrix
            if index_val in vocab_dict.values():
                vector = np.asarray(vectors[index_val].todense()).squeeze()
                # Assign the vector to the 'vector' column in df
                df.at[index, 'vector'] = vector
    else:
        # w2v already has all the mappings based on word -> vector key values
        df['vocab'] = list(model.wv.key_to_index.keys())
        df['index'] = pd.Series([i for i in range(len(df['vocab']))])
        df['vector'] = pd.Series([vector for vector in model.wv.vectors])

    sims = cosine_similarity(df['vector'].tolist(), df['vector'].tolist())

    # Create a new column 'topk' in df
    df['topk'] = None

    # Iterate over each row in df
    for index, row in df.iterrows():
        # Get the top k similar words for the current word
        top_k_similar_words = get_top_k_similar_words(index, sims, df['vocab'], k=10)
        # Assign the top k similar words to the 'topk' column in df
        df.at[index, 'topk'] = top_k_similar_words

    return df

# returns the predicted ranking for a predicted topk list (single row)
# f_df = filtered df
# f_simlex = filtered simlex
# both f_df and f_simlex contain the same set of words
def calculate_ranking(f_df, f_simlex, f_simlex_index):
    f_df['rankings'] = None
    for r_id, row in f_df.iterrows():
        ranking = []
        word = row['vocab']
        simlex_row_id = f_simlex_index[word] # the row in f_simlex containing the same word as f_df
        simlex_topk = f_simlex.loc[simlex_row_id, 'topk'] # fetch the relevant gold topk
        for word in row['topk']:
            # Check if the word is present in simlex_vocab's topk
            if word in simlex_topk:
                # Get the position ranking of the word in simlex_vocab's topk
                rank = len(simlex_topk) - simlex_topk.index(word)
                ranking.append(rank)
            else:
                # If the word is not in simlex_vocab's topk, assign 0
                ranking.append(0)
        f_df.at[r_id, 'rankings'] = ranking
    return f_df['rankings']


# calculate average ndcg
def eval(Y, Y_, metrics={'ndcg_cut_10'}):
    qrel = dict();
    run = dict()
    print(f'Processing pytrec_eval input for {Y_.shape[0]} instances ...')
    from tqdm import tqdm
    with tqdm(total=Y_.shape[0]) as pbar:
        for i, (y, y_) in enumerate(zip(Y, Y_)):
            qrel['q' + str(i)] = {'d' + str(idx): val for idx, val in enumerate(y)}
            run['q' + str(i)] = {'d' + str(j): v for j, v in enumerate(y_)}
            pbar.update(1)
    print(f'Evaluating {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    df_mean = df.mean(axis=1).to_frame('mean')

    return df_mean

# run the required semantics operations based on the given corpus and vectors
# c = corpus, v = baseline
def run_semantics(c, v, settings, simlex_vocab):
        
    if v == 'tfidf':
        tfidf = 1
        model, vectors = tfidf_train(c)
    elif v == 'w2v':
        tfidf = 0
        if not isinstance(c[0], list): c = tokenize(c) # we need a specific tokenized format for training w2v
        model = Word2Vec(sentences = c, vector_size = settings['d'], window = settings['cw'], min_count = 1, workers = 4)
        vectors = model.wv.vectors
    else : return

    c_df = index_vectors(model, vectors, tfidf)  # create a dataframe containing the vocabulary, index and their tfidf vectors and the list of similar words
    filtered_words = [word for word in simlex_vocab['word'] if c_df['vocab'].str.contains(word).any()]  # we only work on words in the simlex_vocab, which are present in our brown corpus genres
    f_simlex_vocab = simlex_vocab[simlex_vocab['word'].isin(filtered_words)].reset_index(drop=True)  # Filter simlex_vocab based on the filtered words
    f_c_df = c_df[c_df['vocab'].isin(filtered_words)].reset_index(drop=True)  # filter f_c_df based on the same set of common words
    f_simlex_index = {word: index for index, word in enumerate(f_simlex_vocab['word'])}  # we need to work on only common words

    # return a new column rankings after calculating the topk of both dfs
    f_c_df['rankings'] = calculate_ranking(f_c_df, f_simlex_vocab, f_simlex_index)

    Y = [[(k - i) for i in range(k)] for _ in range(len(f_simlex_vocab['word']))]
    # calculate average ndcg
    ndcg_10 = eval(Y, f_c_df['rankings'])
    print(f'average ndcg_10 : {ndcg_10}')

    return ndcg_10['mean'].iloc[0]


if __name__ == '__main__':

    # c -> given corpus

    # brown_romance = extract_corpus(brown, 'romance')
    # print(c)
    # print(brown_romance)
    #
    # print("Raw data from the news genre of Brown corpus (first 100 characters):", brown.raw(categories=['news'])[:100])
    # print("Number of sentences in the news genre of Brown corpus:", len(brown.sents(categories=['news'])))
    # print("Number of sentences in the romance genre of Brown corpus:", len(brown.sents(categories=['romance'])))

    k = 10
    simlex_vocab = prepare_simlex('../data/simlex_prep.pkl') # contains col : 'word', 'topk'
    # simlex_vocab['rankings'] = pd.Series([[(k-i) for i in range(k)] for _ in range(len(simlex_vocab['word']))])
    # simlex_index = {word : index for word, index in enumerate(simlex_vocab['word'])} # now contains the index of the words in col 'word' of the df

    timings = []
    scores = []
    best_score = -1 # max possible is 1, lowest 0
    best_timing = 10000 # minutes
    best_baseline = {}

    categories = ['news', 'romance']

    for category in categories:
        print(f'')
        corpus = extract_corpus(brown, category)

        for v in ['tfidf', 'w2v']: # tfidf and w2v
            CW = [1, 2, 5, 10] if v == 'w2v' else [1]
            D = [10, 50, 100, 300] if v == 'w2v' else [1]
            for cw in CW:
                for d in D:
                    settings = {'e': 1000, 'cw': cw,
                                'd': d}  # context window size : {1, 2, 5, 10}, vector size : {10, 50, 100, 300}

                    start_time = time.time()

                    print(f'---------- Starting category {category}, Baseline : {v} ------------')

                    score = run_semantics(corpus, v, settings, simlex_vocab) # run the entire pipeline for this setting
                    end_time = time.time()
                    runtime = end_time - start_time

                    print(f'--------------------------------------------------------------------------------------------')
                    print(f'Time taken for Brown {category} corpus, Baseline {v} : {(runtime)/ 3600} mins')
                    print(f'--------------------------------------------------------------------------------------------')

                    if runtime < best_timing:
                        print(f'|||||||||||||||')
                        print(f'Best Time Recorded : {runtime}')
                        best_timing = runtime
                        best_baseline['v'] = v
                        best_baseline['cw'] = cw
                        best_baseline['d'] = d
                        print(f'\t .... For Settings : {best_baseline}')

                    if score > best_score:
                        print(f'|||||||||||||||')
                        print(f'Best Score Recorded : {score}')
                        best_score = score
                        best_baseline['v'] = v
                        best_baseline['cw'] = cw
                        best_baseline['d'] = d
                        print(f'\t .... For Settings : {best_baseline}')


    