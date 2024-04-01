'''
This file is required to prepare the simlex corpus
We preprocess the corpus to show us the topk most similar words for each
word in simlex word1 column based on the rating in column SimLex999

If a word has less than k top similar words, we infer the others based
on transitivity property of -
a is similar to b
b is similar to c
therefore, a is similar to c

'''

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np
import nltk
import pickle
import os
from nltk.corpus import brown

simlex_filepath = '../data/SimLex-999.txt'
simlex_preprocessed_filepath = '../data/simlex_prep.pkl'
cols = ["word1", "word2", "SimLex999"]  # the columns to fetch from SimLex


def load_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def save_file(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# load SimLex and return the df with the relevant columns
def preprocess_simlex(filepath, cols):

    df = pd.read_csv(filepath, sep="\t", usecols=cols) # Read the file into a DataFrame, selecting only the required columns
    vocab = pd.Series(list(set(df['word1'].tolist() + df['word2'].tolist()))) # Step 1: Extract unique words from columns word1 and word2
    sorted_df = df.sort_values(by='SimLex999', ascending=False) # Sort DataFrame by rating SimLex999 ratings in descending order
    simlex_vocab = pd.DataFrame({'word': vocab, 'topk' : pd.Series(), 'sim_rating' : pd.Series()})

    return sorted_df, simlex_vocab

# create a column which contains the top 10 most similar word indices
# the dataframe should already be sorted
def add_topk(df, simlex_vocab, k=10):
    for row_id, row in simlex_vocab.iterrows():

        word = row['word']
        similar_words, ratings = calculate_topk(df, word)
        # update the values in the particular row_id
        simlex_vocab.loc[row_id, 'topk'] = list(similar_words) # first initialization of values
        simlex_vocab.loc[row_id, 'sim_rating'] = list(ratings) # first initialization of values

        if len(row['topk']) < 10:
            word2 = word
            current_topk = row['topk'].copy() # we need to iterate through a static list, and update the row['topk'] dynamically
            for word1 in current_topk:
                similar_words, ratings = calculate_topk(df, word1, word2)

                if similar_words is not None and ratings is not None :
                    # update the values in the particular row_id
                    simlex_vocab.loc[row_id, 'topk'].extend(list(similar_words))  # first initialization of values
                    simlex_vocab.loc[row_id, 'sim_rating'].extend(list(ratings))  # first initialization of values

        # truncate the list to topk words
        simlex_vocab.loc[row_id, 'topk'] = simlex_vocab.loc[row_id, 'topk'][:10]
        simlex_vocab.loc[row_id, 'sim_rating'] = simlex_vocab.loc[row_id, 'sim_rating'][:10]

    return simlex_vocab

# returns the most similar words to word1 having word2 absent in all of the cases (because we already populated that one)
def calculate_topk(df, word1, word2 = None):

    # Filter rows where word1 or word2 matches the current word
    filtered_df = df.loc[(df['word1'] == word1) | (df['word2'] == word1)] if not word2 else df[((df['word1'] == word1) & (df['word2'] != word2)) | ((df['word2'] == word1) & (df['word1'] != word2))]

    if len(filtered_df) == 0: return None, None

    # Take top 10 most similar words and their ratings
    top_10_similar_words = filtered_df.head(10)[['word1', 'word2', 'SimLex999']]
    # Convert DataFrame rows to a list of tuples (word, rating)
    similar_word_rating_pairs = [
        (row['word1'], row['SimLex999']) if row['word2'] == word1 else (row['word2'], row['SimLex999']) for
        index, row in top_10_similar_words.iterrows()]

    # Separate similar words and ratings into two lists
    similar_words, ratings = zip(*similar_word_rating_pairs)

    return similar_words, ratings

# lazy load the simlex preprocessed dataframe
def prepare_simlex(filepath):

    data = load_file(filepath)
    if data is not None: return data

    simlex_df, simlex_vocab = preprocess_simlex(simlex_filepath, cols) # simlex_df = original df, simlex_vocab = unique words and their topk similar words and ratings
    simlex_vocab = add_topk(simlex_df, simlex_vocab, 10)  # append new cols topk and sim_rating based showing lists of topk words to the unique word in col 'word'
    save_file(simlex_vocab, filepath if filepath else simlex_preprocessed_filepath)

    return simlex_vocab

if __name__ == '__main__':
    prepare_simlex(simlex_preprocessed_filepath)