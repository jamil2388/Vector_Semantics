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
from nltk.corpus import brown

simlex_filepath = '../data/SimLex-999.txt'
cols = ["word1", "word2", "SimLex999"]  # the columns to fetch from SimLex

# load SimLex and return the df with the relevant columns
def preprocess_simlex(filepath, cols):
    # Read the file into a DataFrame, selecting only the required columns
    df = pd.read_csv(filepath, sep="\t", usecols=cols)
    return df

# create a dict which stores the index of the unique words in the given col
def create_word_index(df, col):
    d = {}
    for id, word in enumerate(df[col]):
        if not d.get(word, 0):
            d[word] = [id]
        else:
            d[word].append(id)
    print(len(d.values()))
    return d

# create a column which contains the top 10 most similar word indices
def calculate_topk(df, k=10):
    topk_indices = []
    for idx, row in df.iterrows():
        current_word = row["word1"]
        # Find the indices of top k most similar words based on SimLex999 ratings
        similar_indices = df[df["word1"] == current_word]["SimLex999"].nlargest(k).index.tolist()
        topk_indices.append(set(similar_indices))
    return topk_indices


# Function to sort each set in the 'topk' column based on 'SimLex999' values
def sort_topk_with_simlex(df, row):
    # List to store SimLex999 values corresponding to elements in 'topk'
    lst = [df.iloc[id]['SimLex999'] for id in row['topk']]
    # Sort 'topk' and lst based on descending order of lst
    sorted_pairs = sorted(zip(row['topk'], lst), key=lambda x: x[1], reverse=True)
    sorted_topk = [pair[0] for pair in sorted_pairs]
    return sorted_topk

# fill the 'topk' col with k values
def fill_topk(df, word2_index):
    col_topk = [{-1} for _ in range(len(df))] # create a col for final topk results, then append the col in the df
    for id in range(len(df)):
        row = df.iloc[id][:]
        bk_topk = row['topk']
        # skip if there are already k indices in this row
        if len(row['topk']) < 10:
            for id2 in bk_topk:
                word2 = df.iloc[id2]['word2'] # get the similar word from this row
                print(word2)
                row['topk'] = row['topk'].union(word2_index[word2])
            # sort the ids in the row['topk'] based on the scores in the corresponding col
            row['topk'] = sort_topk_with_simlex(df, row)
            for id_w1 in bk_topk:
                 col_topk[id_w1] = list(row['topk'])[:10] # filter out only the topk results and store it in topk col (col index = 3)
    col_topk = pd.Series(col_topk)
    col_topk.reset_index(drop = True, inplace = True)
    df['topk'] = col_topk
    return df

def prepare_simlex():

    simlex_df = preprocess_simlex(simlex_filepath, cols)
    word1_index = create_word_index(simlex_df, 'word1')  # create the index of the unique words in col 'word1'
    word2_index = create_word_index(simlex_df, 'word2')  # create the index of the unique words in col 'word2'
    simlex_df['topk'] = calculate_topk(simlex_df,10)  # list the topk words for each word in word1 col and create a col named 'topk'
    simlex_df = fill_topk(simlex_df, word2_index)  # fill the topk values for rows which have less than k values

    return simlex_df

if __name__ == '__main__':
    prepare_simlex()