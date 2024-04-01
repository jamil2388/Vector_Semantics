'''

Create vectors of words from given corpus
using methods like 'tfidf', 'word2vec' etc.

Initially we are using brown corpus (news and romance genre)

We will evaluate in the same file

'''



from nltk.corpus import brown
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from simlex import prepare_simlex
import pytrec_eval
import pandas as pd
import numpy as np
import re

# Preprocess the corpus
def preprocess_text(text):
    text = ' '.join(text) # Join the tokens into a single string
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove unnecessary characters
    return text.lower()

def extract_corpus(raw_corpus, category):
    c = []
    for i, sent in enumerate(brown.sents(categories=[category])):
        if i > 5 : break
        c.append((preprocess_text(sent)))
    return c


def tfidf(c):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(c) # Transform the corpus into TF-IDF vectors
    print("Shape of TF-IDF matrix:", tfidf_vectors.shape)
    return vectorizer, tfidf_vectors.transpose() # now each row would represent each word vector

# extract vocab, index and the vectors from the model and store it in a dataframe
def index_vectors(model, vectors, tfidf = 1):
    df = pd.DataFrame()
    vocab_dict = model.vocabulary_ if tfidf else None

    df['vocab'] = vocab_dict.keys()
    df['index'] = vocab_dict.values()

    # Create a new column 'vector' in df
    df['vector'] = None

    # # Iterate over each row in df
    # for index, row in df.iterrows():
    #     # Get the index from the row
    #     index_val = row['index']
    #     # Check if the index exists in the tfidf_vectors matrix
    #     if index_val in vocab_dict.values():
    #         vector = vectors[:, index_val]
    #         # Assign the vector to the 'vector' column in df
    #         df.at[index, 'vector'] = vector.todense()

    return df

def eval():
    pass

if __name__ == '__main__':

    brown_news = extract_corpus(brown, 'news')
    brown_romance = extract_corpus(brown, 'romance')
    print(brown_news)
    print(brown_romance)

    print("Total number of words in Brown corpus:", len(brown.words()))
    print("Total number of sentences in Brown corpus:", len(brown.sents()))

    print("Raw data from the news genre of Brown corpus (first 100 characters):", brown.raw(categories=['news'])[:100])
    print("Number of sentences in the news genre of Brown corpus:", len(brown.sents(categories=['news'])))
    print("Number of sentences in the romance genre of Brown corpus:", len(brown.sents(categories=['romance'])))

    simlex_vocab = prepare_simlex('../data/simlex_prep.pkl')

    # two different vectors for news and romance corpora
    vectorizer_news, tfidf_vectors_news = tfidf(brown_news)
    print(tfidf_vectors_news.todense())
    news_df = index_vectors(vectorizer_news, tfidf_vectors_news, 1)# create a dataframe containing the vocabulary, index and their tfidf, word2vec vectors
    filtered_words = [word for word in simlex_vocab['word'] if word in vectorizer_news.vocabulary_] # we only work on words in the simlex_vocab, which are present in our brown corpus genres
    f_simlex_vocab = simlex_vocab[simlex_vocab['word'].isin(filtered_words)] # Filter simlex_vocab based on the filtered words





    