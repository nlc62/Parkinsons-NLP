# LDA.py

# Author:   Alex Bock and Andy Valenti
# Source:   https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Date:     13 August 2018

# This script provides functions to preprocess Gensim document sets and train
# a topic model.

# Setup ========================================================================

# imports

from itertools import chain
import re
import copy
# import contractions

import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
from collections import OrderedDict

import spacy

# nlp = spacy.load("en_core_web_sm")

import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

# import nltk

# silence unnecessary warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Building LDA Mallet Model

mallet_path = '../tools/mallet-2.0.8/bin/mallet'  # when running locally
# mallet_path = '/cluster/home/avalen02/parkinsons/mallet-2.0.8/bin/mallet'   # use when running on the HPC

# convergence logging

import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


# Preprocessing ================================================================

stop_words_exp = re.compile(
        r'xxxx|xxx|Xxxx|youknow|Youknow|rrWell|RrWell|rrwell|rrlike|\bMy\b|\bmy\b|xxbarrexx|Imean|xxProvidencexx')
stops = ["know", "thing", "go", "get", "be", "do", "come", "think", "well", "lot", "take", "so"]
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]

stop_words_exp

# Input: List of sentence strings
# Output: List containing lists of words per sentence
def preprocess(data):
    processed_sentences = []
    for text in data:
        # Replace STOPWORDS with spaces
        text = re.sub(stop_words_exp, " ", text)

        # Make LOWERCASE and REMOVE ACCENTS from letters
        text_list = gensim.utils.simple_preprocess(text, deacc=True)

        # Create full sentence strings for lemmatization
        text_str = " ".join(text_list)
        doc = nlp(text_str)

        # LEMMATIZE (finds base word eg. studies -> study)
        lemmatized_words = [token.lemma_ for token in doc if token.pos_ in allowed_postags]

        # Remove STOPWORDS
        processed_sentences.append([word for word in lemmatized_words if word not in stops])

    # Bigram model
    bigram = gensim.models.Phrases(processed_sentences,
                                   min_count=5,
                                   threshold=10)

    # Cuts down bigram's unneeded states
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Apply bigram on all sentences
    processed_sentences = [bigram_mod[line] for line in processed_sentences]

    return processed_sentences


# input: vector of data; each entry includes a filename and a string con-
#        taining the contents of the file
# output: data with file contents separated into words and lemmatized
# def old_preprocess(data):
#     stop_words_exp = re.compile(
#         r'xxxx|xxx|Xxxx|youknow|Youknow|rrWell|rrwell|rrlike|\bMy\b|\bmy\b|xxbarrexx|Imean|xxProvidencexx')
#
#     # split file contents into separate words
#     data1 = []
#
# # we are adding another dimension to the data1 list to hold the length of the text which we'll use as another feature
#     for text in data:
#         # text = contractions.fix(text)
#         # In text, replace stopwords with spaces
#         text = re.sub(stop_words_exp, " ", text)
#         data1.append([gensim.utils.simple_preprocess(text, deacc=True)])
#         data1[-1].extend([[len(data1[-1][-1])]])
#     data = data1
#
#     # lemmatize words in files
#     data1 = []
#     allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]
#     nlp = spacy.load("en", disable=["parser", "ner"])
#
#     for text in data:    # note we have to index the text and not the len
#         doc = nlp((" ".join(text[0])).decode("utf8"))
#         data1.append([[token.lemma_ for token in doc if token.pos_ in allowed_postags]])
#         data1[-1].extend([text[1]])
#     data = data1
#
#     data1 = []
#     stops = ["know", "thing", "go", "get", "be", "do", "come", "think", "well", "lot", "take", "so"]
#
#     for text in data:
#         data1.append([word for word in text[0] if word not in stops])  # note to index only the text
#
#     # data = data1
#     bigram = gensim.models.Phrases(data1, min_count=5, threshold=10)
#     bigram_mod = gensim.models.phrases.Phraser(bigram)
#     data1 = [bigram_mod[line] for line in data1]
#
#     # easy way, in not most efficient, to restore the word counts before we return data1
#     post_processed = []
#     for idx, elem in enumerate(data):
#         post_processed.append([data1[idx], elem[1]])
#
#     return post_processed

# Input: List of lists of words per sentence
# Output: List of topic proportion lists per sentence
def text_to_topic_vectors(documents, lda_dictionary, lda_model, text_len=False):

    # Bag Of WordS
    bows = [lda_dictionary.doc2bow(document) for document in documents]  # NB: must index the doc text and not length

    if text_len:
        #   the following lines add length as a feature
        feature_vectors = []
        for idx, bow in enumerate(bows):
            feature_vector = [p for (_, p) in lda_model[bow]]
            text_len = float(documents[idx][1][0])
            feature_vector.extend([text_len])
            feature_vectors.append(feature_vector)

    else:  # we don't want to add length feature
        feature_vectors = [[p for (_, p) in lda_model[bow]] for bow in bows]

    return feature_vectors


def make_corpus(data):
    id2word = corpora.Dictionary(data)

    # removes the most common and rarest words
    # id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.filter_extremes(no_below=5)
    id2word.compactify()

    corpus = [id2word.doc2bow(text) for text in data]

    return corpus, id2word


# input: number of topics, preprocessed training set
# output: trained LDA model, dictionary
def build_lda_model(corpus, num_topics, id2word, mallet=False):
    lda_params = None
    lda_model = None

    if mallet:
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     id2word=id2word,
                                                     num_topics=num_topics)
    else:
        lda_params = OrderedDict({'corpus': corpus,  # default: None
                                  'num_topics': num_topics,  # default: 100
                                  'id2word': id2word,  # default: None
                                  # 'distributed': False,  # default: False
                                  # 'chunksize': 228,  # default: 2000
                                  'passes': 20,  # default: 1
                                  # 'update_every': 1,  # default: 1
                                  'alpha': 'auto',  # default: 'symmetry'
                                  'eta': 'auto',  # default: None
                                  # 'decay': .5,  # default: .5
                                  # 'offset': 1.0,  # default: 1.0
                                  # 'eval_every': 10,  # default :10
                                  'iterations': 10000,  # default: 50
                                  # 'gamma_threshold': .001,  # default: .001
                                  'minimum_probability': 0,  # default: .01
                                  'random_state': 100,  # default: None
                                  # 'ns_conf': None,  # default: None
                                  # 'minimum_phi_value': .01,  # default: .01
                                  # 'per_word_topics': False,  # default: False
                                  # 'callbacks': None,  # default: None
                                  # 'dtype':,  # default: <type 'numpy.float32'>
                                  })
        lda_model = gensim.models.ldamulticore.LdaModel(**lda_params)
        lda_params = copy.deepcopy(lda_params)
        lda_params.pop("corpus")
        lda_params.pop("id2word")

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    vis
    # pyLDAvis.save_html(vis, 'topics.html')

    return lda_model, lda_params


# save model to disk
def save_LDA_model(lda):
    temp_file = datapath("LDA_model")
    lda.save(temp_file)


# load a pretrained model from disk
def load_LDA_model(dict_filename):
    id2word = corpora.Dictionary.load_from_text(dict_filename)
    temp_file = datapath("LDA_model")
    return gensim.models.ldamodel.LdaModel.load(temp_file), id2word


#   The following computes coherence values only for LDA MALLET
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, if_mallet=False):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("** computing coherence for {} topics".format(num_topics))
        if if_mallet:
            model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     num_topics=num_topics,
                                                     id2word=dictionary)
        else:
            model = gensim.models.ldamulticore.LdaModel(corpus=corpus,
                                                        id2word=dictionary,
                                                        num_topics=num_topics,
                                                        iterations=10000,
                                                        random_state=100,
                                                        # update_every=1,
                                                        passes=20,
                                                        # chunksize=228,
                                                        minimum_probability=0,
                                                        alpha='auto', eta='auto')

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
