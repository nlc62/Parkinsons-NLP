# REVISION HISTORY
# 02JUL19   mgold Implemented clickdata class for storing all user data used for creating models
# 20AUG19   mgold 5-class -n=100 -k=5 mlp c hidden_layer_sizes=(50, 50), activation='tanh', solver='adam', alpha=.05, learning_rate='constant', max_iter=2500, random_state=1
# 26AUG19   mgold Added MovieData dataset, made ClickData and MovieData inherit from abstract class Dataset
# 01SEP19   mgold Added full ClickData dataset
# 05SEP19   mgold EVEN SPLIT (780) FOR TRAINING BELOW:
# 05SEP19   mgold 5-class -n=500 -k=5 mlp c hidden_layer_sizes=(50, 50), activation='tanh', solver='adam', alpha=.05, learning_rate='constant', max_iter=2500, random_state=1
# 12SEP19   mgold Re-completed saving prediction info to csv


# create_models.py
#
# Reads existing interview data and AMT evaluation data. Builds a latent Dirich-
# let allocation topic model from the collection of interviews. Builds and tests
# a machine learning classifier from associations between topic representations
# of interview sentences and the sentences' corresponding AMT ratings.
#
# Requires:
#
#   - [corpus_path]/dataset_master/: Directory containing all interview tran-
#     scripts obtained from the occupational therapy study (including those that
#     have been AMT-evaluated); these are used for building the topic model
#   - [data_path]/transcripts/: Directory containing interview transcripts that
#     have been AMT-evaluated; each file presents a list of the individual sen-
#     tences presented to evaluators so that each line of text has a correspon-
#     ding evaluation
#   - [data_path]/emitagreement.csv: Spreadsheet listing, for every evaluated
#     sentence, its sentence ID (source interview plus relative position in in-
#     terview), average AMT evaluation, and consensus statistics (>80% agreement
#     on valence, arousal, and valence/arousal combination)
#   - [data_path]/sentences.csv: Lists the same sentences as emitagreement.csv,
#     with sentence ID, raw sentence text, sentence word count, true emotional
#     label, difference in emotional label from neighboring sentences, and
#     (optionally) predicted emotional label and position in an predefined tes-
#     ting set
#
# Produces:
#
#   - LDA topic model written to LDA.sav, and ML model written to classifier.sav
#   - testing_data.csv: Spreadsheet listing testing sentence ID, raw text, con-
#     sensus statistic, true emotional label, and predicted emotional label
#
# Parameters:
#
#   - Required: Identifier for desired ML classification algorithm can be one of
#     {knn, lr, svm, mlp}
#   - -d: Use sentences.csv to build testing set from predefined positions in
#     sentences.csv (default: use same set of sentences every run)
#   - -a: Predict arousal label (default: predict valence label)
#   - -ml: Predict both valence and arousal
#   - -t x: Provide value to define middle prediction class; enables multiclass
#     classification
#   - -n: Number of LDA model topics (default: 4)
#   - -T: Do not test prediction model with testing set
#   - -v: Assign emotional label based on most common evaluation (default:
#     assign based on average evaluation)
#   - -c: Get LDA model coherence report
#   - -w: Write testing set data to testing_data.csv (only occurs if specified)
#   - -rr: Re-randomize testing set
#   - -wc x: Only use sentences with at least x words for training and testing
#     sets
#   - -r: Use radial calculation for value-to-class translation (default is
#     using bare Cartesian coordinates)

__author__ = "Andy Valenti, Alex Bock, & Michael Gold"
__copyright__ = "Copyright 2019. Tufts University"


# For updating all packages
from subprocess import call

# imports
import nltk
import sklearn
import spacy
import gensim
import numpy
import scipy

import pickle
import argparse
import sys
import copy

# Ignore all warnings
from utilities import print_classification_summary
from utilities import append_prediction_info_to_csv
from gensim.models import HdpModel

from LDA import preprocess
from LDA import make_corpus
from LDA import text_to_topic_vectors
from LDA import build_lda_model
from LDA import compute_coherence_values

from clickdata import ClickData
from moviedata import MovieData
from utilities import get_data
from feature_classifier import eval_model, test_model, sweep_model

import matplotlib.pyplot as plt
from pickle import dump



#####################################################################
#                                                                   #
#                         INITIAL ARGUMENTS                         #
#                                                                   #
#####################################################################
CORPUS_PATH = "../dataset_master/all_parkinsons_texts/"  # All parkinsons data files (containing sentences).
EE_PATH = "../dataset_rated/ee/"  # Parkinsons Psiturk ratings.  Contains "batch[NUM]", which contains "eval_click.csv"
MR_PATH = "../dataset_rated/mr/"  # Movie review ratings.
LDA_PATH = "LDA.sav"
MODEL_PATH = "classifier.sav"
CLASS_CENTER_PTS_PATH = "class_center_pts.txt"
PREDICTION_SAVE_PATH = "predictions.csv"

# Default settings
TEST_SPLIT_PERCENT = 0.5

# Column identifiers for standardized spreadsheet indexing
# a2z = 'abcdefghijklmnopqrstuvwxyz'
# spreadsheet_dict = {v: k for k, v in enumerate(a2z)}
import nltk

#####################################################################
#####################################################################
#####################################################################


parser = argparse.ArgumentParser(description='LDA model and prediction model creation')

# Training Data Filtering Commands
parser.add_argument('-q', '--req_agreement_percent', action='store', type=int, default='0',
                    help='Percent agreement of individual user-ratings that is required of each sentence.')
parser.add_argument('-s', '--training_sentence_max', action='store', type=int, default='100000',
                    help='Number of training sentences to use per class.')
parser.add_argument('-sd', '--stdev_max', action='store', type=int, default='10000',
                    help='Only accept sentences with standard deviation ratings lower than input.')
parser.add_argument('-wc', '--word_count_min', action='store', type=int, default='0',
                    help='Minimum number of words allowed for original sentence word counts.')

# Testing Data Filtering Commands
parser.add_argument('-T', '--test_split_percent', action='store', type=float, default=0,
                    help='Use percent in range [0, 1] testing split.')
parser.add_argument('-HA', '--test_high_agreement', action='store_true',
                    help='Use filtering params for testing set.')

# Class Mapping Commands
parser.add_argument('-cx', '--class_center_pts_x', action='store', nargs='+', type=int, default=[],
                    help='Manual input of center points for classes (along x axis).  Can pair with -cy.')
parser.add_argument('-cy', '--class_center_pts_y', action='store', nargs='+', type=int, default=[],
                    help='Manual input of center points for classes (along y axis).  Can pair with -cx.')
parser.add_argument('-k', '--kmeans', action='store', type=int, default=0,
                    help='Enter the number of kmeans classes to derive.')

# LDA Topic Model Setup Commands
parser.add_argument('-l', '--length', action='store_true',
                    help='Include text length as feature')
parser.add_argument('-n', '--numtopics', action='store', type=int, default='0',
                    help='Generate a new LDA topic model from the master dataset.  Input number of LDA topics.')

# ML Model Setup Commands
parser.add_argument('model_algorithm', action='store', choices=['dt', 'knn', 'lin', 'log', 'mlp', 'rf', 'sgd', 'svm'], nargs='?', type=str, default='mlp',
                    help='ML algorithm')
parser.add_argument('model_technique', action='store', choices=['c', 'r'], nargs='?', type=str, default='c',
                    help='Classification or regression.')

# General Commands
parser.add_argument('-c', '--coherence', action='store_true',
                    help='Predict highest-voted valence/arousal score')
parser.add_argument('-i', '--infer', action='store_true',
                    help='Infer number of topics')
parser.add_argument('-p', '--sweep', action='store_true',
                    help='Parameter sweep')
parser.add_argument('-rr', '--rerandomize', action='store_true',
                    help='Rerandomize testing set')
parser.add_argument('-u', '--update', action='store_true',
                    help='Update libraries and quit')
parser.add_argument('-w', '--write_test_data', action='store_true',
                    help='Write testing data (text and evaluations) to file.')


options = parser.parse_args()

# Training Data Filtering Commands
REQUIRED_AGREEMENT_PERCENT = options.req_agreement_percent
TRAINING_SENTENCE_MAX = options.training_sentence_max
STANDARD_DEVIATION_MAX = options.stdev_max
WORD_COUNT_MIN = options.word_count_min

# Testing Data Filtering Commands
TEST_SPLIT_PERCENT = options.test_split_percent
TEST_HIGH_AGREEMENT = options.test_high_agreement

# Mapping Mapping Commands
MANUAL_CLASS_CENTERS_X = options.class_center_pts_x
MANUAL_CLASS_CENTERS_Y = options.class_center_pts_y
NUM_KMEANS_CLASSES = options.kmeans

# LDA Topic Model Setup Commands
TEXT_LEN = options.length
LDA_CREATE_NUMTOPICS = options.numtopics

# ML Model Setup Commands
MODEL_ALGORITHM = options.model_algorithm
MODEL_TECHNIQUE = options.model_technique

# General Commands
COHERENCE = options.coherence
INFER_TOPICS = options.infer
SWEEP = options.sweep
RERANDOMIZE = options.rerandomize
UPDATE = options.update
WRITE_TEST_DATA = options.write_test_data  # Currently unused


###########################
# FURTHER PARAMETER SETUP #
###########################
# Setup manual class-centers
MANUAL_CLASS_CENTERS = [[0, 0] for _ in range(max(len(MANUAL_CLASS_CENTERS_X), len(MANUAL_CLASS_CENTERS_Y)))]
for i, x in enumerate(MANUAL_CLASS_CENTERS_X):
    MANUAL_CLASS_CENTERS[i][0] = x
for i, y in enumerate(MANUAL_CLASS_CENTERS_Y):
    MANUAL_CLASS_CENTERS[i][1] = y
# with open(CLASS_CENTER_PTS_PATH, "w") as outfile:
#     outfile.write(str(traindata.get_class_center_pts()))

# Conflicts
if len(MANUAL_CLASS_CENTERS) == 0 and NUM_KMEANS_CLASSES == 0:  # Default to kmeans=3
    NUM_KMEANS_CLASSES = 3
assert max(len(MANUAL_CLASS_CENTERS), NUM_KMEANS_CLASSES) >= 2  # Must have at least 2 classes


##########
# UPDATE #
##########
# Update all relevant packages for the model
packages = ["nltk", "scikit-learn", "spacy", "gensim", "scipy", "python"]


def get_versions_str():
    versions = [
        nltk.__version__,
        sklearn.__version__,
        spacy.__version__,
        gensim.__version__,
        scipy.__version__,
        sys.version
    ]
    versions_str = ", ".join([packages[i] + ": " + versions[i] for i in range(len(packages))])
    return versions_str


if UPDATE:
    print "---==| UPDATING |==---"
    prev_versions = get_versions_str()
    call("sudo pip install --upgrade " + ' '.join(packages), shell=True)
    curr_versions = get_versions_str()
    print ""
    print "PREV VERSIONS: {}".format(prev_versions)
    print "CURR VERSIONS: {}".format(curr_versions)
    sys.exit("\nPerfection is imminent... until then, we update.")

print "CURR VERSIONS: {}".format(get_versions_str())


print("")
print("================================= {} {} =================================".format(MODEL_ALGORITHM, MODEL_TECHNIQUE))
if SWEEP:
    print("*** parameter sweep mode ***\n")


# print "Number of sentences: \t\t{}".format(num_sentences)
print "number LDA topics:                {}".format(LDA_CREATE_NUMTOPICS)
print "ML algorithm:                     {}".format(MODEL_ALGORITHM)
print "ML technique:                     {}".format(MODEL_TECHNIQUE)

print "Filtering for rater consensus of: {}%".format(REQUIRED_AGREEMENT_PERCENT) if REQUIRED_AGREEMENT_PERCENT else "Using rater average"
print "Minimum word count threshold:     {}".format(WORD_COUNT_MIN)


##########################
##########################
#     SELECT DATASET     #
##########################
##########################
dataset_name = "clickdata"
# dataset_name = "moviedata"
traindata, testdata = get_data(dataset_name=dataset_name,
                               model_technique=MODEL_TECHNIQUE,
                               manual_class_centers=MANUAL_CLASS_CENTERS,
                               num_kmeans_classes=NUM_KMEANS_CLASSES,
                               test_split_percent=TEST_SPLIT_PERCENT,
                               rerandomize=RERANDOMIZE,
                               training_sentence_max=TRAINING_SENTENCE_MAX)


#############
# PLOT DATA #
#############
# traindata.plot_valences()
# traindata.plot_all_data("Traindata (Indiv. User Ratings) ({})".format(traindata.num_ratings()))
# traindata.plot_mean_data("Traindata (Mean Ratings) ({})".format(traindata.num_sentences()))
# if TEST_SPLIT_PERCENT > 0:
#     testdata.plot_all_data("Testdata (Indiv. User Ratings) ({})".format(testdata.num_ratings()))
#     testdata.plot_mean_data("Testdata (Mean Ratings) ({})".format(testdata.num_sentences()))

# raw_input("enter")

print("")
print("================================= DATASETS =================================")
# Training
print "Test datacounts:"
print "sentences:           {}".format(testdata.get_n_texts())
print "Test sentence share: {}%".format(TEST_SPLIT_PERCENT)

if MODEL_TECHNIQUE == "c":
    print_classification_summary(traindata.get_class_counts(), testdata.get_class_counts())


#####################
#####################
#     SETUP LDA     #
#####################
#####################
lda_params_dict = None
lda_model = None
lda_dictionary = None
lda_corpus = None

# If a number of topics is entered, train the LDA model!  Otherwise, use the saved LDA model.
if LDA_CREATE_NUMTOPICS > 0:
    # LDA sentences = ALL SENTENCES, RATED OR UNRATED, except testing sentences
    # sentences_by_file_to_add_to_corpus = []
    # filenames_already_rated = []
    # if TEST_SPLIT_PERCENT > 0:
    #     sentences_by_file_to_add_to_corpus = traindata.get_all_strings_by_file()
    #     filenames_already_rated = dataset.get_all_filenames()
    # lda_texts = compile_corpus_without_test_sentences(CORPUS_PATH, sentences_by_file_to_add_to_corpus, filenames_already_rated)

    print "Collecting LDA corpus files"
    lda_texts = traindata.get_all_strings_for_lda()

    print "Preprocessing LDA corpus"
    lda_processed = preprocess(lda_texts)

    print "Creating LDA model"
    (lda_corpus, lda_dictionary) = make_corpus(lda_processed)

    print "\n*** Building LDA model on {} topics for {} documents...".format(LDA_CREATE_NUMTOPICS, len(lda_texts))
    lda_model, lda_params_dict = build_lda_model(corpus=lda_corpus,
                                                 num_topics=LDA_CREATE_NUMTOPICS,
                                                 id2word=lda_dictionary,
                                                 mallet=False)

else:
    print "Loading LDA model"
    lda_model = pickle.load(open(LDA_PATH, "rb"))
    lda_params_dict = copy.deepcopy(vars(lda_model))
    lda_params_dict.pop("alpha")
    print(lda_params_dict)
    # del lda_params_dict["expElogbeta"]
    lda_dictionary = lda_model.id2word


lda_model.print_topics()


########################
#     INFER TOPICS     #
########################
if INFER_TOPICS:
    print("*** using Hierarchical Dirichlet Process to infer number of topics ***")
    hdp = HdpModel(lda_corpus, lda_dictionary)
    topic_info = hdp.print_topics(num_topics=-1, num_words=10)
    num_topics = len(topic_info)
    print ("num topics inferred by HDP: {}".format(num_topics))
    sys.exit("HDA")


#####################
#     COHERENCE     #
#####################
# Compute Coherence Score

# coherence_model_ldamallet = CoherenceModel(model=lda_model, texts=lemmatized_data, dictionary=id2word,
#                                            coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: {}'.format(round(coherence_ldamallet),2))

if COHERENCE:
    model_list, coherence_values = compute_coherence_values(dictionary=lda_dictionary,
                                                            corpus=lda_corpus,
                                                            texts=lda_processed,
                                                            start=5,
                                                            limit=100,
                                                            step=5,
                                                            if_mallet=False)
    start=5; stop=100; step=5
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()
    sys.exit("Coherence completed")


######################
######################
#     EVAL MODEL     #
######################
######################
# process documents into lists of words
print "Preprocessing training/testing sentences"
training_processed = preprocess(traindata.get_all_strings())
testing_processed = preprocess(testdata.get_all_strings())

# Create feature vectors
print "Creating feature vectors"
training_features = text_to_topic_vectors(documents=training_processed,
                                          lda_dictionary=lda_dictionary,
                                          lda_model=lda_model,
                                          text_len=TEXT_LEN)
testing_features = text_to_topic_vectors(documents=testing_processed,
                                         lda_dictionary=lda_dictionary,
                                         lda_model=lda_model,
                                         text_len=TEXT_LEN)

# Choose data for model training
print "Selecting training/testing data"
train_targets = []
test_targets = []
train_weights = []
test_weights = []

if MODEL_TECHNIQUE == "c":
    train_targets = traindata.get_all_labels()
    test_targets = testdata.get_all_labels()
    train_weights = None
elif MODEL_TECHNIQUE == "r":
    train_targets = traindata.get_all_valences()
    test_targets = testdata.get_all_valences()
    train_weights = None
    test_weights = None
    # train_weights = [wt * 10000 for wt in traindata.get_all_inverse_variances()]
    # test_weights = [wt * 10000 for wt in testdata.get_all_inverse_variances()]
    # train_weights = [1 if stdev > 50 else 100 for stdev in traindata.get_all_stdevs()]
    # train_weights = traindata.get_all_stdevs()

    ##################################################
    # training_targets = traindata.get_all_rating_coords()
    # testing_targets = testdata.get_all_rating_coords()
    # training_targets = [item for sublist in training_targets for item in sublist]
    # testing_targets = [item for sublist in testing_targets for item in sublist]

    print "training_features and training_targets: {}, {}".format(len(training_features), len(train_targets))
    print "testing_features and testing_targets: {}, {}".format(len(testing_features), len(test_targets))
    ##################################################


# Find the best "feature" parameters
if SWEEP:
    print("\n *** grid search beginning ***")
    sweep_params = sweep_model(train_features=training_features,
                               train_targets=train_targets,
                               algorithm=MODEL_ALGORITHM,
                               technique=MODEL_TECHNIQUE)
    print ("\n *** best algorithm parameters ***")
    print sweep_params
    sys.exit("\nSomeday, all the world's floors will need sweeping")

# evaluate and test prediction model
(ml_model, auc, matrix, report_dict, report_string) = eval_model(train_features=training_features,
                                                                 train_targets=train_targets,
                                                                 train_weights=train_weights,
                                                                 algorithm=MODEL_ALGORITHM,
                                                                 technique=MODEL_TECHNIQUE,
                                                                 text_len=TEXT_LEN)


if MODEL_TECHNIQUE == "c":
    print report_string
    print "Confusion Matrix:"
    print "(y-axis actual, x-axis predicted)"
    print matrix


# Save LDA model, ML model, and class center pts to disk for later use
with open(LDA_PATH, "wb") as outfile:
    dump(lda_model, outfile)
with open(MODEL_PATH, "wb") as outfile:
    dump(ml_model, outfile)
# with open(CLASS_CENTER_PTS_PATH, "w") as outfile:
#     outfile.write(str(traindata.get_class_center_pts()))

# Save prediction info to CSV
append_prediction_info_to_csv(csv_path=PREDICTION_SAVE_PATH,
                              algorithm=MODEL_ALGORITHM, technique=MODEL_TECHNIQUE,
                              report_dict=report_dict, conf_matrix=matrix,
                              traindata=traindata, lda_params=lda_params_dict, ml_model=ml_model)


# Test with predictions
if TEST_SPLIT_PERCENT > 0:
    test_predicted = test_model(test_sentences=testdata.get_all_strings())

    # Plot
    testdata.plot_actual_vs_predicted(predicted_list=test_predicted)

    # Calculate mean error
    test_mse = testdata.get_all_mean_squared_errors(test_predicted)
    test_accuracy = sum(test_mse) / len(test_mse)
    print "mse accuracy: {}".format(test_accuracy)

    error_list = [abs(actual - predicted) for actual, predicted in zip(test_targets, test_predicted)]
    mean_error = sum(error_list) / len(error_list)
    print "Mean error: {}".format(mean_error)


# raw_input("Press enter to exit")
sys.exit("\nHow do you eat an elephant? One bite at a time.")
