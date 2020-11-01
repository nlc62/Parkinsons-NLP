
# utilities.py
#
# Contains definitions of functions used by create_models.py in order to remove
# large/modular blocks of code and reduce clutter

import os
import csv
from datetime import datetime
from math import sqrt


import nltk
import sklearn
import spacy
import gensim
import numpy
import scipy
import sys
import pkg_resources
from subprocess import call
from clickdata import ClickData
from moviedata import MovieData




# Compile a dataset
def get_data(dataset_name="clickdata",
             model_technique="c",
             manual_class_centers=[],
             num_kmeans_classes=3,
             test_split_percent=.5,
             rerandomize=False,
             training_sentence_max=100000):
    dataset = None

    # Choose dataset
    dataset_path = ""
    if dataset_name == "clickdata":
        dataset = ClickData()
        dataset_path = "dataset_rated/ee/"
    elif dataset_name == "moviedata":
        dataset = MovieData()
        dataset_path = "dataset_rated/mr/"

    # Init data
    print "Initializing Raw Data"
    dataset.add_ratings_from_files(dataset_path=dataset_path)
    print "Num sentences: {}".format(dataset.get_n_texts())

    # Remove empty sentences
    print "Removing Empty Sentences"
    dataset = dataset.get_only_nonempty_preprocessed_texts()
    print "Num sentences: {}".format(dataset.get_n_texts())

    # Remove arousal
    print "Flattening data by valence"
    dataset.flatten_by_valence()

    # Classify
    if model_technique == "c":
        if len(manual_class_centers) > 0:
            print "Classifying MANUALLY"
            dataset.classify_by_input(manual_class_centers)

        else:
            print "Classifying KMEANS"
            dataset.classify_by_kmeans(num_kmeans_classes)

    # Training/testing split
    train = type(dataset)()
    test = type(dataset)()
    if test_split_percent > 0:
        train, test = dataset.training_testing_split(test_split_percent, rerandomize)
    else:
        train = dataset

    # Limit training set
    if model_technique == "c":
        train = train.get_only_n_texts_per_class(training_sentence_max)
        train = train.get_only_equal_texts_per_class()

    return train, test









# translate
#
# inputs:   raw x and y coordinate values, middle class threshold, radial?
# outputs:  discrete classifications for classifier training and emotional
#           prediction
#
# value-to-class translation

def translate(x, y, threshold, radial):

    if threshold == 0:
        x_val = 2 if x > 0 else 1
        y_val = 2 if y > 0 else 1
    elif radial:
        distance = sqrt((x**2) + (y**2))
        if abs(distance) <= abs(threshold):
            x_val = y_val = 0
        else:
            x_val = 2 if x > 0 else 1
            y_val = 2 if y > 0 else 1
    else:
        x_div = int(x / threshold)
        y_div = int(y / threshold)
        x_val = 2 if x_div >= 1 else (1 if x_div <= -1 else 0)
        y_val = 2 if y_div >= 1 else (1 if y_div <= -1 else 0)

    return x_val, y_val


# print_classification_summary
#
# inputs:   middle class threshold, classification distribution data structure
# outputs:  none
#
# prints distribution of assigned classes for data points in training and
# testing sets
def get_class_names(num_classes):
    target_names = []
    target_names.append("Most Negative (0): ")
    target_names.extend(["              (" + str(i) + "): " for i in range(1, num_classes - 1)])
    target_names.append("Most Positive (" + str(num_classes - 1) + "): ")
    return target_names


def print_classification_summary(training_counts, testing_counts):
    num_classes = len(training_counts)
    target_names = get_class_names(num_classes=num_classes)

    print ""
    print "------ Training counts ------"
    for i, train_count in enumerate(training_counts):
        line = target_names[i] + str(train_count)
        print line

    print ""
    print "------ Testing counts -------"
    for i, test_count in enumerate(testing_counts):
        line = target_names[i] + str(test_count)
        print line


# Purpose: Compiles all sentences, except "testing sentences" left out of sentences_by_file_to_append
# Inputs: path to all transcripts (to append those transcripts),
#         list of file sentences to append (presumably leaving out testing sentences),
#         list of filenames (to omit from appending with all transcripts)
# Output: List by file, each spot containing a single string of all sentences in that file
#         eg. ["file sentences", "file sentences", ...]
def compile_corpus_without_test_sentences(corpus_path, sentences_by_file_to_append, filename_list_already_added):
    corpus_filenames = []
    corpus_filepaths = []
    for (dirpath, dirnames, filenames) in os.walk(corpus_path):
        corpus_filepaths += [os.path.join(dirpath, filename) for filename in filenames]
        corpus_filenames += [filename for filename in filenames]

    corpus_list = []

    # Add training sentences
    for file_sentences in sentences_by_file_to_append:
        file_sentences = "\n".join(file_sentences)
        corpus_list.append(file_sentences)

    # Add rest of files
    for corpus_filename, corpus_filepath in zip(corpus_filenames, corpus_filepaths):
        # Add files from dataset_master if not already added as sentences
        if corpus_filename not in filename_list_already_added:
            with open(corpus_filepath, "r") as infile:
                raw_text = infile.read()
                corpus_list.append(raw_text)
    return corpus_list


# Update all relevant packages for the model
def update_packages():
    # packages = [dist.project_name for dist in pkg_resources.working_set]  # Get all... Some don't update correctly :(

    packages = ["nltk", "scikit-learn", "spacy", "gensim", "scipy", "python"]
    print ""
    print "COMMAND: sudo pip install --upgrade " + ' '.join(packages)

    # Collect previous versions
    prev_versions = [
        nltk.__version__,
        sklearn.__version__,
        spacy.__version__,
        gensim.__version__,
        scipy.__version__,
        sys.version
    ]

    # Collect updated versions
    call("sudo pip install --upgrade " + ' '.join(packages), shell=True)
    curr_versions = [
        nltk.__version__,
        sklearn.__version__,
        spacy.__version__,
        gensim.__version__,
        scipy.__version__,
        sys.version
    ]

    # Format and print info
    prev_string = ", ".join([packages[i] + ": " + prev_versions[i] for i in range(len(packages))])
    curr_string = ", ".join([packages[i] + ": " + curr_versions[i] for i in range(len(packages))])
    print "PREVIOUS VERSIONS: {}".format(prev_string)
    print "CURRENT VERSIONS:  {}".format(curr_string)


# Save prediction info to CSV
def append_prediction_info_to_csv(csv_path,
                                  algorithm, technique,
                                  report_dict, conf_matrix,
                                  traindata, lda_params, ml_model):
    header = [
        "Timestamp",
        "Algorithm",
        "Technique",

        "Accuracy",
        "Conf Matrix",

        "Dataset Info"
        "LDA Params",
        "Model Params",
    ]

    # Collect previous data (minus header)
    old_data = []
    if os.path.isfile(csv_path):
        with open(csv_path, "rU") as infile:
            old_data = list(csv.reader(infile))[1:]

    # Collect new data to append
    new_data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                algorithm,
                technique,
                report_dict,
                conf_matrix,
                traindata.get_info(),
                lda_params,
                ml_model.get_params()]

    # Write
    with open(csv_path, "w+") as outfile:
        output_writer = csv.writer(outfile)
        output_writer.writerow(header)
        for row in old_data:
            output_writer.writerow(row)
        output_writer.writerow(str(new_data))


# strip_corpus
#
# inputs:   path to corpus, path to line-separated documents, lines to remove
#           from corpus
# outputs:  full-document corpus with lines removed
# def strip_corpus(corpus_path, data_path, removals):
#
#     corpus = [os.path.basename(filename) for filename in os.listdir(corpus_path)]
#     result = []
#
#     ######
#     counter = 0
#
#     # List of filenames in dataset_master dir
#     for filename in corpus:
#         # Added from dataset_master
#         if filename not in removals:
#             with open(corpus_path + filename, "r") as infile:
#                 raw_text = infile.read()
#             result.append(raw_text)
#
#         # Not added from dataset_master
#         else:
#             include = []
#             exclude = removals[filename]
#             with open(data_path + "transcripts/" + filename, "r") as infile:
#                 lines = infile.readlines()
#
#             # Foreach sentence
#             for i in range(0, len(lines)):
#                 if i not in exclude:
#                     include.append(lines[i])
#                     counter += 1
#             raw_text = "\n".join(include)
#             result.append(raw_text)
#     print "*****COUNTER ONE"
#     print counter
#
#     return result


# trim_empty_sentences
#
# inputs:   lists of preprocessed word lists and corresponding emotional evalu-
#           ations; additional data structure to be trimmed (optional)
# outputs:  word list and evaluation lists with empty word lists and correspon-
#           ding evaluations removed

# def trim_empty_sentences(processed, evals, coords, a=None):
#
#     # ensure data structures are equivalent in size (assumed to be parallel)
#
#     num_sentences = len(processed)
#     assert(len(evals) == num_sentences)
#     assert(len(coords) == num_sentences)
#     if a:
#         assert(len(a) == num_sentences)
#
#     # look for empty word lists and remove corresponding data from evals and optional additional data structure
#     i = 0
#
#     while i < num_sentences:
#         if len(processed[i][0]) == 0:          # if word list is empty  NB: index the sentence and not the length
#             processed.pop(i)
#             evals.pop(i)
#             coords.pop(i)
#             if a:
#                 a.pop(i)
#             num_sentences -= 1
#         else:
#             i += 1
#
#     # ensure resulting data structures are equivalent size
#
#     assert(len(processed) == num_sentences)
#     assert(len(evals) == num_sentences)
#     assert(len(coords) == num_sentences)
#     if a:
#         assert(len(a) == num_sentences)
#
#     return processed, evals, coords, a


# def append_prediction_info_to_csv(csv_output_filename, num_train, num_test, num_topics,
#                                   min_agreement, neutral_threshold, neutral_format, classifier,
#                                   model, lda_iters, report_dict, confusion_matrix):
#     header = [
#         "Timestamp",
#         "Num Train",
#         "Num Test",
#         "Num Topics",
#         "Agree%",
#         "Neu Range",
#         "Neu Format",
#         "Classif",
#
#         "Activ",
#         "Solver",
#         "Hidden",
#         "Model Iter",
#
#         "LDA Iter",
#
#         "0 Prec",
#         "0 Recall",
#         "0 f1",
#         "0 Sup",
#         "1 Prec",
#         "1 Recall",
#         "1 f1",
#         "1 Sup",
#         "2 Prec",
#         "2 Recall",
#         "2 f1",
#         "2 Sup",
#
#         "u Prec",
#         "u Recall",
#         "u f1",
#         "u Sup",
#         "Mac Prec",
#         "Mac Recall",
#         "Mac f1",
#         "Mac Sup",
#         "WA Prec",
#         "WA Recall",
#         "WA f1",
#         "WA Sup",
#
#         "Conf Matrix",
#     ]
#
#     # Collect previous data (minus header)
#     previous_data = []
#     if os.path.isfile(csv_output_filename):
#         with open(csv_output_filename, "rU") as infile:
#             previous_data = list(csv.reader(infile))[1:]
#
#     # Collect new data to append
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     new_data = [
#         timestamp,
#         num_train,
#         num_test,
#         num_topics,
#         min_agreement,
#         neutral_threshold,
#         neutral_format,
#         classifier,
#     ]
#
#     model_info = ["", "", "", "", ""]
#     if classifier is "MLP":
#         model_info = [
#             model.activation,
#             model.solver,
#             model.hidden_layer_sizes,
#             model.max_iter,
#             lda_iters,
#         ]
#     new_data.extend(model_info)
#
#     # Classification report
#     label_order = ["0", "1", "2", "micro avg", "macro avg", "weighted avg"]
#     element_order = ["recall", "precision", "f1-score", "support"]
#     for label in label_order:
#         for element in element_order:
#             datum = report_dict[label][element]
#             datum = round(datum, 3)
#             new_data.append(datum)
#
#     # Confusion matrix
#     new_data.append(confusion_matrix)
#
#     # Write
#     with open(csv_output_filename, "w+") as outfile:
#         output_writer = csv.writer(outfile)
#         output_writer.writerow(header)
#         for previous_datum in previous_data:
#             output_writer.writerow(previous_datum)
#         output_writer.writerow(new_data)

    # previous_filename = ""
    # sentence_in_file_counter = 0
    # for sentence_data in self.rated_sentences:
    #     if sentence_data.get_filename() != previous_filename:
    #         sentence_in_file_counter = 0
    #     row = [
    #         sentence_data.get_filename(),
    #         sentence_in_file_counter,
    #         sentence_data.get_sentence_string(),
    #         sentence_data.get_mean_class(neutral_threshold, radial),
    #         sentence_data.get_agreement_percent(),
    #     ]
    #     previous_filename = sentence_data.get_filename()
    #     output_writer.writerow(row)


# write testing sentences and corresponding evaluations to file
# if WRITE_TEST_DATA:
#
#     with open("testing_data.csv", "w") as outfile:
#         test_data_writer = csv.writer(outfile)
#
#     if PRED_QUADRANT:
#         header = ["Document", "Sentence", "Participants", "Actual", "Predicted"]
#     elif PRED_AROUSAL:
#         header = ["Document", "Sentence", "Participants", "Actual arousal", "Predicted arousal"]
#     else:
#         header = ["Document", "Sentence", "Participants", "Actual valence", "Predicted valence"]
#
#     test_data_writer.writerow(header)
#
#     five_participants_counter = 0
#     ten_participants_counter = 0
#
#     five_participants_correct = 0
#     ten_participants_correct = 0
#
#
#     for i in range(0, num_test_sentences):
#         document = testing_dataset[i][0]
#         sentence = testing_dataset[i][1]
#         num_participants = testing_dataset[i][2]
#         actual_class = testing_dataset[i][3]
#         predicted_class = prediction[i]
#
#         # Accumulate participant accuracy data
#         if num_participants == 5:
#             five_participants_counter += 1
#             if actual_class == predicted_class:
#                 five_participants_correct += 1
#         elif num_participants == 10:
#             ten_participants_counter += 1
#             if actual_class == predicted_class:
#                 ten_participants_correct += 1
#
#         # Write data
#         row = [
#             document,
#             sentence,
#             num_participants,
#             actual_class,
#             predicted_class,
#         ]
#         test_data_writer.writerow(row)
#
#     # Write number-of-participants accuracy data
#     five_participants_accuracy = float(five_participants_correct) / five_participants_counter
#     ten_participants_accuracy = float(ten_participants_correct) / ten_participants_counter
#
#     test_data_writer.writerow("")
#     test_data_writer.writerow("")
#
#     header = [
#         "Five Participants Amount",
#         "Five Participants Correct",
#         "Five Participants Accuracy",
#     ]
#     test_data_writer.writerow(header)
#
#     row = [
#         five_participants_counter,
#         five_participants_correct,
#         five_participants_accuracy,
#     ]
#     test_data_writer.writerow(row)
#
#     header = [
#         "Ten Participants Amount",
#         "Ten Participants Correct",
#         "Ten Participants Accuracy",
#     ]
#     test_data_writer.writerow(header)
#
#     row = [
#         ten_participants_counter,
#         ten_participants_correct,
#         ten_participants_accuracy,
#     ]
#     test_data_writer.writerow(row)
