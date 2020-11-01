import sys
import numpy as np
import nltk
from tabulate import tabulate
from dataset_rated import mr
from dataset import TextData
import re
from moviesentencedata import MovieSentenceData
from moviedata import MovieData
from clickdata import ClickData
from mosidata import MosiData
from moseidata import MoseiData
from melddata import MELDdata
SOME_FIXED_SEED = 42

# before training/inference:
np.random.seed(SOME_FIXED_SEED)

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    

def vader_dataset(dataset_name, n_texts=None):
    
    #load dataset and get ground truth labels
    
    if dataset_name == 'ms':
        dataset = MovieSentenceData()
        dataset_path = "vader_Sentiment/"
        dataset.add_ratings_from_files(dataset_path=dataset_path)
        dataset.sanitize_all_strings()
        if n_texts != None:
            dataset.classify_by_class3()
            dataset, _ = dataset.get_only_n_texts_per_class(n_texts)
            
        #Extract ground truth labels
        ground_truth_list = dataset.get_all_labels()
        #point_list = dataset.get_all_coords()
        #point_list = dataset.get_all_ratings()
        #coord_list = []
    
        #for number in point_list:
            #for coord in number:
                #coord_list.append(float(coord))

        #ground_truth_list = []
        #for integer in coord_list:
            #if integer < (-0.05):
                #ground_truth_list.append(0)

            #elif integer < (0.05):
                #ground_truth_list.append(1)

            #elif integer < (4):
                #ground_truth_list.append(2)

    elif dataset_name == 'm':
        dataset = MovieData()
        dataset_path = "dataset_rated/mr/"
        dataset.add_ratings_from_files(dataset_path=dataset_path)
        
        print("Flattening data by valence")
        dataset.flatten_by_valence()
        
        dataset.sanitize_all_strings()
        
        if n_texts != None:
            dataset.classify_by_class3()
            dataset,_ = dataset.get_only_n_texts_per_class(n_texts)
        
        ground_truth_list = dataset.get_all_labels()
            
    elif dataset_name == 'pd':
        dataset = ClickData()
        dataset_path = "dataset_rated/ee/"
        dataset.add_ratings_from_files(dataset_path=dataset_path)
        
        if n_texts != None:
            dataset.classify_by_kmeans(3)
            dataset, _ = dataset.get_only_n_texts_per_class(n_texts)
            
        dataset.sanitize_all_strings()
    
        ground_truth_list = dataset.get_all_labels()

    elif dataset_name == 'meld':
        dataset = MELDdata()
        dataset_path = ""
        dataset.add_ratings_from_files(dataset_path=dataset_path)
        
        dataset.classify_by_class3()
        
        if n_texts != None:
            dataset, _ = dataset.get_only_n_texts_per_class(n_texts)
            
        dataset.sanitize_all_strings()

        ground_truth_list = dataset.get_all_labels()
        
    elif dataset_name == 'mosi':
        dataset = MosiData()
        dataset_path = ""
        dataset.add_ratings_from_files(dataset_path=dataset_path)
        
        dataset.classify_by_class3()
        
        if n_texts != None:
            dataset, _ = dataset.get_only_n_texts_per_class(n_texts)
            
        dataset.sanitize_all_strings()

        ground_truth_list = dataset.get_all_labels()

    elif dataset_name == 'mosei':

        dataset = MoseiData()
        #dataset_path = "/users/nathanielchin/Desktop/parkinsons/pipeline/"
        dataset_path = ""
        dataset.add_ratings_from_files(dataset_path=dataset_path)
    
        dataset.classify_by_class3()

        if n_texts != None:
            dataset, _ = dataset.get_only_n_texts_per_class(n_texts)

        ground_truth_list = dataset.get_all_labels()
        
    sentences = dataset.get_all_strings()

    return sentences, ground_truth_list
        
def run_vader(sentences, ground_truth_list):
    
    # Create a SentimentIntensityAnalyzer object.d
    vader_model = SentimentIntensityAnalyzer()
    
    label_list = []

    #truth_predicted
    review_dict = {'pos_pos': 0, 'pos_neutral': 0, 'pos_neg': 0, 'neutral_pos': 0, 'neutral_neutral': 0, 'neutral_neg': 0, 'neg_pos': 0, 'neg_neutral': 0, 'neg_neg': 0}
    
    for s in sentences:
        print("Sentence: {}".format(s))

        # polarity_scores method of SentimentIntensityAnalyzer
        # oject gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = vader_model.polarity_scores(s)

        print("Overall sentiment dictionary is: {}".format(sentiment_dict))
        print(" {}% Negative".format(sentiment_dict['neg'] * 100))
        print(" {}% Neutral".format(sentiment_dict['neu'] * 100))
        print(" {}% Positive".format(sentiment_dict['pos'] * 100))

        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            label_list.append(2)
            leng = len(label_list)
            print(" Overall: Positive")
            print(' Ground Truth: ' + str(ground_truth_list[leng-1]))

        elif sentiment_dict['compound'] <= -0.05:
            label_list.append(0)
            leng = len(label_list)
            print(" Overall: Negative")
            print(' Ground Truth: ' + str(ground_truth_list[leng-1]))

        else:
            label_list.append(1)
            leng = len(label_list)
            print(" Overall: Neutral")
            print(' Ground Truth: ' + str(ground_truth_list[leng-1]))

        print("")

    label_length = len(label_list)

    #Ground truth to vader_label comparison dictionary
    for index in range(0, label_length):
        if ground_truth_list[index] == 2:
            if label_list[index] == 2:
                review_dict['pos_pos'] += 1
                
            elif label_list[index] == 1:
                review_dict['pos_neutral'] += 1
                
            elif label_list[index] == 0:
                review_dict['pos_neg'] += 1

        elif ground_truth_list[index] == 1:
            if label_list[index] == 2:
                review_dict['neutral_pos'] += 1

            elif label_list[index] == 1:
                review_dict['neutral_neutral'] += 1

            elif label_list[index] == 0:
                review_dict['neutral_neg'] += 1

        elif ground_truth_list[index] == 0:
            if label_list[index] == 2:
                review_dict['neg_pos'] += 1

            elif label_list[index] == 1:
                review_dict['neg_neutral'] += 1

            elif label_list[index] == 0:
                review_dict['neg_neg'] += 1

    return review_dict

### Confusion Matrix Statistics

def confusion_metrics(review_dict):
                
    pos_count = review_dict['pos_pos'] + review_dict['pos_neutral'] + review_dict['pos_neg']
    neutral_count = review_dict['neutral_pos'] + review_dict['neutral_neutral'] + review_dict['neutral_neg']
    neg_count = review_dict['neg_pos'] + review_dict['neg_neutral'] + review_dict['neg_neg']
    total_count = pos_count + neutral_count + neg_count
    
    #Recall and Precision values
    
    pos_recall = round((review_dict['pos_pos'])/(review_dict['pos_pos']+review_dict['pos_neutral']+review_dict['pos_neg']),2)
    pos_precision = round((review_dict['pos_pos'])/(review_dict['pos_pos']+review_dict['neutral_pos']+review_dict['neg_pos']),2)

    neutral_recall = round((review_dict['neutral_neutral'])/(review_dict['neutral_neutral']+review_dict['neutral_pos']+review_dict['neutral_neg']),2)
    neutral_precision = round((review_dict['neutral_neutral'])/(review_dict['neutral_neutral']+review_dict['pos_neutral']+review_dict['neg_neutral']),2)

    neg_recall = round((review_dict['neg_neg'])/(review_dict['neg_neg']+review_dict['neg_pos']+review_dict['neg_neutral']),2)
    neg_precision = round((review_dict['neg_neg'])/(review_dict['neg_neg']+review_dict['pos_neg']+review_dict['neutral_neg']),2)

    #Accuracy
    pos_accuracy = round(review_dict['pos_pos']/total_count, 2)
    neutral_accuracy = round(review_dict['neutral_neutral']/total_count, 2)
    neg_accuracy = round(review_dict['neg_neg']/total_count, 2)
    total_diagonal = review_dict['pos_pos'] + review_dict['neutral_neutral'] + review_dict['neg_neg']
    total_accuracy = round(total_diagonal/total_count, 2)
    
    #F1 value
    pos_f1 = round(2 * ((pos_precision * pos_recall) / (pos_precision + pos_recall)), 2)
    neutral_f1 = round(2 * ((neutral_precision * neutral_recall) / (neutral_precision + neutral_recall)), 2)
    neg_f1 = round(2 * ((neg_precision * neg_recall) / (neg_precision + neg_recall)), 2)

    #weight avg
    weighted_average_precision = round((((pos_precision * pos_count) +
                                         (neutral_precision * neutral_count) +
                                         (neg_precision * neg_count))/total_count), 2)
    weighted_average_recall = round((((pos_recall * pos_count) +
                                      (neutral_recall * neutral_count) +
                                      (neg_recall * neg_count))/total_count), 2)
    weighted_average_f1 = round((((pos_f1 * pos_count) +
                                  (neutral_f1 * neutral_count) +
                                  (neg_f1 * neg_count))/total_count), 2)

    confusion_matrix = [[pos_precision, pos_recall, pos_f1, pos_accuracy, pos_count],
                        [neutral_precision, neutral_recall, neutral_f1, neutral_accuracy, neutral_count],
                        [neg_precision, neg_recall, neg_f1, neg_accuracy, neg_count],
                        [weighted_average_precision, weighted_average_recall, weighted_average_f1, total_accuracy, total_count]
                        ]
    
    print(tabulate([['pos', pos_precision, pos_recall, pos_f1, pos_accuracy, pos_count],
                    ['neutral', neutral_precision, neutral_recall, neutral_f1, neutral_accuracy, neutral_count],
                    ['neg', neg_precision, neg_recall, neg_f1, neg_accuracy, neg_count],
                    ['w_avg', weighted_average_precision, weighted_average_recall, weighted_average_f1, total_accuracy, total_count]
                    ], headers=['Class', 'Precision', 'Recall', 'F1', 'Accuracy', 'Count']))
                                   
    return confusion_matrix


