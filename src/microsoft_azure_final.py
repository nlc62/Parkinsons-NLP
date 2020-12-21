import requests
from pprint import pprint
import os
import csv 
import sys
from tabulate import tabulate

from moviesentencedata import MovieSentenceData
from moviedata import MovieData
from clickdata import ClickData
from mosidata import MosiData
from moseidata import MoseiData
from melddata import MELDdata

def sentiment_dataset(dataset_name, n_texts=None):    
    """
    Load the specified dataset and outputs a list of sentences
    and corresponding labels in a tuple
    """
    
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
        dataset = dataset.filter_sentence_length(0, 5120)
        
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

def preprocess_text(sentences):
    """
    Processes a list of sentences and outputs a dictionary with key "documents"
    and value that contains a list of dicts, which contains "ID" and "text" as keys
    Parameters: A list of strings
    Returns: A dict
    """
    sentences = sentences[290:300] #only choose 10 documents
    
    max_length = 0   #find character of longest sentence
    for sent in sentences:
        if max_length < len(sent):
            max_length = len(sent)
    
    print(max_length)
    json_sentences = {"documents": []}
    num_sentences = len(sentences)
    
    for index in range(num_sentences):
        temp_dict = {"language": "en", "id": str(index), "text": sentences[index]}
        json_sentences["documents"].append(temp_dict)
        
    return json_sentences
        
def run_azure(json_sentences):
    """
    Runs Microsoft Azure Sentiment Analysis on preprocessed text
    Parameters: A dict that contains the sentences in the required format
    Returns: A dict of azure results
    """
    print(len(json_sentences["documents"]))
    # Microsoft Azure API key
    sub_key = "fec683d1ad1d4f579e4e522c31e18151"
    endpoint = "https://azure-sentiment-analysis.cognitiveservices.azure.com/"

    sentiment_url = endpoint + "/text/analytics/v3.0/sentiment"
    
    headers = {"Ocp-Apim-Subscription-Key": sub_key}
    response = requests.post(sentiment_url, headers=headers, json=json_sentences)
    sentiments = response.json()
    #pprint(sentiments)
    #print(len(sentiments["documents"]))
    return sentiments

def analyze_azure_results(sentiment_dict):
    """
    Analyzes Microsoft Azure results
    Parameters: A dict that contains the output from the sentiment analysis tool
    Returns: A list of sentiment labels
    """
    with open('azure_pd_' + str(29) + '.csv', mode='w') as result_file:
        result_writer = csv.writer(result_file, delimiter='\n')
        
        #Pos: 2, Neutral: 1, Neg: 0
        sentiment_labels = []
    
        for result in sentiment_dict["documents"]:
            sentiment_label = result["sentiment"]
            if sentiment_label == "positive":
                sentiment_labels.append(2)
                result_writer.writerow('2')
            elif sentiment_label == "neutral":
                sentiment_labels.append(1)
                result_writer.writerow('1')
            elif sentiment_label == "negative":
                sentiment_labels.append(0)
                result_writer.writerow('0')
    
            #mixed result (see if match)
            else:
                pos_score = result["confidenceScores"]["positive"]
                neutral_score = result["confidenceScores"]["neutral"]
                neg_score = result["confidenceScores"]["negative"]
                score = max(pos_score, neutral_score, neg_score)
                if pos_score == score:
                    sentiment_labels.append(2)
                    result_writer.writerow('2')
                elif neutral_score == score:
                    sentiment_labels.append(1)
                    result_writer.writerow('1')
                else:
                    sentiment_labels.append(0)
                    result_writer.writerow('0')
            #print(len(sentiment_labels))
    return sentiment_labels


def preprocess_results(sentiment_labels, ground_truth_list):
    """
    Preprocesses results through comparing sentiment output and the ground truth labels for
    calculating confusion metrics
    Parameters: A list of sentiment labels and a list of ground truth labels
    Returns: A dict of a confusion matrix
    """

    ground_truth_list = ground_truth_list[180:190] #Choose 10 documents
    
    if len(sentiment_labels) != len(ground_truth_list):
        print("Warning: sentiment labels do not match ground truth labels")

    #[truth_pred]
    review_dict = {'pos_pos': 0, 'pos_neutral': 0, 'pos_neg': 0,
                        'neutral_pos': 0, 'neutral_neutral': 0, 'neutral_neg': 0,
                        'neg_pos': 0, 'neg_neutral': 0, 'neg_neg': 0}
    
    label_length = len(sentiment_labels)

    for index in range(label_length):
        if ground_truth_list[index] == 2:
            if sentiment_labels[index] == 2:
                review_dict['pos_pos'] += 1
                
            elif sentiment_labels[index] == 1:
                review_dict['pos_neutral'] += 1
                
            elif sentiment_labels[index] == 0:
                review_dict['pos_neg'] += 1

        elif ground_truth_list[index] == 1:
            if sentiment_labels[index] == 2:
                review_dict['neutral_pos'] += 1

            elif sentiment_labels[index] == 1:
                review_dict['neutral_neutral'] += 1

            elif sentiment_labels[index] == 0:
                review_dict['neutral_neg'] += 1

        elif ground_truth_list[index] == 0:
            if sentiment_labels[index] == 2:
                review_dict['neg_pos'] += 1

            elif sentiment_labels[index] == 1:
                review_dict['neg_neutral'] += 1

            elif sentiment_labels[index] == 0:
                review_dict['neg_neg'] += 1

    return review_dict

def preprocess_results_from_files(datapath, num_of_files, dataset, ground_truth_list):
    """
    Preprocesses results through comparing sentiment output from csv files and the ground truth labels for
    calculating confusion metrics
    Parameters: [datapath] is the path to the files, [num_of_files] is the number of csv files,
    and [ground_truth_list] is a list of corresponding ground truth labels
    Returns: A dict of a confusion matrix
    """

    sentiment_labels = []
    for num in range(num_of_files):
        with open(str(datapath) + 'azure_' + str(dataset) + '_' + str(num) + '.csv') as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                for label in row:
                    sentiment_labels.append(int(label))
        
    if len(sentiment_labels) != len(ground_truth_list):
        print("Warning: sentiment labels do not match ground truth labels")

    #[truth_pred]
    review_dict = {'pos_pos': 0, 'pos_neutral': 0, 'pos_neg': 0,
                        'neutral_pos': 0, 'neutral_neutral': 0, 'neutral_neg': 0,
                        'neg_pos': 0, 'neg_neutral': 0, 'neg_neg': 0}
    
    label_length = len(sentiment_labels)

    for index in range(label_length):
        if ground_truth_list[index] == 2:
            if sentiment_labels[index] == 2:
                review_dict['pos_pos'] += 1
                
            elif sentiment_labels[index] == 1:
                review_dict['pos_neutral'] += 1
                
            elif sentiment_labels[index] == 0:
                review_dict['pos_neg'] += 1

        elif ground_truth_list[index] == 1:
            if sentiment_labels[index] == 2:
                review_dict['neutral_pos'] += 1

            elif sentiment_labels[index] == 1:
                review_dict['neutral_neutral'] += 1

            elif sentiment_labels[index] == 0:
                review_dict['neutral_neg'] += 1

        elif ground_truth_list[index] == 0:
            if sentiment_labels[index] == 2:
                review_dict['neg_pos'] += 1

            elif sentiment_labels[index] == 1:
                review_dict['neg_neutral'] += 1

            elif sentiment_labels[index] == 0:
                review_dict['neg_neg'] += 1

    return review_dict

def confusion_metrics(review_dict):
    """
    Calculates precision, recall, f1, and accuracy metrics for
    each individual sentiment label and as a weighted average
    Parameters: A dict of a 3x3 confusion matrix
    Returns: A dict of calculated metrics
    """
    
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




