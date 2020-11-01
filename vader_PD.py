import sys
import numpy as np
import nltk
from tabulate import tabulate
#from nltk.corpus import wordnet
from clickdata import ClickData
from dataset_rated import ee
from dataset import TextData

SOME_FIXED_SEED = 42

# before training/inference:
np.random.seed(SOME_FIXED_SEED)
#nltk.download('wordnet')
#nltk.download('punkt')

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    

if __name__ == '__main__':

    dataset = MosiData()
    dataset_path = "dataset_rated/ee/"
    dataset.add_ratings_from_files(dataset_path=dataset_path)
    
    dataset.sanitize_strings()
    sentences = dataset.get_all_strings()

    dataset.classify_by_kmeans(3)
    ground_truth_list = dataset.get_all_labels()

    # Create a SentimentIntensityAnalyzer object.
    vader_model = SentimentIntensityAnalyzer()
    label_list = []
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
            print('Ground Truth: ' + str(ground_truth_list[leng-1]))

        elif sentiment_dict['compound'] <= -0.05:
            label_list.append(0)
            leng = len(label_list)
            print(" Overall: Negative")
            print('Ground Truth: ' + str(ground_truth_list[leng-1]))

        else:
            label_list.append(1)
            leng = len(label_list)
            print(" Overall: Neutral")
            print('Ground Truth: ' + str(ground_truth_list[leng-1]))

        print("")

    label_length = len(label_list)
    for index in range(0, label_length):
        if ground_truth_list[index] == 2 and label_list[index] == 2:
            review_dict['pos_pos'] += 1

        elif ground_truth_list[index] == 2 and label_list[index] == 1:
            review_dict['pos_neutral'] += 1

        elif ground_truth_list[index] == 2 and label_list[index] == 0:
            review_dict['pos_neg'] += 1

        elif ground_truth_list[index] == 1 and label_list[index] == 2:
            review_dict['neutral_pos'] += 1

        elif ground_truth_list[index] == 1 and label_list[index] == 1:
            review_dict['neutral_neutral'] += 1

        elif ground_truth_list[index] == 1 and label_list[index] == 0:
            review_dict['neutral_neg'] += 1

        elif ground_truth_list[index] == 0 and label_list[index] == 2:
            review_dict['neg_pos'] += 1

        elif ground_truth_list[index] == 0 and label_list[index] == 1:
            review_dict['neg_neutral'] += 1

        else:
            review_dict['neg_neg'] += 1

    #precision_dict = {'pos_pos': 4, 'pos_neutral': 1, 'pos_neg': 0, 'neutral_pos': 3, 'neutral_neutral': 5, 'neutral_neg': 2, 'neg_pos': 1, 'neg_neutral': 1, 'neg_neg': 4}

    pos_count = ground_truth_list.count(2)
    neutral_count = ground_truth_list.count(1)
    neg_count = ground_truth_list.count(0)
    
    #Recall and Precision values
    
    pos_recall = round((review_dict['pos_pos'])/(review_dict['pos_pos']+review_dict['pos_neutral']+review_dict['pos_neg']),2)
    pos_precision = round((review_dict['pos_pos'])/(review_dict['pos_pos']+review_dict['neutral_pos']+review_dict['neg_pos']),2)

    neutral_recall = round((review_dict['neutral_neutral'])/(review_dict['neutral_neutral']+review_dict['neutral_pos']+review_dict['neutral_neg']),2)
    neutral_precision = round((review_dict['neutral_neutral'])/(review_dict['neutral_neutral']+review_dict['pos_neutral']+review_dict['neg_neutral']),2)

    neg_recall = round((review_dict['neg_neg'])/(review_dict['neg_neg']+review_dict['neg_pos']+review_dict['neg_neutral']),2)
    neg_precision = round((review_dict['neg_neg'])/(review_dict['neg_neg']+review_dict['pos_neg']+review_dict['neutral_neg']),2)

    #F1 value
    pos_f1 = round(2 * ((pos_precision * pos_recall) / (pos_precision + pos_recall)), 2)
    neutral_f1 = round(2 * ((neutral_precision * neutral_recall) / (neutral_precision + neutral_recall)), 2)
    neg_f1 = round(2 * ((neg_precision * neg_recall) / (neg_precision + neg_recall)), 2)

    #weight avg
    #weighted_average_count = review_dict['pos_pos'] + review_dict['neutral_neutral'] + review_dict['neg_neg']
    weighted_average_count = pos_count + neutral_count + neg_count
    weighted_average_precision = round((((pos_precision * pos_count) +
                                         (neutral_precision * neutral_count) +
                                         (neg_precision * neg_count))/weighted_average_count), 2)
    weighted_average_recall = round((((pos_recall * pos_count) +
                                      (neutral_recall * neutral_count) +
                                      (neg_recall * neg_count))/weighted_average_count), 2)
    weighted_average_f1 = round((((pos_f1 * pos_count) +
                                  (neutral_f1 * neutral_count) +
                                  (neg_f1 * neg_count))/weighted_average_count), 2)

    confusion_matrix = [[pos_precision, pos_recall, pos_f1, review_dict['pos_pos']],
                        [neutral_precision, neutral_recall, neutral_f1, review_dict['neutral_neutral']],
                        [neg_precision, neg_recall, neg_f1, review_dict['neg_neg']],
                        [weighted_average_precision, weighted_average_recall, weighted_average_f1, 'N/A']
                        ]
    
    print(tabulate([['pos', pos_precision, pos_recall, pos_f1, pos_count],
                    ['neutral', neutral_precision, neutral_recall, neutral_f1, neutral_count],
                    ['neg', neg_precision, neg_recall, neg_f1, neg_count],
                    ['w_avg', weighted_average_precision, weighted_average_recall, weighted_average_f1, weighted_average_count]
                    ], headers=['Class', 'Precision', 'Recall', 'F1', 'Count']))


