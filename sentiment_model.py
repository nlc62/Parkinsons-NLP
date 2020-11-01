import argparse
import sys
from vader_final import vader_dataset
from vader_final import run_vader
from vader_final import confusion_metrics
from tone_analyzer_final import run_tone_analyzer

parser = argparse.ArgumentParser(description='VADER model creation')

parser.add_argument('model', action='store', choices=['vader', 'stanford-corenlp', 'tone-analyzer'],
                    help='Models: vader, stanford-corenlp, IBM tone analyzer (tone-analyzer) runs on MELD and Mosei dataset')

parser.add_argument('dataset', action='store', choices=['ms', 'm', 'pd', 'mosi', 'meld', 'mosei'],
                    help='Datasets: movie review sentences (ms), movie review (m), \
                    parkinsons (pd), mosi, meld (only use for tone-analyzer)')

parser.add_argument('-t', '--num_texts', action='store', type=int,
                    help='Number of texts for each class to evaluate')

parser.add_argument('-e', '--no_empty_emotions', action='store_false',
                    help='Exclude empty tone analyzer predictions')

parser.add_argument('-m', '--metrics', action='store_true',
                    help='Calculate confusion matrix metrics for vader.')

options = parser.parse_args()

model = options.model
dataset_name = options.dataset
n_texts = options.num_texts
empty_pred = options.no_empty_emotions
calc_metrics = options.metrics

if model == 'vader':
    
    sentences, ground_truth_list = vader_dataset(dataset_name, n_texts)
    review_dict = run_vader(sentences, ground_truth_list)
    print(review_dict)

    if calc_metrics == True:
        confusion_metrics = confusion_metrics(review_dict)
        print(confusion_metrics)

elif model == 'stanford-corenlp':
    #Input stanford script
    pass

elif model == 'tone-analyzer':
    if dataset_name == 'meld':
        accuracy, anger, fear, sadness, joy, missed = run_tone_analyzer(dataset_name, n_texts, empty_pred)
        print('Accuracy: ' + str(accuracy))
        print('Anger Accuracy: ' + str(anger))
        print('Fear Accuracy: ' + str(fear))
        print('Sadness Accuracy: ' + str(sadness))
        print('Joy Accuracy: ' + str(joy))
        print('No-Emotion Percentage: ' + str(missed))
        

    else:
        match_accuracy, fear, sadness, anger, happiness, complete_match_accuracy, fear_comp, sadness_comp, anger_comp, happiness_comp, missed = run_tone_analyzer(dataset_name, n_texts, empty_pred)
        print('Match Accuracy: ' + str(match_accuracy))
        print('Fear Match Accuracy: ' + str(fear))
        print('Sadness Match Accuracy: ' + str(sadness))
        print('Anger Match Accuracy: ' + str(anger))
        print('Happiness Match Accuracy: ' + str(happiness))
        print('Complete Match Accuracy: ' + str(complete_match_accuracy))
        print('Fear Complete Match Accuracy: ' + str(fear_comp))
        print('Sadness Complete Match Accuracy: ' + str(sadness_comp))
        print('Anger Complete Match Accuracy: ' + str(anger_comp))
        print('Happiness Complete Match Accuracy: ' + str(happiness_comp))
        print('No-Emotion Percentage: ' + str(missed))
    




