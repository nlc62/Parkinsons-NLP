import argparse

from vader_final import vader_dataset
from vader_final import run_vader
from vader_final import confusion_metrics

parser = argparse.ArgumentParser(description='VADER model creation')

parser.add_argument('dataset', action='store', choices=['ms', 'm', 'pd', 'you'],
                    help='Datasets: movie review sentences, movie review, parkinsons, youtube')

parser.add_argument('-t', '--num_texts', action='store', type=int, default='100',
                    help='Sample size for dataset evaluation')

parser.add_argument('-m', '--metrics', action='store_true',
                    help='Calculate confusion matrix metrics.')

options = parser.parse_args()

dataset_name = options.dataset
n_texts = options.num_texts
calc_metrics = options.metrics

sentences, ground_truth_list = vader_dataset(dataset_name, n_texts)

review_dict = run_vader(sentences, ground_truth_list)
print(review_dict)

if calc_metrics == True:
    confusion_metrics = confusion_metrics(review_dict)
    print(confusion_metrics)




    

