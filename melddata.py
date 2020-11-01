import csv
import sys
import math
import os
from dataset import Dataset, TextData


class ReviewData(TextData):
    # Unique ID for this review comes from "html filename."
    def __init__(self, id, string, rating, emotion_label):
        super(ReviewData, self).__init__(id, string)
        self.rating = rating
        self.emotion_label = emotion_label
        
    def get_rating(self):
        return [self.rating]

    def get_emotion_label(self):
        emotions_7 = ['fear', 'sadness', 'anger', 'joy', 'surprise', 'neutral', 'disgust']
        emotion_class = emotions_7.index(self.emotion_label)
        return emotion_class
    
    def get_min_max_rating(self):
        return [[0.0, 2.0]]

    def get_label(self, class_center_pts):
        # No class
        if not class_center_pts:
            return 0

        if len(class_center_pts) == 7:
            return self.get_emotion_label()

        # Distance formula: sqrt( (x1-x2)^2 + (y1-y2)^2 + ... )
        distances = []
        for point in class_center_pts:
            sum = 0
            for r, p in zip(self.get_rating(), point):
                sum += (float(r) - float(p)) ** 2
            d = math.sqrt(sum)
            distances.append(d)
        
        # Index of smallest distance = classification
        closest_distance = min(distances)
        label = distances.index(closest_distance)
        return label

    def get_reviewer_name(self):
        return self.reviewer_name



class MELDdata(Dataset):
    def __init__(self):
        super(MELDdata, self).__init__()

    def load_data(self, dataset, dataset_path):
        """
        Loads the three MELD datasets dev, test, and train
        Returns: ground truth list in 3 classes with corresponding utterances and utterance id
        """
    
        with open(dataset_path + "MELD.RAW/" + str(dataset) + "_sent_emo.csv") as infile:
            bad_text = {}
            labels = []
            text = []
            sentences = []
            ground_truth = []
            emotions = []
            utterance_id = []
            dataset_list = list(csv.reader(infile))

            for sentence in dataset_list:
                labels.append(sentence[4])
                text.append(sentence[1])
                utterance_id.append(sentence[6])
                emotions.append(sentence[3])
                
            labels = labels[1:]
            text = text[1:]
            utterance_id = utterance_id[1:]
            emotions = emotions[1:]

            for label in labels:
                if label == 'negative':
                    ground_truth.append(0)
                elif label == 'neutral':
                    ground_truth.append(1)
                else:
                    ground_truth.append(2)

            if dataset == 'dev':
                for string in text:
                    utterance = string.encode('latin1').decode('cp1252')
                    sentences.append(utterance)

            else:
                for string in text:
                    #Filter out characters not in latin1 codec
                    if string.find('…') != -1 or string.find('—') != -1:
                        index = text.index(string)
                        bad_text[index] = string
            
                    else:
                        new_string = string.replace("’", "\x92")
                        utterance = new_string.encode('latin1').decode('cp1252')
                        sentences.append(utterance)
                        
                for keys in bad_text.keys():
                    sentences.insert(keys, bad_text[keys])
                    
        return ground_truth, sentences, utterance_id, emotions
    
    # Purpose: Fills the class with the proper data
    #          (Uses files "dev_sent_emo.csv"
    #                      "test_sent_emo.csv
    #                      "train_sent_emo.csv).
    # Inputs: dataset_path is the path to the directory
    #         where the "MELD.RAW" directory is located
    #         (with extra '/' at end).
    # Returns: nothing
    def add_ratings_from_files(self, dataset_path):
        
        dev_ground_truth, dev_sentences, dev_utterance_id, dev_emotions = self.load_data('dev', dataset_path)
        test_ground_truth, test_sentences, test_utterance_id, test_emotions = self.load_data('test', dataset_path)
        train_ground_truth, train_sentences, train_utterance_id, train_emotions = self.load_data('train', dataset_path)

        ground_truth = []
        ground_truth.extend(dev_ground_truth)
        ground_truth.extend(test_ground_truth)
        ground_truth.extend(train_ground_truth)

        sentences = []
        sentences.extend(dev_sentences)
        sentences.extend(test_sentences)
        sentences.extend(train_sentences)

        utterance_id = []
        utterance_id.extend(dev_utterance_id)
        utterance_id.extend(test_utterance_id)
        utterance_id.extend(train_utterance_id)

        emotion_labels = []
        emotion_labels.extend(dev_emotions)
        emotion_labels.extend(test_emotions)
        emotion_labels.extend(train_emotions)
        
        for index in range(len(ground_truth)):
            id = utterance_id[index]
            string = sentences[index]
            rating = ground_truth[index]
            emotion = emotion_labels[index]
            self.text_data.append(ReviewData(id, str(string), float(rating), str(emotion)))

    def flatten_by_valence(self):
        return

    def get_all_strings_for_lda(self):
        return self.get_all_strings()

    def classify_by_class3(self):
        #[valence, arousal]
        #self.class_center_pts = [[-0.1, 0], [0, 0], [0.1, 0]]
        self.class_center_pts = [[0.0, 0], [1.0, 0], [2.0, 0]]

    def classify_by_class7(self):
        self.class_center_pts = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
