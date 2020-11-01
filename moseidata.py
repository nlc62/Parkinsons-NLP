import os
import csv
import sys
from dataset import Dataset, TextData
import math

class ReviewData(TextData):
    # Purpose: Initialization.
    # Inputs: Integer unique id,
    #         String sentence,
    #         Float sentiment rating value,
    #         List of string emotional labels (eg. ["sadness", "joy"]).
    # Returns: Nothing.
    def __init__(self, id, string, rating, emotion_labels):
        super(ReviewData, self).__init__(id, string)
        self.rating = rating
        self.emotion_labels = emotion_labels
        
    def get_rating(self):
        return [self.rating]

    def get_emotion_labels(self):
        return self.emotion_labels
    
    def get_emotion_label(self):
        return self.emotion_labels.index(max(self.emotion_labels))

    def has_emotion_label(self, label_index):
        return bool(self.emotion_labels[label_index])

    def is_complete_match(self, labels):

        emotion_count = 0
        for i in range(len(self.emotion_labels) -1):  #Exclude 7th class
            if self.emotion_labels[i] != 0:
                emotion_count += 1

        #Check if number of labels match
        if emotion_count != len(labels):
            return False

        #No emotion label case
        if self.emotion_labels[6] == 1 and len(labels) == 0:
            return True

        #Cross check labels
        for ind in range(len(labels)):
            if self.has_emotion_label(labels[ind]) == False:
                return False
            
        return True

    def get_min_max_rating(self):
        return [[-3.0, 3.0]]

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



class MoseiData(Dataset):
    def __init__(self, dataset_path=None):
        super(MoseiData, self).__init__(dataset_path)
    
    # Purpose: Fills the class with the proper data
    #          (Uses files "dev_sent_emo.csv"
    #                      "test_sent_emo.csv
    #                      "train_sent_emo.csv).
    # Inputs: dataset_path is the path to the directory
    #         where the "MELD.RAW" directory is located
    #         (with extra '/' at end).
    # Returns: Nothing
    def add_ratings_from_files(self, dataset_path):
        text_path = dataset_path + "Transcript/Segmented/Combined/"
        label_path = dataset_path + "Labels/"

        text_filenames = sorted(os.listdir(text_path))
        label_filenames = sorted(os.listdir(label_path))

        # Compile all sentences
        text_dict = {}
        for f in text_filenames:
            with open(text_path + f) as infile:
                # Each line contains:
                # DOC_ID, SENTENCE_NUM, TIMESTAMP_START, TIMESTAMP_END, SENTENCE
                for line in infile:
                    line = line.replace("_____", "")  # At least one line has underscores in the sentence...
                    doc_id, text_index, _, _, sentence = line.split("___")
                    text_dict[str(doc_id) + "_" + str(text_index)] = str(sentence)
        
        # Compile all labels (sentiment and emotions)
        rating_dict = {}
        emo_dict = {}
        for f in label_filenames:
            with open(label_path + f) as infile:
                labels = list(csv.reader(infile))[1:]

                for l in labels:
                    doc_id = l[27]
                    text_index = l[28]
                    sentiment = l[38]

                    # emotion labels, CHANGE ORDER:
                    # 29:anger, 30:disgust, 31:fear, 33:happiness, 34:sadness, 39:surprise
                    # 31:fear, 34:sadness, 29:anger, 33:happiness, 30:disgust, 39:surprise
                    anger = l[29]
                    disgust = l[30]
                    fear = l[31]
                    happiness = l[33]
                    sadness = l[34]
                    surprise = l[39]

                    # Some lines are partially empty... ignore these "bad label" lines.
                    if "" in [sentiment, anger, disgust, fear, happiness, sadness, surprise]:
                        continue

                    # Add new label if doesn't exist (there are repeats, 3 raters per sentence)
                    key = str(doc_id) + "_" + str(text_index)
                    if key not in rating_dict:
                        rating_dict[key] = []
                    if key not in emo_dict:
                        emo_dict[key] = []
                    
                    rating_dict[key].append(int(sentiment))
                    emo_dict[key].append([int(fear), int(sadness), int(anger), int(happiness), int(disgust), int(surprise)])

        # Average out multiple sentiment and emotion scores per sentence
        for k in rating_dict.keys():
            rating_dict[k] = float(sum(rating_dict[k])) / float(len(rating_dict[k]))
        for k in emo_dict.keys():
            #Set emotion threshold to 1
            emo_dict[k] = [float(sum(v)) / float(len(v)) if (float(sum(v)) / float(len(v)) >= 0.5) else 0.0 for v in zip(*emo_dict[k])]
            #emo_dict[k] = [float(sum(v)) / float(len(v)) for v in zip(*emo_dict[k])]
            
        # Remove mismatches (texts without labels, labels without texts)
        valid_keys = list(set(text_dict.keys()) & set(rating_dict.keys()) & set(emo_dict.keys()))

        # Add neutral emotion for emotionless ratings
        for labels in emo_dict.keys():
            if emo_dict[labels].count(0) == 6:
                emo_dict[labels].append(1.0)
            else:
                emo_dict[labels].append(0.0)
                
        # Add ratings to ReviewData
        for k in valid_keys:
            id = k
            sentence = text_dict[k]
            rating = rating_dict[k]
            emotion_labels = emo_dict[k]
            self.text_data.append(ReviewData(id, str(sentence), float(rating), list(emotion_labels)))

    def flatten_by_valence(self):
        return

    def get_all_strings_for_lda(self):
        return self.get_all_strings()

    def classify_by_class3(self):
        #[valence, arousal]
        self.class_center_pts = [[-1.0, 0], [0.0, 0], [1.0, 0]]

    def classify_by_class7(self):
        self.class_center_pts = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
