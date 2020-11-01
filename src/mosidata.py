import os
from dataset import Dataset, TextData
import pickle

class ReviewData(TextData):
    # Unique ID for this review comes from "html filename."
    def __init__(self, id, string, rating):
        super(ReviewData, self).__init__(id, string)
        self.rating = rating

    def get_rating(self):
        return [self.rating]

    def get_min_max_rating(self):
        return [[-3.0, 3.0]]

    def get_class(self, class_center_pts):
        # No centers means REGRESSION
        if not class_center_pts:
            return 0

        # Distance formula: sqrt( (x1-x2)^2 + (y1-y2)^2 )
        distances = [abs(self.get_rating()[0] - center[0]) for center in class_center_pts]  # [0] for valence
        closest_distance = min(distances)
        classification = distances.index(closest_distance)
        return classification

    def get_reviewer_name(self):
        return self.reviewer_name



class MosiData(Dataset):
    def __init__(self):
        super(MosiData, self).__init__()

    # Purpose: Fills the class with the proper data
    #          (Only uses file "movieReviewSnippets_GroundTruth.txt").
    # Inputs: dataset_path is the path to the directory
    #         where the "vaderSentiment" directory is located
    #         (with extra '/' at end).
    # Returns: nothing
    def add_ratings_from_files(self, dataset_path):

        #Initializing data structures
        labels = []
        ids = []
        label_dict = {}
        text_dict = {}
        
        #Extracting labels in dict
        with open(dataset_path + "Mosi/mosi_data.pkl", 'rb') as infile:

            objects = pickle.load(infile, encoding= 'bytes')

            for key in objects.keys():
                id_list = objects[key]['id'].tolist()
                label_list = objects[key]['labels'].tolist()
                for i in id_list:
                    iD = i[0].decode()
                    ids.append(iD)
            
                for label in label_list:
                    for j in label:
                        labels.append(j[0])
            
            for index in range(len(labels)):
                label_dict[ids[index]] = labels[index]

        #Extracting text
        path = dataset_path + "Mosi/Raw/Transcript/Segmented/"
        file_names = sorted(os.listdir(path))
        
        for text in file_names:
            #Text Paths
            text_path = path + text
            identifier = text[:11]

            with open(text_path) as text_file:
                sentences = text_file.read()
                sentences = sentences.split('\n')
                for sentence in sentences:
                    index = sentence.find('_')
                    index_ = sentence.find('_', index+1)
                    text_dict[identifier + '_' + sentence[:index]] = sentence[index_ + 1:]

        #Match labels with text       
        for k in label_dict.keys():
            id = k
            rating = label_dict[k]
            string = text_dict[k]
            self.text_data.append(ReviewData(id, str(string), float(rating)))

    def flatten_by_valence(self):
        return

    def get_all_strings_for_lda(self):
        return self.get_all_strings()
    
    # Returns: [var, var, ...]
    def get_all_variances(self):
        return [1 for _ in self.text_data]

    def classify_by_class3(self):
        self.class_center_pts = [[-1.0, 0], [0.0, 0], [1.0, 0]]
