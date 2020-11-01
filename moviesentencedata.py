# REVISION HISTORY
# 16JUN20   mgold Created MovieSentenceData and ReviewData as children of "Dataset" and "TextData" abstract classes.



# FROM THE GITHUB README:
# 
# movieReviewSnippets_GroundTruth.txt
# FORMAT:
# the file is tab delimited with
# ID, MEAN-SENTIMENT-RATING, and TEXT-SNIPPET

# DESCRIPTION:
# includes 10,605 sentence-level snippets from rotten.tomatoes.com.
# The snippets were derived from an original set of 2000 movie reviews
# (1000 positive and 1000 negative) in Pang & Lee (2004);
# we used the NLTK tokenizer to segment the reviews into sentence phrases,
# and added sentiment intensity ratings. The ID and MEAN-SENTIMENT-RATING
# correspond to the raw sentiment rating data provided in
# 'movieReviewSnippets_anonDataRatings.txt' (described below).

# movieReviewSnippets_anonDataRatings.txt
# FORMAT:
# the file is tab delimited with
# ID, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-SENTIMENT-RATINGS

# DESCRIPTION: Sentiment ratings from a minimum of 20 independent
# human raters (all pre-screened, trained, and quality checked for
# optimal inter-rater reliability).

import os
from dataset import Dataset, TextData


class ReviewData(TextData):
    # Unique ID for this review comes from "html filename."
    def __init__(self, id, string, rating):
        super(ReviewData, self).__init__(id, string)
        self.rating = rating

    def get_rating(self):
        return [self.rating]

    def get_min_max_rating(self):
        return [[-4.0, 4.0]]

    def get_class(self, class_center_pts):
        # No centers means REGRESSION
        if not class_center_pts:
            return 0

        # Distance formula: sqrt( (x1-x2)^2 + (y1-y2)^2 )
        distances = [abs(self.get_normalized_rating()[0] - center[0]) for center in class_center_pts]  # [0] for valence
        closest_distance = min(distances)
        classification = distances.index(closest_distance)
        return classification

    def get_reviewer_name(self):
        return self.reviewer_name



class MovieSentenceData(Dataset):
    def __init__(self, dataset_path=None):
        super(MovieSentenceData, self).__init__(dataset_path)

    # Purpose: Fills the class with the proper data
    #          (Only uses file "movieReviewSnippets_GroundTruth.txt").
    # Inputs: dataset_path is the path to the directory
    #         where the "vaderSentiment" directory is located
    #         (with extra '/' at end).
    # Returns: nothing
    def add_ratings_from_files(self, dataset_path):
        path = dataset_path + "additional_resources/hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.txt"

        with open(path) as infile:
            for line in infile:
                line = line.strip("\n\r")
                id, rating, string = line.split(None, 2)
                self.text_data.append(ReviewData(id, str(string), float(rating)))
                print(str(string))
                print(float(rating))

    def flatten_by_valence(self):
        return

    def get_all_strings_for_lda(self):
        return self.get_all_strings()
    
    # Returns: [var, var, ...]
    def get_all_variances(self):
        return [1 for _ in self.text_data]

    def classify_by_class3(self):
        #self.class_center_pts = [[-.1, 0], [.0, 0], [.1, 0]]
        #self.class_center_pts = [[-2, 0], [0, 0], [2, 0]]
        self.class_center_pts = [[-2.67, 0], [0.0, 0], [2.67, 0]]
