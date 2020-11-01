# REVISION HISTORY
# 20AUG19   mgold Created MovieData and ReviewData as children of "Dataset" and "TextData" abstract classes.


import os
from dataset import Dataset, TextData


class ReviewData(TextData):
    # Unique ID for this review comes from "html filename."
    def __init__(self, id, string, rating, class3, class4, reviewer_name):
        super(ReviewData, self).__init__(id, string)
        self.rating = rating
        self.reviewer_name = reviewer_name
        self.class3 = class3
        self.class4 = class4

    def get_rating(self):
        return [self.rating]

    def get_min_max_rating(self):
        return [[0.0, 1.0]]

    def get_reviewer_name(self):
        return self.reviewer_name

    def get_class3(self):
        return self.class3

    def get_class4(self):
        return self.class4


class MovieData(Dataset):
    def __init__(self, dataset_path=None):
        super(MovieData, self).__init__(dataset_path)

    # The entire review for each subjective extract in $author/subj.$author 
    # (of scale dataset v1.0) can be identified by the id number specified
    # in the correponding line of $author/id.$author and located as file
    # $author/txt.parag/$id.txt
    # where each line of $id.txt corresponds to one paragraph of the review.
    def add_ratings_from_files(self, dataset_path):
        reviewer_names = sorted(os.listdir(dataset_path))
        reviewer_paths = [dataset_path + reviewer + "/" for reviewer in reviewer_names]

        # Files are organized per user
        for reviewer_name, reviewer_path in zip(reviewer_names, reviewer_paths):
            # Paths
            ids_path = reviewer_path + "id." + reviewer_name
            reviews_path = reviewer_path + "subj." + reviewer_name
            ratings_path = reviewer_path + "rating." + reviewer_name
            class3_path = reviewer_path + "label.3class." + reviewer_name
            class4_path = reviewer_path + "label.4class." + reviewer_name

            # Lists with file info
            ids = []
            reviews = []
            ratings = []
            class3 = []
            class4 = []
            with open(ids_path) as ids_file,\
                 open(reviews_path) as reviews_file,\
                 open(ratings_path) as ratings_file,\
                 open(class3_path) as class3_file,\
                 open(class4_path) as class4_file:
                ids = ids_file.readlines()
                reviews = reviews_file.readlines()
                ratings = ratings_file.readlines()
                class3 = class3_file.readlines()
                class4 = class4_file.readlines()

            # Format data
            ids = [int(id.replace('\n', '')) for id in ids]
            reviews = [review.replace('\n', '') for review in reviews]
            ratings = [float(rating.replace('\n', '')) for rating in ratings]
            class3 = [int(label.replace('\n', '')) for label in class3]
            class4 = [int(label.replace('\n', '')) for label in class4]

            # Remove excess rating data that was placed in the paragraphs
            # eg. Everything before "rating : * ..."
            reviews = [r.split("rating : *")[0] for r in reviews]
            reviews = [r.split("* * * * = ")[0] for r in reviews]
            # reviews = [r.replace('*', '') for r in reviews]

            # Rating objects
            for id, review, rating, c3, c4 in zip(ids, reviews, ratings, class3, class4):
                self.text_data.append(ReviewData(id, review, rating, c3, c4, reviewer_name))

    def flatten_by_valence(self):
        return

    def get_all_strings_for_lda(self):
        return self.get_all_strings()

    # List of unique filenames
    # Returns: [filename1, filename2, ...]
    def get_all_filenames(self):
        return list(set([text.get_reviewer_name() for text in self.text_data]))

    # List of file contents (all strings of a file, as a list)
    # Returns: [[file1_string, file1_string, ...], [file2_string, file2_string, ...], ...]
    def get_all_strings_by_file(self):
        filenames = self.get_all_filenames()
        texts_by_file = [[] for _ in filenames]

        # Correlate filename indices with where sentences will go
        for text in self.text_data:
            file_index = filenames.index(text.get_reviewer_name())
            texts_by_file[file_index].append(text.get_string())
        return texts_by_file

    def classify_by_class3(self):
        self.class_center_pts = [[.2], [.5], [.8]]
    
    def classify_by_class4(self):
        self.class_center_pts = [.25, .45, .65, .85]
