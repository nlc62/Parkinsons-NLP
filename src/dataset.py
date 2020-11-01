# REVISION HISTORY
# 20AUG19   mgold Created Dataset and TextData as abstract classes


from abc import ABCMeta, abstractmethod
from sklearn.cluster import KMeans
import math
import random
import sys
import matplotlib.pyplot as plt


class TextData(object):
    __metaclass__ = ABCMeta

    def __init__(self, id, string):
        self.id = id  # int
        self.string = string  # string

    # Purpose: Get the rating.
    # Inputs: Nothing.
    # Returns: List of float ratings, eg. [x, y, ...]
    @abstractmethod
    def get_rating(self):
        pass

    # Purpose: Gets the smallest and largest possible rating values.
    # Inputs: Nothing.
    # Returns: [[xMIN, xMAX], [yMIN, yMAX], ...]
    @abstractmethod
    def get_min_max_rating(self):
        pass

    # Purpose: Get the unique id for this datapoint.
    # Inputs: Nothing.
    # Returns: Integer unique id.
    def get_id(self):
        return self.id

    # Purpose: Get the sentence.
    # Inputs: Nothing.
    # Returns: String sentence.
    def get_string(self):
        return self.string

    # Purpose: Get the normalized rating, where each coordinate in the rating
    #          is scaled from its min-max range to fit in [0, 1].
    # Inputs: Nothing.
    # Returns: List of float normalized ratings, eg. [xNorm, yNorm, ...]
    def get_normalized_rating(self):
        normalized_rating = []
        for coord, min_max in zip(self.get_rating(), self.get_min_max_rating()):
            zero_oriented_rating = coord - min_max[0]  # Shift the coord to range [0, max]
            scalar = 1.0 / abs(min_max[0] - min_max[1])  # Scaling ratio
            norm = zero_oriented_rating * scalar
            normalized_rating.append(norm)
        return normalized_rating

    # Purpose: Get the classification number in range "[0, num_classes)".
    # Inputs: List of class center point coordinates, where each class center
    #         point is a list of values (eg. valence, arousal, etc).
    #         eg. [[x1, y1, ...], [x2, y2, ...], ...], where x is valence,
    #         y is arousal, etc.  
    # Returns: Integer label in range "[0, num_classes)"
    def get_label(self, class_center_pts):
        # No class
        if not class_center_pts:
            return 0

        # Distance formula: sqrt( (x1-x2)^2 + (y1-y2)^2 + ... )
        distances = []
        for ctrpt in class_center_pts:
            sum = 0.0
            for r, p in zip(self.get_rating(), ctrpt):
                sum += (float(r) - float(p)) ** 2
            d = math.sqrt(sum)
            distances.append(d)
        
        # Index of smallest distance = classification
        closest_distance = min(distances)
        label = distances.index(closest_distance)
        return label

    # Purpose: Get the number of words in the string sentence.
    # Inputs: Nothing.
    # Returns: Number of words in the sentence.
    def get_word_count(self):
        string = self.get_string().replace('\n', '').replace('.', '').replace(';', ',')
        num_words = len(string.split())
        return num_words

    # Purpose: Changes the string sentence to only contain safe characters,
    #          eg. no newlines and no strange apostrophes.
    # Inputs: Nothing.
    # Returns: Nothing.
    def sanitize(self):
        self.string = "".join(char for char in self.get_string() if ord(char) < 128)
        self.string = self.get_string().replace('\n', '').replace('\r', '').replace('.', '')


class Dataset(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, dataset_path=None):
        self.text_data = []  # FORM: [TextData, TextData, ...]
        self.class_center_pts = []  # FORM: [[x, y, ...], [x, y, ...], ...]  RANGE: normalized [0, 1]
        if dataset_path is not None:
            self.add_ratings_from_files(dataset_path)

    #######################
    #     CHANGE DATA     #
    #######################
    # Purpose: Adds ratings, sentences, and any other data from files.
    # Inputs: The relative path from place where running to
    #         the input dataset base folder.
    # Returns: Nothing.
    @abstractmethod
    def add_ratings_from_files(self, dataset_path):
        pass

    # Purpose: Removes all rating values (eg. "arousal"), except "valence."
    # Inputs: Nothing.
    # Returns: Nothing.
    @abstractmethod
    def flatten_by_valence(self):
        pass

    # Purpose: Convenience method for doing common 3-class classification.
    #          Intended for setting the class center points to the particular
    #          dataset's hard-coded 3-class mapping of ratings to classes.
    # Inputs: Nothing.
    # Returns: Nothing.
    @abstractmethod
    def classify_by_class3(self):
        pass

    # Purpose: Convenience method for doing 7-class emotional classification.
    #          Intended for setting the class center points to the particular
    #          dataset's hard-coded 7-class mapping of emotional ratings to
    #          classes.
    # Inputs: Nothing.
    # Returns: Nothing.
    @abstractmethod
    def classify_by_class7(self):
        pass
    
    # Purpose: Classifies ratings by the input class center points.
    # Inputs: List of class center point coordinates, where each class center
    #         point is a list of values (eg. valence, arousal, etc).
    #         eg. [[x1, y1, ...], [x2, y2, ...], ...], where x is valence,
    #         y is arousal, etc.  The number of class center points is the
    #         number of classes to classify by.
    # Returns: Nothing.
    def classify_by_input(self, centers):
        self.class_center_pts = centers

    # Purpose: Classifies ratings by unsupervised k-means clustering.
    #          Sets class center points to the result.
    # Inputs: The number of classes for k-means clustering to discover.
    # Returns: Nothing
    def classify_by_kmeans(self, n_classes):
        kmeans = KMeans(n_clusters=n_classes, random_state=0)
        kmeans.fit(self.get_all_ratings())

        # Sorts by x-value (Equivalent to sorting by valence)
        self.class_center_pts = sorted(kmeans.cluster_centers_.tolist())

        # Normalize
        # centers = [[(val + 100.0) / 200.0 for val in center] for center in centers]
        # self.class_center_pts = centers

    # Only use ASCII chars (some files have non-ASCII apostrophes)
    # Returns: nothing
    def sanitize_all_strings(self):
        for text in self.text_data:
            text.sanitize()

    ############################
    #     FILTERED DATASET     #
    ############################
    # Purpose: Gets new dataset without any data of the input class.
    # Inputs: The integer representing the class to remove.
    # Returns: Dataset without data classified to the input class.
    def filter_out_class(self, class_num):
        filtered_data = type(self)()
        filtered_data.class_center_pts = self.get_class_center_pts()
        for text in self.text_data:
            if text.get_label(self.get_class_center_pts()) != class_num:
                filtered_data.text_data.append(text)
        return filtered_data

    def filter_duplicate_data(self):
        filtered_data = type(self)()
        filtered_data.class_center_pts = self.get_class_center_pts()
        duplicate_data = type(self)()
        duplicate_data.class_center_pts = self.get_class_center_pts()
        sentence_set = set()
        for text in self.text_data:
            if text.get_string() not in sentence_set:
                filtered_data.text_data.append(text)
                sentence_set.add(text.get_string())
            else:
                duplicate_data.text_data.append(text)
        return filtered_data, duplicate_data

    #Paramter: Filters out sentences not in the specified character length range
    #Returns: Dataset with sentences only in the specified character length
    def filter_sentence_length(self, min_char_length, max_char_length):
        filtered_data = type(self)()
        filtered_data.class_center_pts = self.get_class_center_pts()
        for text in self.text_data:
            if len(text.get_string()) > min_char_length:
                if len(text.get_string()) < max_char_length:
                    filtered_data.text_data.append(text)       
        return filtered_data

    ###############
    #     GET     #
    ###############

    # Purpose: Gets all string "documents" to be used for LDA models in
    #          the "tokenization" step.  It is intended for documents to
    #          be multiple sentences grouped by file, review, etc.
    # Inputs: Nothing.
    # Returns: List of string "documents," where a document is one or more
    #          sentences as a single string, eg. [doc1, doc2, ...]
    @abstractmethod
    def get_all_strings_for_lda(self):
        pass

    # Purpose: Gets all texts, in original input order.
    # Inputs: Nothing.
    # Returns: List of sentences, eg. [txt1, txt2, ...]
    def get_all_strings(self):
        return [text.get_string() for text in self.text_data]

    # All coordinates, in order
    # Returns: [[[x, y], [x, y], ...]
    def get_all_ratings(self):
        return [text.get_rating() for text in self.text_data]

    def get_all_emotions(self):
        return [text.get_emotion_labels() for text in self.text_data]

    def get_all_complete_emotion_matches(self, y_pred_emotions, empty_pred):
        output_list = []
        for g,p in zip(self.text_data, y_pred_emotions):
            #Exclude empty predictions
            if empty_pred == False:
                if len(p) > 0:
                    output_list.append(g.is_complete_match(p))

            #Include empty predictions
            else:
                output_list.append(g.is_complete_match(p))
            
        return output_list

    def get_all_emotion_matches(self, y_pred_emotions, empty_pred):
        output_dict = {}
        for g, p in zip(self.text_data, y_pred_emotions):
            #Exclude empty predictions
            if empty_pred == False:
                if len(p) > 0:
                    output_dict[g] = []
                    for label in p:
                        if g.has_emotion_label(label):
                            output_dict[g].append(True)
                        else:
                            output_dict[g].append(False)                    
            else:
                output_dict[g] = []
                if len(p) == 0:
                    output_dict[g].append(g.is_complete_match(p))
                else:
                    for label in p:
                        if g.has_emotion_label(label):
                            output_dict[g].append(True)
                        else:
                            output_dict[g].append(False)
        
        output_list = []
        for key in output_dict.keys():
            if True in output_dict[key]:
                output_list.append(True)
            else:
                output_list.append(False)

        return output_list
            
    # Gives all valence values, in order
    # Returns: [float, float, ...]
    def get_all_valences(self):
        return [0 if len(text.get_rating()) <= 0 else text.get_rating()[0] for text in self.text_data]

    def get_all_arousals(self):
        return [0 if len(text.get_rating()) <= 1 else text.get_rating()[1] for text in self.text_data]

    # Returns: [mse, mse, ...]
    def get_all_mean_squared_errors(self, prediction_list):
        return [(prediction - actual) ** 2 for prediction, actual in zip(prediction_list, self.get_all_valences())]
    
    # Purpose: Gets all variances.  This is a meaningful evaluation if there
    #          is more than one rater per sentence.
    # Inputs: Nothing.
    # Returns: List of float variances, eg. [var, var, ...]
    def get_all_variances(self):
        return [1 for _ in self.text_data]

    # Returns: [label, label, ...]
    def get_all_labels(self):
        return [text.get_label(self.get_class_center_pts()) for text in self.text_data]

    # Returns: [acc, acc, ...]
    def get_all_accuracies(self, prediction_list):
        mse_list = self.get_all_mean_squared_errors(prediction_list)
        var_list = self.get_all_variances()
        accuracy_list = [mse / var for mse, var in zip(mse_list, var_list)]
        return accuracy_list

    # Returns: [[x, y, ...], [x, y, ...], ...]
    def get_class_center_pts(self):
        return self.class_center_pts

    # Returns: int
    def get_n_texts(self):
        return len(self.text_data)

    # NOTE: If no regression (no classes), still returns 1: technically, everything is in at least "one" class.
    # Returns: int
    def get_n_classes(self):
        return max(len(self.get_class_center_pts()), 1)

    # How many texts fall into each class, organized in class order
    # Returns: [class0_count, class1_count, ...]
    def get_class_counts(self):
        class_counts = [0 for _ in range(self.get_n_classes())]
        for text in self.text_data:
            class_counts[text.get_label(self.get_class_center_pts())] += 1
        return class_counts

    #################
    #     SPLIT     #
    #################
    # Splits data in two.  One dataset has constant number of sentences per class.  Other dataset has the remainder.
    def get_only_n_texts_per_class(self, ideal_max_per_class, rerandomize=False):
        selected_data = type(self)()
        extra_data = type(self)()
        selected_data.class_center_pts = self.get_class_center_pts()
        extra_data.class_center_pts = self.get_class_center_pts()

        # Can't select from more than there are in a given class
        total_per_class = self.get_class_counts()
        n_chosen_per_class = [min(ideal_max_per_class, total_per_class[label]) for label in range(self.get_n_classes())]

        # Randomize sentence selection, per class
        if rerandomize:
            seed = random.randrange(sys.maxsize)
            random.seed(seed)
            print("*** New random train/test split: random seed = {}".format(seed))
        else:
            seed = 33
            random.seed(seed)
            print("*** Constant train/test split: seed = {}".format(seed))

        # Choose texts
        text_ids = [random.sample(range(0, total), chosen) for total, chosen in zip(total_per_class, n_chosen_per_class)]
        # sentence_ids = [random.sample(range(0, limit_per_class[i]), max_per_class[i]) for i in range(len(max_per_class))]
        text_id_counters = [0 for _ in range(self.get_n_classes())]

        # Split data
        for text in self.text_data:
            label = text.get_label(self.get_class_center_pts())
            # Selected
            if text_id_counters[label] in text_ids[label]:
                selected_data.text_data.append(text)
            # Extra
            else:
                extra_data.text_data.append(text)
            text_id_counters[label] += 1
        return selected_data, extra_data

    # def get_only_n_texts_per_class(self, ideal_max_per_class):
    #     equalized_data, extra_data = self.get_only_n_texts_per_class(ideal_max_per_class)
    #     return equalized_data

    # Each class ends up with an equal number of sentences
    def get_only_equal_texts_per_class(self):
        ideal_max_per_class = min(self.get_class_counts())
        equalized_data, extra_data = self.get_only_n_texts_per_class(ideal_max_per_class)
        return equalized_data

    # Equal number in training, rest in testing
    def training_testing_split(self, test_split_percent, rerandomize=False):
        # Test percent of total sentences, split among each class
        ideal_max_per_class = int(test_split_percent * self.get_n_texts() / self.get_n_classes())
        testdata, traindata = self.get_only_n_texts_per_class(ideal_max_per_class, rerandomize)
        return traindata, testdata

    ################
    #     PLOT     #
    ################
    def plot_data(self, plot_name=""):
        # Datapoints
        x_points = [[] for _ in range(self.get_n_classes())]
        y_points = [[] for _ in range(self.get_n_classes())]
        colors = ["b", "r", "g", "c", "m", "y", "k", "w"]

        # Class Centers
        x_centers = [center[0] for center in self.get_class_center_pts()]
        y_centers = [center[1] for center in self.get_class_center_pts()]
        color_center = "#000000"

        # Collect all x,y values
        for label, x, y in zip(self.get_all_labels(), self.get_all_valences(), self.get_all_arousals()):
            x_points[label].append(x)
            y_points[label].append(y)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Plot
        for x_vals, y_vals, color in zip(x_points, y_points, colors):
            ax.scatter(x=x_vals, y=y_vals, color=color, marker=".")
        ax.scatter(x_centers, y_centers, color=color_center, marker="o")

        # Setup display
        # ax.set_xlim([-105, 105])
        # ax.set_ylim([-105, 105])
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_title(plot_name + " ({})".format(self.get_n_texts()))
        fig.show()

    def plot_valences(self, hist_columns=200, plot_name=""):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Plot
        ax.hist(x=self.get_all_valences(),
                bins=hist_columns,
                color="b")

        # Display
        ax.set_xlabel("Valence")
        ax.set_ylabel("Texts Per Rating")
        ax.set_title(plot_name + " ({})".format(self.get_n_texts()))
        fig.show()

    def plot_actual_vs_predicted(self, predicted_list, plot_name=""):
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Points on this line are perfect predictions.
        ideal_line = plt.Line2D([-100, 100], [-100, 100], color="k")
        ax.add_artist(ideal_line)

        # All points
        ax.scatter(x=self.get_all_valences(), y=predicted_list, color="#FF00FF", marker=".")

        # Setup window
        ax.set_xlim([-105, 105])
        ax.set_ylim([-105, 105])
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(plot_name + " ({})".format(self.get_n_texts()))
        fig.show()
