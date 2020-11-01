# REVISION HISTORY
# 30AUG19   mgold Converted ClickData and SentenceData into children of "Dataset" and "TextData" abstract classes.


__author__ = "Andy Valenti & Michael Gold"
__copyright__ = "Copyright 2019. Tufts University"

# imports
import numpy
import collections
import csv
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt
import warnings
import random
import sys
import copy
import numpy as np
from dataset import Dataset, TextData

# warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# warnings.filterwarnings(action='ignore', category=FutureWarning)


# Individual user rating for a given sentence
class UserRating(object):
    def __init__(self, x, y):
        # x and y are between -100 and 100
        self.x = float(x)
        self.y = float(y)

    def get_neutral_threshold_distance(self, neutral_threshold_percent, radial, proportioned_radial):
        # Case: No neutral zone (1neg, 2pos)
        if neutral_threshold_percent == 0:
            difference = -1

        # Case: Neutral radial zone (0neu, 1neg, 2pos)
        elif radial and not proportioned_radial:
            distance = sqrt((self.x ** 2) + (self.y ** 2))  # distance formula (cartesian -> polar "r" value)
            difference = abs(abs(neutral_threshold_percent) - abs(distance))

        # Case: Neutral proportioned zone, neutral as percent of width at y-level of circle (0neu, 1neg, 2pos)
        elif radial and proportioned_radial:
            # Width of circle at y (arousal) level
            radius = 100
            circle_width = sqrt(radius ** 2 - self.y ** 2)  # pythagorean
            circle_width_ratio = circle_width / radius  # as a percent
            neutral_threshold_scaled = neutral_threshold_percent * circle_width_ratio
            difference = abs(abs(neutral_threshold_scaled) - abs(self.x))

        # Case: Neutral non-radial zone (0neu, 1neg, 2pos)
        else:
            difference = abs(abs(neutral_threshold_percent) - abs(self.x))

        return difference

    #######
    # GET #
    #######
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_normalized_x(self):
        return (self.x + 100.0) / 200.0

    def get_normalized_y(self):
        return (self.y + 100.0) / 200.0

    #######
    # SET #
    #######
    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y


class SentenceData(TextData):
    # Unique ID for this review comes from "html filename."
    def __init__(self, id, string, filename):
        super(SentenceData, self).__init__(id, string)
        self.filename = filename
        self.user_ratings = []

    def add_rating(self, user_rating):
        self.user_ratings.append(user_rating)

    def flatten_ratings_by_valence(self):
        for rating in self.user_ratings:
            rating.y = 0.0

    def get_mean_rating_obj(self):
        mean_x = sum([rating.get_x() for rating in self.user_ratings]) / len(self.user_ratings)
        mean_y = sum([rating.get_y() for rating in self.user_ratings]) / len(self.user_ratings)
        rating_obj = UserRating(mean_x, mean_y)
        return rating_obj

    def get_filename(self):
        return self.filename

    # Mean of individual ratings
    def get_rating(self):
        rating_obj = self.get_mean_rating_obj()
        return [rating_obj.get_x(), rating_obj.get_y()]

    def get_min_max_rating(self):
        return [[-100.0, 100.0], [-100.0, 100.0]]

    # Range 0 to 100
    def get_standard_deviation(self):
        x_vals = [rating.get_x() for rating in self.ratings]
        y_vals = [rating.get_y() for rating in self.ratings]
        x_mean = self.get_mean_rating_obj().get_x()
        y_mean = self.get_mean_rating_obj().get_y()

        x_distances = [(x - x_mean) for x in x_vals]
        y_distances = [(y - y_mean) for y in y_vals]

        # VARIANCE and AROUSAL
        distances = [sqrt(x ** 2 + y ** 2) for x, y in zip(x_distances, y_distances)]

        variances = [d ** 2 for d in distances]
        denom = max(len(variances) - 1, 1)
        variances_mean = sum(variances) / denom
        dev = sqrt(variances_mean)

        # dev = numpy.std(valences) ** 2
        return dev

    def get_variance(self):
        variances = [((self.get_rating()[0] - rating.get_x()) ** 2) for rating in self.user_ratings]
        variances = (sum(variances) / len(variances))
        return variances



# ALL CLASSIFICATION DATA
# [
#     SENTENCE(sentence_string)
#     [
#         RATING(x, y)
#         RATING(x, y)
#         RATING...
#     ]
#     SENTENCE(sentence_string)
#     [
#         RATING(x, y)
#         RATING(x, y)
#         RATING...
#     ]
#     SENTENCE...
# ]
class ClickData(Dataset):
    def __init__(self, dataset_path=None):
        self.full_transcripts = []
        super(ClickData, self).__init__(dataset_path)

    # Purpose: CSV user ratings turn into Python object "ClickData"
    # Inputs: The path to the base parkinsons dataset directory,
    #         float neutral_threshold and boolean ratio for creating UserRating classification
    # Outputs: "ClickData" object
    def add_ratings_from_files(self, dataset_path):
        full_transcripts_path = dataset_path + "full_transcripts/"
        csv_path = dataset_path + "csv/"
        batches_path = dataset_path + "clickdata/"
        batch_names = sorted(os.listdir(batches_path))
        sentence_id_counter = 0

        # Full transcripts (for LDA)
        for (dirpath, dirnames, filenames) in os.walk(full_transcripts_path):
            for f in filenames:
                with open(os.path.join(dirpath, f), "r") as infile:
                    self.full_transcripts.append(infile.read())




        
        # for b in batch_names:
        #     # Filenames
        #     with open(dataset_path + "csv/" + b + ".csv", "r") as infile:
        #         table = list(csv.reader(infile))
        #         files_string = table[1][15].rsplit(".txt")
        #         filenames = [((name[1:] + ".txt") if (name[0] == " ") else (name + ".txt")) for name in files_string[:-1]]
        #         filenames = [filenames[i].strip() for i in range(len(filenames))]  # Remove any leading/trailing spaces from filename
        #         filenames = [f.replace(",", "") for f in filenames]  # Remove commas

        #     # Sentences
        #     user_names = os.listdir(dataset_path + batches_path + b + "/")
        #     with open(dataset_path + "clickdata/" + b + "/" + user_names[0] + "/lines.txt", "r") as infile:
        #         sentences = infile.readlines()[20:]
            
        #     # Ratings
        #     for u in user_names:
        #         with open(dataset_path + "clickdata/" + b + "/" + u + "/eval_click.csv", "r") as infile:
        #             table = list(csv.reader(infile))
        #             for row in table:
                        





        # Every batch of user-rated sentences
        for batch_name in batch_names:
            users_path = batches_path + batch_name + "/"
            user_names = os.listdir(users_path)

            # Collect filenames into list
            with open(csv_path + batch_name + ".csv") as infile:
                table = list(csv.reader(infile))

            files_string = table[1][15].rsplit(".txt")
            filenames = [((name[1:] + ".txt") if (name[0] == " ") else (name + ".txt")) for name in files_string[:-1]]
            filenames = [filenames[i].strip() for i in range(len(filenames))]  # Remove any leading/trailing spaces from filename
            filenames = [f.replace(",", "") for f in filenames]  # Remove commas


            current_doc = 0

            # Sentence strings
            with open(users_path + user_names[0] + "/lines.txt") as infile:
                sentences = infile.readlines()[20:]

            # Inherited: All user data in this batch.  Triple list in form [userIndex[sentenceIndex[x, y, counter??]]].
            # FORM: [user1[line1[x, y, counter(?)], line2[x, y, counter(?)], ...], user2[...], ...]
            users_clickdata = []
            for user_name in user_names:
                filename = users_path + user_name + "/eval_click.csv"
                try:
                    with open(filename) as clicks_csv:
                        users_clickdata.append(list(csv.reader(clicks_csv)))
                except IOError:
                    print("File not found: {}".format(filename))
                    continue

            # Create SentenceData objects
            for i, sentence_string in enumerate(sentences):
                filename = filenames[current_doc]

                # Don't add this sentence
                if sentence_string == "You have finished evaluating a transcript. Click to continue to the next transcript.\n":
                    current_doc += 1
                    continue

                # Case: Sentence already rated, add UserRatings to that Sentence object
                # Select sentence from already-created SentenceData with same filename
                bool = False
                for text in self.text_data:
                    if text.get_filename() == filenames[current_doc]:
                        if text.get_string() == sentence_string:
                            for user in users_clickdata:
                                x = user[i][0]
                                y = user[i][1]
                                user_rating = UserRating(x, y)
                                text.add_rating(user_rating)
                                bool = True

                if bool == False:
                    # Case: Add sentence
                    # Add Sentence object and its UserRating objects
                    sentence_data = SentenceData(id=sentence_id_counter,
                                                 string=sentence_string,
                                                 filename=filename)
                    sentence_id_counter += 1
                    for user in users_clickdata:
                        x = user[i][0]
                        y = user[i][1]
                        user_rating = UserRating(x, y)
                        sentence_data.add_rating(user_rating)
                    self.text_data.append(sentence_data)

    def flatten_by_valence(self):
        for text in self.text_data:
            text.flatten_ratings_by_valence()
    
    def classify_by_class3(self):
        self.classify_by_kmeans(3)

    # List of unique filenames
    # Returns: [filename1, filename2, ...]
    def get_all_filenames(self):
        return list(set([text.get_filename() for text in self.text_data]))

    # List of file contents (all strings of a file, as a list)
    # Returns: [[file1_string, file1_string, ...], [file2_string, file2_string, ...], ...]
    def get_all_strings_by_file(self):
        filenames = self.get_all_filenames()
        texts_by_file = [[] for _ in filenames]

        # Correlate filename indices with where sentences will go
        for text in self.text_data:
            file_index = filenames.index(text.get_filename())
            texts_by_file[file_index].append(text.get_string())
        return texts_by_file

    # One long string per file
    def get_all_strings_for_lda(self):
        return self.full_transcripts

    # Returns: [var, var, ...]
    def get_all_variances(self):
        return [text.get_variance() for text in self.text_data]

    ###################
    ###################
    #     FILTERS     #
    ###################
    ###################
    # Purpose: Removes sentences with ratings below agreement threshold
    def get_only_agreed_clickdata(self, min_required_agreement_percent):
        agreed_clickdata = ClickData()
        agreed_clickdata.class_center_pts = self.class_center_pts

        for sentence in self.text_data:
            if sentence.get_agreement_percent(self.class_center_pts) >= min_required_agreement_percent:
                agreed_clickdata.text_data.append(sentence)
        return agreed_clickdata

    def get_only_undeviating_clickdata(self, max_deviation):
        filtered_clickdata = ClickData()
        filtered_clickdata.class_center_pts = self.class_center_pts

        for sentence in self.text_data:
            if sentence.get_standard_deviation() <= max_deviation:
                filtered_clickdata.text_data.append(sentence)
        return filtered_clickdata

    def get_only_sentences_above_min_wordcount(self, min_words):
        filtered_clickdata = ClickData()
        filtered_clickdata.class_center_pts = self.class_center_pts
        
        for sentence in self.text_data:
            if min_words <= sentence.get_original_word_count():
                filtered_clickdata.text_data.append(sentence)
        return filtered_clickdata

    def get_only_sentences_by_name(self, sentence_list):
        filtered_clickdata = ClickData()
        filtered_clickdata.class_center_pts = self.class_center_pts
        for sentence in self.text_data:
            if sentence.get_sentence_string() in sentence_list:
                filtered_clickdata.text_data.append(sentence)
        return filtered_clickdata

    #################
    #################
    #     WRITE     #
    #################
    #################
    def write_sentences_to_csv(self, csv_output_filename, neutral_threshold, radial):
        with open(csv_output_filename, "w") as outfile:
            output_writer = csv.writer(outfile)
        header = [
            "Filename",
            "Sentence Number",
            "Sentence",
            "Actual Mean Class",
            "Agreement Percent",
        ]
        output_writer.writerow(header)
        previous_filename = ""
        sentence_in_file_counter = 0
        for sentence_data in self.text_data:
            if sentence_data.get_filename() != previous_filename:
                sentence_in_file_counter = 0
            row = [
                sentence_data.get_filename(),
                sentence_in_file_counter,
                sentence_data.get_sentence_string(),
                sentence_data.get_mean_class(neutral_threshold, radial),
                sentence_data.get_agreement_percent(),
            ]
            previous_filename = sentence_data.get_filename()
            output_writer.writerow(row)

    ################
    ################
    #     PLOT     #
    ################
    ################
    def plot_all_data(self, plot_name):
        num_lists = max(len(self.get_available_classes()), 1)

        annotation_list = [[] for _ in range(num_lists)]
        stdev_per_sentence = [[] for _ in range(num_lists)]
        x_points = [[] for _ in range(num_lists)]
        y_points = [[] for _ in range(num_lists)]
        x_all_points_per_sentence = [[] for _ in range(num_lists)]
        y_all_points_per_sentence = [[] for _ in range(num_lists)]
        x_mean_per_sentence = [[] for _ in range(num_lists)]
        y_mean_per_sentence = [[] for _ in range(num_lists)]
        colors = ["b", "r", "g", "c", "m", "y", "k", "w"]

        x_centers = [center[0] for center in self.class_center_pts]
        y_centers = [center[1] for center in self.class_center_pts]
        color_center = "#000000"

        # Collect all x,y values
        # CLASSIFICATION
        for sentence in self.text_data:
            for rating in sentence.ratings:
                label = rating.get_label(self.class_center_pts)

                annotation_list[label].append(sentence.get_sentence_string())
                stdev_per_sentence[label].append(sentence.get_standard_deviation())
                x_points[label].append(rating.get_x())
                y_points[label].append(rating.get_y())
                x_all_points_per_sentence[label].append([user.get_x() for user in sentence.ratings])
                y_all_points_per_sentence[label].append([user.get_y() for user in sentence.ratings])
                x_mean_per_sentence[label].append(sentence.get_mean_rating_obj().get_x())
                y_mean_per_sentence[label].append(sentence.get_mean_rating_obj().get_y())

        ############
        # PLOTTING #
        ############
        global ax
        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax = fig.add_subplot(111)

        # Plot
        line_list = []
        for i in range(num_lists):
            line_list.append(ax.scatter(x_points[i], y_points[i], color=colors[i], marker="."))
        center_pts = ax.scatter(x_centers, y_centers, color=color_center, marker="o")

        # Circle and grid
        boundary_circle = plt.Circle((0, 0), 100, color="k", fill=False, clip_on=False)
        x_line = plt.Line2D([-100, 100], [0, 0], color="k")
        y_line = plt.Line2D([0, 0], [-100, 100], color="k")
        ax.add_artist(boundary_circle)
        ax.add_artist(x_line)
        ax.add_artist(y_line)
        boundary_circle.set_visible(True)
        x_line.set_visible(True)
        y_line.set_visible(True)

        #############
        # Animation #
        #############
        global annot_box
        global stdev_circle
        global sentence_ratings_plot
        global mean_rating_plot
        global figure_is_frozen

        annot_box = ax.annotate("", xy=(0, 0), xytext=(11, 11), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
        stdev_circle = plt.Circle((0, 0), .2, color="k", fill=False, clip_on=False)
        sentence_ratings_plot = ax.scatter(0, 0, color="#00ff88", marker="o")
        mean_rating_plot = ax.scatter(0, 0, color="#0088ff", marker="o")

        ax.add_artist(annot_box)
        ax.add_artist(stdev_circle)
        ax.add_artist(sentence_ratings_plot)
        ax.add_artist(mean_rating_plot)

        annot_box.set_visible(False)
        stdev_circle.set_visible(False)
        sentence_ratings_plot.set_visible(False)
        mean_rating_plot.set_visible(False)

        figure_is_frozen = False

        ##########################
        # Annotate on mouse-over #
        ##########################
        def on_hover(event):
            global ax
            global annot_box
            global stdev_circle
            global sentence_ratings_plot
            global mean_rating_plot
            global figure_is_frozen

            if figure_is_frozen:
                return

            currently_hovering = False
            for line, annot, stdev, x, y, x_all, y_all, x_mean, y_mean in \
                    zip(line_list, annotation_list, stdev_per_sentence, x_points, y_points,
                        x_all_points_per_sentence, y_all_points_per_sentence, x_mean_per_sentence, y_mean_per_sentence):
                if line.contains(event)[0]:
                    index = line.contains(event)[1]["ind"]
                    print("\n\n\n========================================")
                    for i in index:
                        print("({}, {}): {}".format(x[i], y[i], annot[i].replace("\n", "")))
                    index = index[0]  # Select only the first point for that location

                    # Display annotation (Cannot display non-ASCII!)
                    annot_box.xy = (x[index], y[index])
                    try:
                        annot_box.set_text("".join(c for c in annot[index] if ord(c) < 128))
                        annot_box.set_visible(True)
                    except:
                        print("Invalid character: Cannot display on plot.")
                        annot_box.set_visible(False)

                    # Display stdev circle
                    stdev_circle.center = (x_mean[index], y_mean[index])
                    stdev_circle.set_radius(stdev[index])
                    stdev_circle.set_visible(True)

                    # # Display other user ratings
                    xy_all = np.array(zip(x_all[index], y_all[index]))
                    sentence_ratings_plot.set_offsets(xy_all)
                    sentence_ratings_plot.set_visible(True)

                    # Display mean rating
                    xy_mean = np.array([x_mean[index], y_mean[index]])
                    mean_rating_plot.set_offsets(xy_mean)
                    mean_rating_plot.set_visible(True)

                    # Make rest of plot invisible
                    for line in line_list:
                        line.set_visible(False)
                    center_pts.set_visible(False)

                    currently_hovering = True

            if not currently_hovering:
                annot_box.set_visible(False)
                stdev_circle.set_visible(False)
                sentence_ratings_plot.set_visible(False)
                mean_rating_plot.set_visible(False)

                # Make rest of plot visible
                for line in line_list:
                    line.set_visible(True)
                center_pts.set_visible(True)

            fig.canvas.draw_idle()

        def on_click(event):
            global figure_is_frozen
            figure_is_frozen = not figure_is_frozen

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        fig.canvas.mpl_connect("button_press_event", on_click)

        # Setup display
        ax.set_xlim([-105, 105])
        ax.set_ylim([-105, 105])
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_title(plot_name)

        fig.show()

    # PLOT STDEV
    def plot_stdevs(self):
        stdevs = [sentence.get_standard_deviation() for sentence in self.text_data]
        stdevs = sorted(stdevs)
        colors = ["b", "r", "g", "c", "m", "y", "k", "w"]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Plot
        ax.scatter(range(len(stdevs)), stdevs, color=colors[1], marker=".")

        # Setup display
        ax.set_xlim([-1, len(stdevs)])
        ax.set_ylim([-1, 101])
        ax.set_xlabel('Sentences')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Std Dev')
        fig.show()

    def plot_mean_data_colored_stdevs(self):
        rgb_list = []
        x_list = []
        y_list = []

        # Collect all x,y values
        for sentence in self.text_data:
            color_ratio = sentence.get_standard_deviation() / 100.0
            rgb_list.append([0, color_ratio, color_ratio])
            x_list.append(sentence.get_mean_coords()[0])
            y_list.append(sentence.get_mean_coords()[1])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Plot
        boundary_circle = plt.Circle((0, 0), 100, color="k", fill=False, clip_on=False)
        x_line = plt.Line2D([-100, 100], [0, 0], color="k")
        y_line = plt.Line2D([0, 0], [-100, 100], color="k")
        ax.scatter(x_list, y_list, color=rgb_list, marker=".")

        # Setup display
        ax.set_xlim([-105, 105])
        ax.set_ylim([-105, 105])
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("Colored by Standard Deviation (Black is low stdev, blue is high stdev)")
        fig.show()
