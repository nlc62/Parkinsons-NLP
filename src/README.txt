
To run:
python create_models.py


1. Data directories

    dataset_master/: Contains all 448 interview transcripts obtained from the
        occupational health study (including baseline interviews and bi-monthly
        follow-up interviews). Each interview is stored in unprocessed text for-
        mat. These files are used in the prediction scripts to amass the text
        corpus used to train the LDA model.

    dataset_rated/: Contains all ratings given for Parkinsons per-sentence rating data
        (directory "ee").  The label data is not straightforward - since the rating data
        was collected online in batches, each directory in "ee" represents raters who
        rated a particular group of Parkinsons transcripts.

        Also contains the sentences and labels for movie reviews (directory "mr").
        Sentences are in the .subj directory, and labels are separated into the other
        directories depending on what is wanted (10 point scale, 3 class, 4 class, etc).

2. Scripts

    pipeline/create_models.py: Main script, parses datasets and creates the machine learning model.

    pipeline/utilities.py: Extra functions placed in here for various uses.

    pipeline/dataset.py: Abstract class for holding data in sentence and label format.

    pipeline/clickdata.py: Implementation of dataset.py, used for Parkinsons data.

    pipeline/moviereview.py Implementation of dataset.py, used for movie review data.
