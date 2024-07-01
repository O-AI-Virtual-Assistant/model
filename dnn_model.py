from util import csv2dict, helper_collections, topk_accuarcy, CodeTimer, save_results_to_csv
from sklearn.neural_network import MLPRegressor
# from joblib import Parallel, delayed, cpu_count
from math import ceil
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import joblib
import os
import csv
import timeit
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
How Predicted Relevancy Scores are Computed

Feature Extraction:
For each Java file associated with a bug report, several features are extracted. These features typically include:
RVSM (Relevance Vector Space Model) similarity: Measures the textual similarity between the bug report and the Java file.
Collaborative Filtering (CF) score: Considers past interactions between bug reports and files.
Class Name Similarity: Measures how similar class names in the Java file are to the text in the bug report.
Bug Recency: How recently the file has been associated with a bug report.
Bug Frequency: How often the file has been associated with bug reports.

Model Training:
The DNN model (specifically, an MLPRegressor in this case) is trained on a dataset where each entry includes the features 
mentioned above and a label indicating whether the file is actually related to the bug report (match).

Prediction:
When predicting for a new bug report, the trained model takes the feature vectors of the associated Java files as input and 
outputs relevancy scores. These scores represent the model's confidence that each file is relevant to the bug report.

Top-1 Accuracy: Measures how often the correct file is the single highest-ranked file. This is the strictest criterion and 
usually has the lowest accuracy.
Top-2 Accuracy: Measures how often the correct file is among the top 2 highest-ranked files. This is less strict and 
typically has a higher accuracy than Top-1.
Top-3 to Top-20 Accuracies: Follow the same logic, where the accuracy generally increases as the criterion becomes less strict (i.e., the correct file needs to be within the top K files, where K increases).

(Top-1 Accuracy) This criterion is the strictest because only the very first prediction is considered. If the correct file is 
not ranked first, it is counted as incorrect.
Typically, this accuracy is lower because the model must be highly confident and precise to rank the correct file at the very top 
every time.

The files listed for each rank (Top-1, Top-2, etc.) are the ones that the model predicts as the most likely to be buggy.
These files are ranked by their predicted relevancy scores, with higher ranks indicating higher confidence that the file is buggy.

Summary
The code ranks files based on relevancy scores, with the highest scores expected to appear at the top.
The accuracies reported are cumulative, meaning Top-K accuracy includes the correct files being in the top K ranks.
The highest rank (Top-1) has the lowest accuracy because it is the strictest criterion. The accuracy increases for higher ranks 
(Top-2, Top-3, etc.) because the criterion becomes progressively less strict.
"""


def oversample(samples):
    """ Oversamples the features for label "1" 
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    samples_ = []

    # oversample features of buggy files
    for i, sample in enumerate(samples):
        samples_.append(sample)
        if i % 51 == 0:
            for _ in range(9):
                samples_.append(sample)

    return samples_


def features_and_labels(samples):
    """ Returns features and labels for the given list of samples
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    features = np.zeros((len(samples), 5))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        labels[i] = float(sample["match"])

    return features, labels


def train_dnn(samples, sample_dict, bug_reports, br2files_dict):
    with CodeTimer("Training DNN Model"):
        train_samples = oversample(samples)
        np.random.shuffle(train_samples)
        X_train, y_train = features_and_labels(train_samples)

        logging.info("Starting DNN training...")
        clf = MLPRegressor(
            solver="sgd",
            alpha=1e-5,
            hidden_layer_sizes=(300,),
            random_state=1,
            max_iter=10000,
            n_iter_no_change=30,
        )
        
        clf.fit(X_train, y_train.ravel())
        logging.info("DNN training completed.")

        # Save the trained model
        joblib.dump(clf, "trained_model.pkl")

        acc_dict, result_dict = topk_accuarcy(bug_reports, sample_dict, br2files_dict, clf=clf)
        logging.info("Top-K accuracy calculation completed.")

    return acc_dict, result_dict


def evaluate_model(test_samples, sample_dict, br2files_dict, model_filename):
    logging.info("Loading trained model...")
    clf = joblib.load(model_filename)
    logging.info("Model loaded.")

    logging.info("Preparing helper collections for testing set...")
    _, test_bug_reports, _ = helper_collections(test_samples)
    logging.info(f"Prepared {len(test_bug_reports)} bug reports for testing.")

    acc_dict, result_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=clf)
    logging.info("Evaluation on testing set completed.")

    return acc_dict, result_dict


def dnn_model():
    acc_dict, result_dict = {}, {}
    try:
        logging.info("Loading data...")
        samples = csv2dict("data/features.csv")
        logging.info(f"Loaded {len(samples)} samples.")

        logging.info("Splitting data into training and testing sets...")
        train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
        logging.info(f"Training samples: {len(train_samples)}, Testing samples: {len(test_samples)}")

        logging.info("Preparing helper collections...")
        sample_dict, bug_reports, br2files_dict = helper_collections(train_samples)
        logging.info(f"Prepared {len(bug_reports)} bug reports.")

        np.random.shuffle(train_samples)
        logging.info("Shuffled training samples.")

        logging.info("Starting DNN model training and evaluation...")
        acc_dict, result_dict = train_dnn(train_samples, sample_dict, bug_reports, br2files_dict)

        logging.info("Evaluating model on testing set...")
        evaluate_model(test_samples, sample_dict, br2files_dict, "trained_model.pkl")
        
        logging.info("DNN model training and evaluation completed.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    
    return acc_dict, result_dict


if __name__ == "__main__":
    acc_dict, result_dict = dnn_model()
    
    # Print only accuracies and ranks
    for rank, data in acc_dict.items():
        print(f"Top-{rank} Accuracy: {data}")

    # Save detailed results to CSV
    save_results_to_csv(result_dict, "dnn_results.csv")
    logging.info("Results saved to dnn_results.csv")