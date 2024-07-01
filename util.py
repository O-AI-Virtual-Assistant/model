import csv
import os
import timeit
import numpy as np
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Utility functions
def tsv2dict(tsv_path):
    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        line["files"] = [
            os.path.normpath(f[8:])
            for f in line["files"].strip().split()
            if f.startswith("bundles/") and f.endswith(".java")
        ]
        line["raw_text"] = line["summary"] + line["description"]
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )
        dict_list.append(line)
    return dict_list

def csv2dict(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)
    return csv_dict

def helper_collections(samples, only_rvsm=False):
    sample_dict = {}
    for s in samples:
        sample_dict[s["report_id"]] = []

    for s in samples:
        temp_dict = {}
        values = [float(s["rVSM_similarity"])]
        if not only_rvsm:
            values += [
                float(s["collab_filter"]),
                float(s["classname_similarity"]),
                float(s["bug_recency"]),
                float(s["bug_frequency"]),
            ]
        temp_dict[os.path.normpath(s["file"])] = values
        sample_dict[s["report_id"]].append(temp_dict)

    bug_reports = tsv2dict("data/Eclipse_Platform_UI.txt")
    br2files_dict = {}
    for bug_report in bug_reports:
        br2files_dict[bug_report["id"]] = bug_report["files"]
    return sample_dict, bug_reports, br2files_dict

def topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf):
    print("in topk accuracy")
    topk_counters = [0] * 20
    topk_files = [[] for _ in range(20)]
    negative_total = 0
    
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = bug_report["id"]
        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except:
            negative_total += 1
            continue

        relevancy_list = clf.predict(dnn_input) if clf else []

        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list, -i)[-i:]
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    topk_files[i - 1].append(corresponding_file)
                    break

    acc_dict = {}
    result_dict = {}
    for i, (counter, files) in enumerate(zip(topk_counters, topk_files)):
        acc = counter / (len(test_bug_reports) - negative_total)
        acc_dict[i + 1] = round(acc, 3)
        result_dict[i + 1] = {'accuracy': round(acc, 3), 'files': files}

    return acc_dict, result_dict

def save_results_to_csv(result_dict, filename="results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Accuracy", "Files"])
        for rank, data in result_dict.items():
            files_str = "; ".join(data['files'])
            writer.writerow([rank, data['accuracy'], files_str])

class CodeTimer:
    def __init__(self, message=""):
        self.message = message
    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()
    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs.".format(self.took))



# import csv
# import os
# import timeit
# import numpy as np
# from datetime import datetime
# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')

# def tsv2dict(tsv_path):
#     """ Converts a tab separated values (tsv) file into a list of dictionaries

#     Arguments:
#         tsv_path {string} -- path of the tsv file
#     """
#     reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
#     dict_list = []
#     for line in reader:
#         line["files"] = [
#             os.path.normpath(f[8:])
#             for f in line["files"].strip().split()
#             if f.startswith("bundles/") and f.endswith(".java")
#         ]
#         line["raw_text"] = line["summary"] + line["description"]
#         line["report_time"] = datetime.strptime(
#             line["report_time"], "%Y-%m-%d %H:%M:%S"
#         )

#         dict_list.append(line)
#     return dict_list


# def csv2dict(csv_path):
#     """ Converts a comma separated values (csv) file into a dictionary

#     Arguments:
#         csv_path {string} -- path to csv file
#     """
#     with open(csv_path, "r") as f:
#         reader = csv.DictReader(f, delimiter=",")
#         csv_dict = list()
#         for line in reader:
#             csv_dict.append(line)

#     return csv_dict


# def helper_collections(samples, only_rvsm=False):
#     """ Generates helper function for calculations
    
#     Arguments:
#         samples {list} -- samples from features.csv
    
#     Keyword Arguments:
#         only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False})
#     """
#     sample_dict = {}
#     for s in samples:
#         sample_dict[s["report_id"]] = []

#     for s in samples:
#         temp_dict = {}

#         values = [float(s["rVSM_similarity"])]
#         if not only_rvsm:
#             values += [
#                 float(s["collab_filter"]),
#                 float(s["classname_similarity"]),
#                 float(s["bug_recency"]),
#                 float(s["bug_frequency"]),
#             ]
#         temp_dict[os.path.normpath(s["file"])] = values

#         sample_dict[s["report_id"]].append(temp_dict)

#     bug_reports = tsv2dict("data/Eclipse_Platform_UI.txt")
#     br2files_dict = {}

#     for bug_report in bug_reports:
#         br2files_dict[bug_report["id"]] = bug_report["files"]

#     # print("br2files: ", br2files_dict)

#     return sample_dict, bug_reports, br2files_dict


# def topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf):
#     """ Calculates top-k accuracies
    
#     Arguments:
#         test_bug_reports {list of dictionaries} -- list of all bug reports
#         sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
#         br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv" pairs
    
#     Keyword Arguments:
#         clf {object} -- A classifier with 'predict()' function. If None, rvsm relevancy is used. (default: {None})
#     """
#     print("in topk accuracy")

#     topk_counters = [0] * 20
#     topk_files = [[] for _ in range(20)]
#     negative_total = 0
    
#     for bug_report in test_bug_reports:
#         dnn_input = []
#         corresponding_files = []
#         bug_id = bug_report["id"]

#         try:
#             for temp_dict in sample_dict[bug_id]:
#                 java_file = list(temp_dict.keys())[0]
#                 features_for_java_file = list(temp_dict.values())[0]
#                 dnn_input.append(features_for_java_file)
#                 corresponding_files.append(java_file)
#         except:
#             negative_total += 1
#             continue

#         relevancy_list = clf.predict(dnn_input) if clf else []

#         for i in range(1, 21):
#             max_indices = np.argpartition(relevancy_list, -i)[-i:]
#             for corresponding_file in np.array(corresponding_files)[max_indices]:
#                 if str(corresponding_file) in br2files_dict[bug_id]:
#                     topk_counters[i - 1] += 1
#                     topk_files[i - 1].append(corresponding_file)
#                     break

#     acc_dict = {}
#     result_dict = {}
#     for i, (counter, files) in enumerate(zip(topk_counters, topk_files)):
#         acc = counter / (len(test_bug_reports) - negative_total)
#         acc_dict[i + 1] = round(acc, 3)
#         result_dict[i + 1] = {'accuracy': round(acc, 3), 'files': files}

#     return acc_dict, result_dict

# def save_results_to_csv(result_dict, filename="results.csv"):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Rank", "Accuracy", "Files"])
#         for rank, data in result_dict.items():
#             files_str = "; ".join(data['files'])  # Join file names with '; ' separator
#             writer.writerow([rank, data['accuracy'], files_str])


# class CodeTimer:
#     """ Keeps time from the initalization, and print the elapsed time at the end.

#         Example:

#         with CodeTimer("Message"):
#             foo()
#     """

#     def __init__(self, message=""):
#         self.message = message

#     def __enter__(self):
#         print(self.message)
#         self.start = timeit.default_timer()

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.took = timeit.default_timer() - self.start
#         print("Finished in {0:0.5f} secs.".format(self.took))
