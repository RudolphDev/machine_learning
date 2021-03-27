import math
import numpy as np
import pandas as pd
from tabulate import tabulate

def open_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    split_lines = []
    for line in lines:
        split_lines.append(line.split())
    return split_lines


def get_splited_class(data, class_num):
    data_class = []
    for line in data:
        if int(line[0]) == class_num:
            data_class.append(line)
    return data_class


def get_unique_class_num(data):
    class_num = []
    for line in data:
        class_num.append(line[0])
    return np.unique(class_num)

def compute_one_euclidian_dist(first_point, second_point):
    sum_square = 0
    for i in range(0, len(first_point)):
        sum_square = (float(first_point[i]) - float(second_point[i]))**2      
    return math.sqrt(sum_square)

def get_top_n_decision(n, theo_class, dists):
    dists_sorted = sorted(dists.items(), key=lambda kv: kv[1])
    cut_dists = dists_sorted[0:n]
    top_n_result = False
    for dist in cut_dists:
        if theo_class == dist[0]:
            top_n_result = True
    return top_n_result


def update_confusion_matrix(conf_matrix, line_class, scores_dict):
    row_num = int(line_class) - 1
    temp = min(scores_dict.values())
    res = [key for key in scores_dict if scores_dict[key] == temp]
    col_num = int(res[0]) - 1
    conf_matrix[row_num, col_num] = conf_matrix[row_num, col_num] + 1


def transform_matrix_to_df(conf_matrix, class_names):
    df = pd.DataFrame(conf_matrix, index=class_names,
                      columns=class_names, dtype=int)
    return df


def print_decision_model_result(nb_app, nb_dec, top1_rate, top2_rate, conf_matrix):
    print("\tResults :")
    print("\t----------------")
    print("\tNumber of elements for the learning step : ", nb_app)
    print("\tNumber of elements for the decision step : ", nb_dec)
    print("\t----------------")
    print("\n\tTop results :")
    print("\t----------------")
    print("\tTop 1 rate : ", top1_rate)
    print("\tTop 2 rate : ", top2_rate)
    print("\t----------------")
    print("\n\tConfusion matrix :")
    print("\t----------------")
    print(tabulate(conf_matrix, headers='keys', tablefmt='fancy_grid'))
    print("\t----------------")
