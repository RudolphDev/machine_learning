import math
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import Counter
import matplotlib.pyplot as plt


class GeneralModel:
    def __init__(self):
        self._train_data = []
        self._count_top1 = 0
        self._count_top2 = 0
        self._conf_matrix = []
        self._len_test_data = 0
        self._unique_classes = []

    @property
    def train_data(self):
        return self._train_data

    @property
    def count_top1(self):
        return self._count_top1

    @count_top1.setter
    def count_top1(self, count):
        self._count_top1 = count

    @property
    def count_top2(self):
        return self._count_top2

    @count_top2.setter
    def count_top2(self, count):
        self._count_top2 = count

    @property
    def conf_matrix(self):
        return self._conf_matrix

    @property
    def len_test_data(self):
        return self._len_test_data

    def compute_one_euclidian_dist(first_point, second_point):
        sum_square = 0
        for i in range(0, len(first_point)):
            sum_square = (float(first_point[i]) - float(second_point[i]))**2
        return math.sqrt(sum_square)

    def get_unique_class_num(self):
        class_num = []
        for line in self._train_data:
            class_num.append(line[0])
        self._unique_classes = np.unique(class_num)

    def print_model_result(self):
        print("Results :")
        print("----------------")
        print("Number of elements for the learning step : ", len(self._train_data))
        print("Number of elements for the decision step : ", self._len_test_data)
        print("----------------")
        print("\nTop results :")
        print("----------------")
        print("Top 1 rate : ", self._count_top1/self._len_test_data)
        print("Top 2 rate : ", self._count_top2/self._len_test_data)
        print("----------------")
        print("\nConfusion matrix :")
        print("----------------")
        print(tabulate(self._conf_matrix, headers='keys', tablefmt='fancy_grid'))
        print("----------------")

    def update_confusion_matrix(self, line_class, scores_dict):
        row_num = int(line_class) - 1
        temp = min(scores_dict.values())
        res = [key for key in scores_dict if scores_dict[key] == temp]
        col_num = int(res[0]) - 1
        self._conf_matrix[row_num,
                          col_num] = self._conf_matrix[row_num, col_num] + 1

    def show_train_plot(self):
        pd_train = pd.DataFrame(self._train_data)
        scatter = plt.scatter(x=pd.to_numeric(pd_train[1]), y=pd.to_numeric(
            pd_train[2]), c=pd.to_numeric(pd_train[0]))
        plt.legend(*scatter.legend_elements(), loc="lower right")
        plt.show()


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


def get_top_n_decision(n, theo_class, dists):
    dists_sorted = sorted(dists.items(), key=lambda kv: kv[1])
    cut_dists = dists_sorted[0:n]
    top_n_result = False
    for dist in cut_dists:
        if theo_class == dist[0]:
            top_n_result = True
    return top_n_result


def get_n_max_occurence_from_list(classes_list: list, n_max: int = 1):
    counted_list = list(Counter(classes_list).items())
    counted_list.sort(key=lambda a: a[1], reverse=True)
    result = counted_list[n_max-1][0]
    return result


def transform_matrix_to_df(conf_matrix, class_names):
    df = pd.DataFrame(conf_matrix, index=class_names,
                      columns=class_names, dtype=int)
    return df
