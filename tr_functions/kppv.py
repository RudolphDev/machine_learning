# local functions import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tr_functions.general import GeneralModel


class KppvModel(GeneralModel):

    def __init__(self):
        """Create the K Nearest Neighbours model and initialize attributes
        """
        self.__k = 1
        self.__cross_val = 1
        self.__k_results_cv = []
        self.__vote_method = "unanimous"
        GeneralModel.__init__(self)

    # Getter and setter
    @property
    def k(self):
        """return the number of neighbours used in this model

        Returns:
            int: number of neighbours used in this model
        """
        return self.__k

    @k.setter
    def k(self, value: int):
        """Set the number of neighbours used in this model

        Args:
            value (int): number of neighbours to use in this model
        """
        self.__k = value

    @property
    def cross_val(self):
        """get the number of cross validation used in this model

        Returns:
            int: number of subpart used in cross validation
        """
        return self.__cross_val

    @cross_val.setter
    def cross_val(self, value: int):
        """set the number of cross validation used in this model

        Args:
            value (int): positive integer represnting the number of subparts used in the cross validation
        """
        if value > 0:
            self.__cross_val = value
        else:
            print("The cross validation value must be > 0!")

    @property
    def vote_method(self):
        """get the result vote method

        Returns:
            str: vote method used in the model
        """
        return self.__vote_method

    @vote_method.setter
    def vote_method(self, value: str):
        """set the voting method in the knn model

        Args:
            value (str): vote method name can bee "unanimous" or "majority"
        """
        if value in ("unanimous", "majority"):
            self.__vote_method = value
        else:
            print("Accepted methods are \"unanimous\" and \"majority\"")

    # Public methods
    def print_k_results_cv(self):
        """print the error rate of each k used in the model
        """
        print("Cross Validation results:")
        for k in self.__k_results_cv:
            print("For k = {} the error rate is {}".format(
                k[0], round(1-k[1], 3)))

    def compute_kppv(self, app_data: list, dec_data: list):
        """compute the knn from the training dataset, and create results from the testing dataset. use the k set in the model

        Args:
            app_data (list): 2D nested list of training data with class name as first element of the sublist
            dec_data (list): 2D nested list of testing data with class name as first element of the sublist
        """
        print("Will use k = {} neighbours".format(self.__k))
        self._train_data = app_data
        self.test_data = dec_data
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))

        for line in self._test_data:
            kppv_list = self.__get_kppv_list(line[1:])
            k_list = list(Counter(kppv_list[0]).items())
            self.__compute_vote_kppv(k_list, int(line[0]))
            k_list.sort(key=lambda a: a[1], reverse=True)
            row_num = int(line[0]) - 1
            col_num = int(k_list[0][0]) - 1
            line.append(k_list[0][0])
            self._conf_matrix[row_num,
                              col_num] = self._conf_matrix[row_num, col_num] + 1

    def get_k_cross_validation(self, app_data: list, k_max: int, cv: int):
        """find the best number of nearest neighbours to use in the model and set the private attribute

        Args:
            app_data (list): 2D nested list of training data with class name as first element of the sublist
            k_max (int): max number of neighbours to test in the cross-validation
            cv (int): number of subpart of the training dataset used in cross validation
        """
        self.__cross_val = cv
        best_k = [0, 0]
        df = pd.DataFrame(app_data)
        shuffled = df.sample(frac=1)
        cut_dfs = np.array_split(shuffled, self.__cross_val)
        for i in range(k_max):
            self.__k = i + 1
            sum_error = 0
            for i in range(self.__cross_val):
                df_cv = cut_dfs[i]
                df_train = df.drop(df_cv.index)
                df_cv = df_cv.values.tolist()
                df_train = df_train.values.tolist()
                count_top_1 = 0
                for line in df_cv:
                    kppv_list = self.__get_kppv_list(line[1:], df_train)
                    count_dict = Counter(kppv_list[0])
                    temp = max(count_dict.values())
                    res = [key for key in count_dict if count_dict[key] == temp]
                    if res[0] == line[0]:
                        count_top_1 = count_top_1 + 1
                error_rate = count_top_1/len(df_cv)
                sum_error = sum_error + error_rate
            self.__k_results_cv.append((self.__k, sum_error/self.__cross_val))
            if best_k[1] < sum_error/self.__cross_val:
                best_k[0] = self.__k
                best_k[1] = sum_error/self.__cross_val
        self.__plot_k_error_rate()
        print("The best k found is {} with a error rate = {}".format(
            best_k[0], 1-best_k[1]))
        self.k = best_k[0]

    # Private methods
    def __get_kppv_list(self, dec_point: list, train_data=None):
        """Compute all distances between dec_point and each training point then keep only the k best distances.

        Args:
            dec_point (list): coordinates of the point
            train_data (list): 2D nested list of training data with class name as first element of the sublist

        Returns:
            list: list of lists with the class number and the score in each
        """
        if train_data == None:
            train_data = self._train_data
        dist_list = []
        for app_line in train_data:
            dist_list.append(
                [app_line[0], self._compute_one_euclidian_dist(app_line[1:3], dec_point)])
        app_df = pd.DataFrame(dist_list)
        sorted_df = app_df.sort_values(by=1)
        kppv_list = sorted_df.head(self.__k)
        return kppv_list

    def __compute_vote_kppv(self, k_list: list, theo_class: int):
        """Vote if the k-length list of nearest neigjbours is identical to the theorical class

        Args:
            k_list (list): k-length list of class names find in the training dataset
            theo_class (int): class name of theorical class
        """
        k_list = [(int(k[0]), k[1]) for k in k_list]
        if self.__vote_method == "unanimous":
            self.__compute_unanimous_vote(k_list, theo_class)
        else:
            self.__compute_majority_vote(k_list, theo_class)

    def __compute_unanimous_vote(self, k_list: list, theo_class: int):
        """add +1 to top 1 count if the k_list is unanimously equal to the theorical class

        Args:
            k_list (list): k-length list of class names find in the training dataset
            theo_class (int): class name of theorical class
        """
        if len(k_list) != 1:
            self._error_count += 1
        else:
            if k_list[0][0] == theo_class:
                self._count_top1 += 1

    def __compute_majority_vote(self, k_list: list, theo_class: int):
        """add +1 to top 1 count if the k_list is unanimously equal to the theorical class

        Args:
            k_list (list): k-length list of class names find in the training dataset
            theo_class (int): class name of theorical class
        """
        k_list.sort(key=lambda a: a[1], reverse=True)
        if len(k_list) == 1:
            if k_list[0][0] == theo_class:
                self._count_top1 += 1
                self._count_top2 += 1

        elif len(k_list) == 2:
            if k_list[0][1] == k_list[1][1]:
                if k_list[0][0] == theo_class:
                    self._count_top2 += 1
            else:
                if k_list[0][0] == theo_class:
                    self._count_top1 += 1
                    self._count_top2 += 1
                elif k_list[1][0] == theo_class:
                    self._count_top2 += 1

        elif len(k_list) > 2:
            if k_list[0][1] == k_list[1][1]:
                if k_list[0][1] == k_list[2][1]:
                    self._error_count += 1
                else:
                    if k_list[0][0] == theo_class:
                        self._count_top2 += 1
            else:
                if k_list[0][0] == theo_class:
                    self._count_top1 += 1
                    self._count_top2 += 1
                else:
                    if k_list[1][1] == k_list[2][1]:
                        self._error_count += 1
                    else:
                        if k_list[1][0] == theo_class:
                            self._count_top2 += 1

        else:
            self._error_count += 1

    def __plot_k_error_rate(self):
        """plot the error rate of each k tested in the cross validation
        """
        x = [val[0] for val in self.__k_results_cv]
        y = [1-val[1] for val in self.__k_results_cv]
        plt.plot(x, y)
        plt.title("error rate for each k tested with cross-validation")
        plt.show()
