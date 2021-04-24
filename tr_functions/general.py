
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import Counter
import matplotlib.pyplot as plt
import math


class GeneralModel:
    """General Model class to handle general data and general functions
    """

    def __init__(self):
        """Create the Model with all these attributes
        """
        self._train_data = []
        self._test_data = []
        self._count_top1 = 0
        self._count_top2 = 0
        self._error_count = 0
        self._conf_matrix = []
        self._unique_classes = []

    # Getter and setter
    @property
    def train_data(self):
        """Return the data used to train the model.

        Returns:
            Pandas DataFrame: dataframe with each value and label of the training dataset
        """
        return pd.DataFrame(self._train_data)

    @train_data.setter
    def train_data(self, train_data):
        self._train_data = train_data
    
    @property
    def count_top1(self):
        """Return the number of classes right labeled

        Returns:
            int: number of classes right labeled
        """
        return self._count_top1

    @count_top1.setter
    def count_top1(self, count: int):
        self._count_top1 = count

    @property
    def count_top2(self):
        return self._count_top2

    @count_top2.setter
    def count_top2(self, count: int):
        self._count_top2 = count

    @property
    def conf_matrix(self):
        return self._conf_matrix

    @property
    def test_data(self):
        return pd.DataFrame(self._test_data)

    @test_data.setter
    def test_data(self, test_data):
        self._test_data = test_data
    
    # Public methods
    @staticmethod
    def open_file(filename: str):
        """Open the data file. Create a list from each line.

        Args:
            filename (str): the datafile path

        Returns:
            list: each line into a list
        """
        with open(filename) as f:
            lines = f.readlines()
        split_lines = []
        for line in lines:
            split_lines.append(line.split())
        return split_lines

    def print_model_result(self):
        """Print the results of the model prediction for the test data used.
        """
        print("Results :")
        print("----------------")
        print("Number of elements for the learning step : ", len(self._train_data))
        print("Number of elements for the decision step : ", len(self._test_data))
        print("----------------")
        print("\nTop results :")
        print("----------------")
        print("Top 1 rate : ", self._count_top1/len(self._test_data))
        if self._count_top2 != 0:
            print("Top 2 rate : ", self._count_top2/len(self._test_data))
        print("----------------")
        print("\nConfusion matrix :")
        print("----------------")
        pd_cf = pd.DataFrame(self._conf_matrix, index=self._unique_classes, columns=self._unique_classes, dtype=int)
        print(tabulate(pd_cf, headers='keys', tablefmt='fancy_grid'))
        print("----------------")

    def plot_train_data(self):
        """Plot the dataset used for the training
        """
        self._add_train_point_to_plot()
        plt.show()

    def plot_all_data(self):
        pd_train = pd.DataFrame(self._train_data)
        pd_test = pd.DataFrame(self._test_data)
        
        scatter = plt.scatter(x=pd.to_numeric(pd_train[1]), y=pd.to_numeric(
            pd_train[2]), c=pd.to_numeric(pd_train[0]), marker="+", alpha=0.5)
        scatter2 = plt.scatter(x=pd.to_numeric(pd_test[1]), y=pd.to_numeric(
            pd_test[2]), c=pd.to_numeric(pd_test[0]), marker=".")
        plt.legend(*scatter.legend_elements(), loc="lower right")
        plt.legend(*scatter2.legend_elements(), loc="lower left", title="test")
        plt.show()
        
    def plot_test_data(self):
        self._add_test_point_to_plot()
        plt.show()
        
    def print_conf_matrix(self):
        """ Print the confusion matrix with real classes as rows and found classes as columns
        """
        df = pd.DataFrame(self._conf_matrix, index=self._unique_classes, columns=self._unique_classes, dtype=int)
        print(df) 
        
    # Protected methods
    @staticmethod
    def _compute_one_euclidian_dist(first_point:list, second_point:list):
        """Compute the euclidian distance between first and second points

        Args:
            first_point (list): values of the first point
            second_point (list): values of the second point

        Returns:
            float: the euclidian distance
        """        
        sum_square = 0
        for i in range(0, len(first_point)):
            sum_square = (float(first_point[i]) - float(second_point[i]))**2
        return math.sqrt(sum_square)

    def _compute_unique_class_num(self):
        """get the list of uniques classes from the training dataset
        """        
        class_num = []
        for line in self._train_data:
            class_num.append(line[0])
        self._unique_classes = np.unique(class_num)

    def _update_confusion_matrix(self, line_class:int, scores_dict:dict, order:str = "min"):
        """Add one to the right coordinate of the confusion matrix. The coordinates are the line class number and the best score in dict

        Args:
            line_class (int): class identifier of the line
            scores_dict (dict): dictonary with class numbers as key and score as value
        """        
        row_num = line_class - 1
        if order == "min":
            temp = min(scores_dict.values())
        else:
            temp = max(scores_dict.values())
        res = [key for key in scores_dict if scores_dict[key] == temp]
        col_num = int(res[0]) - 1
        self._conf_matrix[row_num,
                          col_num] = self._conf_matrix[row_num, col_num] + 1

    def _add_train_point_to_plot(self):
        """Add trainig dataset points to the scatter plot
        """        
        pd_train = pd.DataFrame(self._train_data)
        scatter = plt.scatter(x=pd.to_numeric(pd_train[1]), y=pd.to_numeric(
            pd_train[2]), c=pd.to_numeric(pd_train[0]))
        plt.legend(*scatter.legend_elements(), loc="lower right")

    def _add_test_point_to_plot(self):
        """Add test dataset points to the scatter plot
        """        
        pd_test = pd.DataFrame(self._test_data)
        scatter = plt.scatter(x=pd.to_numeric(pd_test[1]), y=pd.to_numeric(
            pd_test[2]), c=pd.to_numeric(pd_test[0]))
        plt.legend(*scatter.legend_elements(), loc="lower right")

    def _get_splited_class(self, class_num:int):
        """Get the data from the given class number

        Args:
            class_num (int): Class name as int

        Returns:
            list: a list of lines from the given class
        """        
        data_class = []
        for line in self._train_data:
            if int(line[0]) == class_num:
                data_class.append(line)
        return data_class

    @staticmethod
    def _get_top_n_decision(n:int, theo_class:int, scores:dict, order:str = "max"):
        """Check if the theorical class is in the N best results of the dictonary

        Args:
            n (int): number of scores to check
            theo_class (int): theorical class to fit
            scores (dict): dictionnary of classes as keys and scores as values
            order (str): must min or max, indicates if the best score is the minimum or the maximum one

        Returns:
            bool: True if the theorical class is in the nth scores
        """    
        if order == "min":    
            scores_sorted = sorted(scores.items(), key=lambda kv: kv[1])
        else:
            scores_sorted = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        cut_scores = scores_sorted[0:n]
        top_n_result = False
        for score in cut_scores:
            if theo_class == int(score[0]):
                top_n_result = True
        return top_n_result
