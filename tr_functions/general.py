
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import Counter
import matplotlib.pyplot as plt
import math
import copy


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
        self._COLORSET = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c', 4:'#d62728', 5:'#9467bd'}

    # Getter and setter
    @property
    def train_data(self):
        """Return the data used to train the model.

        Returns:
            Pandas DataFrame: dataframe with each value and label of the training dataset
        """
        return pd.DataFrame(self._train_data)

    @train_data.setter
    def train_data(self, train_data:list):
        """Set the training data used to train the model.

        Args:
            train_data (list): nested list containing each line of the training dataset
        """        
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
        """set the number of points right labeled

        Args:
            count (int): numbers of right labeled points
        """        
        self._count_top1 = count

    @property
    def count_top2(self):
        """return the number of points right labeled in the two best classes

        Returns:
            int: number of right labeled points in the two best classes
        """        
        return self._count_top2

    @count_top2.setter    
    def count_top2(self, count: int):
        """set the number of points right labeled in the two best classes

        Args:
            count (int): number of right labeled points in the two best classes
        """        
        self._count_top2 = count

    @property
    def conf_matrix(self):
        """return the confusion matrix created from the test dataset

        Returns:
            list: 2D nested lists of confusion matrix
        """        
        return self._conf_matrix

    @property
    def test_data(self):
        """return the dataset used to test the model

        Returns:
            DataFrame: Dataframe containing the dataset used to test the model
        """        
        return pd.DataFrame(self._test_data)

    @test_data.setter
    def test_data(self, test_data:list):
        """Set the test dataset with a deepcopy keep the original dataset unmodified

        Args:
            test_data (list): 2D nested lists of testing dataset
        """        
        self._test_data = copy.deepcopy(test_data)
    
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

    @staticmethod
    def remove_one_class_from_data(table: list, class_name: str):
        """remove on the classe from the given dataset

        Args:
            table (list): 2D nested lists of the dataset
            class_name (str): class name deleted

        Returns:
            list: truncated dataset (line with the given class name removed)
        """        
        clean_table = []
        for line in table:
            if line[0] != class_name:
                clean_table.append(line)
        return clean_table

    def print_model_result(self):
        """Print the results of the model prediction for the test data used.
        """
        print("Results :")
        print("----------------")
        print("\nTop results :")
        print("----------------")
        print("Top 1 rate : ", self._count_top1/len(self._test_data))
        if self._count_top2 != 0:
            print("Top 2 rate : ", self._count_top2/len(self._test_data))
        if self._error_count != 0:
            print("Error rate : ", self._error_count/len(self._test_data))
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
        """Plot both training and test data
        """        
        np_train = np.array(self._train_data)
        np_test = np.array(self.test_data)
        fig, ax = plt.subplots()
        for class_nb in np.unique(np_train[:,0].astype(np.int)):
            idx_train = np.where(np_train[:,0].astype(np.int) == class_nb)
            idx_test = np.where(np_test[:,0].astype(np.int) == class_nb)
            ax.scatter(x=np_train[idx_train,1].astype(np.float), y=np_train[idx_train,2].astype(np.float), c=self._COLORSET[class_nb], alpha=0.5, marker="+")
            ax.scatter(x=np_test[idx_test,1].astype(np.float), y=np_test[idx_test,2].astype(np.float), c=self._COLORSET[class_nb], label=class_nb, marker=".")
        ax.legend()
        plt.show()
        
    def plot_test_data(self):
        """Plot testing dataset
        """        
        plt.rcParams['figure.figsize'] = [8, 8]
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
            sum_square += (float(first_point[i]) - float(second_point[i]))**2
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
        np_train = np.array(self._train_data)
        fig, ax = plt.subplots()
        for class_nb in np.unique(np_train[:,0].astype(np.int)):
            idx = np.where(np_train[:,0].astype(np.int) == class_nb)
            ax.scatter(x=np_train[idx,1].astype(np.float), y=np_train[idx,2].astype(np.float), c=self._COLORSET[class_nb], label=class_nb)
        ax.legend()

    def _add_test_point_to_plot(self):
        """Add test dataset points to the scatter plot
        """        
        np_test = np.array(self._test_data)
        fig, ax = plt.subplots()
        for class_nb in np.unique(np_test[:,0].astype(np.int)):
            idx = np.where(np_test[:,0].astype(np.int) == class_nb)
            e_c = []
            for i in np_test[idx[0],3]:
                if i != None:
                    e_c.append(self._COLORSET[int(i)]) 
                else:
                    e_c.append("#000000")
            ax.scatter(x=np_test[idx,1].astype(np.float), y=np_test[idx,2].astype(np.float), edgecolors=self._COLORSET[class_nb], c=e_c, label=class_nb)
        ax.legend()

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

    def _update_line(self, line:list, scores_dict:dict, order:str="max"):
        """update the given line by adding the predicted class

        Args:
            line (list): line wich will be updated
            scores_dict (dict): dictionary with the score of each class
            order (str, optional): Indicates if the best score is the max or the min one. Defaults to "max".
        """        
        if order == "min":
            temp = min(scores_dict.values())
        else:
            temp = max(scores_dict.values())
        res = [key for key in scores_dict if scores_dict[key] == temp]
        line.append(res[0])
        