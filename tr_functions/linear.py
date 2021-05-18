import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from tr_functions.general import GeneralModel


class LinearSeparationModel(GeneralModel):
    def __init__(self):
        """Create the linear spearation model and initialize attributes
        """
        self.__one_vs_all = False
        self.__hyperplans = []
        self.__epochs = 100
        self.__model = {}
        GeneralModel.__init__(self)
        # print("Linear Seaparation Model created")
        # print("================================")

    # Getter and setter
    @property
    def epochs(self):
        """get the number of epochs used in the model

        Returns:
            int: number of epoch used in the model
        """
        return self.__epochs

    @epochs.setter
    def epochs(self, nb_epoch: int):
        """set the number of epochs to use in the model

        Args:
            nb_epoch (int): number of epochs to use in the model
        """
        self.__epochs = nb_epoch

    # Public methods
    def print_model(self):
        """print the hyperplans model
        """
        if self.__one_vs_all == False:
            for key, val in self.__model.items():
                print("The hyperplan between {} and {} is {}".format(
                    key[0], key[1], val))
        else:
            for key, val in self.__model.items():
                print("The hyperplan between {} and others is {}".format(
                    key, val))
        print("======================")

    def linear_train(self, train_data: list, is_converging: bool = True, one_vs_all: bool = False):
        """Train the linear model and create the weights representing the hyperplans of each class. 

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist
            is_converging (bool, optional): choose if the training dataset separation is converging or if you must set a max number of epochs. Defaults to True.
            one_vs_all (bool, optional): choose if you want to test each class versus each other classes or each class vers all others. Defaults to False.

        Returns:
            [type]: [description]
        """
        # print("Begin linear training")
        self.__one_vs_all = one_vs_all
        self._train_data = train_data
        if not one_vs_all:
            self.__create_hyperplans_classes()
        else:
            self._compute_unique_class_num()
            self.__hyperplans = self._unique_classes
        data_pd = pd.DataFrame(train_data)
        results = {}
        for hyper_plan in self.__hyperplans:
            # Subset with data of the classes split by the hyperplans
            if not one_vs_all:
                hyperplan_data = data_pd[(data_pd[0] == hyper_plan[0]) | (
                    data_pd[0] == hyper_plan[1])].copy()
            else:
                hyperplan_data = data_pd.copy()
            # Transform the classes to 1 and -1
            hyperplan_data[3] = hyperplan_data[0].apply(
                lambda row: 1 if row == hyper_plan[0] else -1)
            del hyperplan_data[0]
            # Compute the linear regression
            if is_converging:
                weights = self.__linear_perceptron_two_classes_converge(
                    hyperplan_data)
                errors = 0
            else:
                weights, errors = self.__linear_perceptron_two_classes_non_converge(
                    hyperplan_data)
            results[hyper_plan] = weights
        # print("Training done")
        # print("================================")
        self.__model = results
        self.__error = errors
        return results

    def test_linear_model(self, test_data: list, weighted_hp: dict = None):
        """test the created linear model on the testing dataset

        Args:
            test_data (list): 2D nested list of testing data with class name as first element of the sublist
            weighted_hp (dict, optional): weights of each hyperplan if None will use the private weights. Defaults to None.
        """
        if weighted_hp is None:
            weighted_hp = self.__model
        self._test_data = test_data.copy()
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))

        for line in self._test_data:
            if not self.__one_vs_all:
                self.__test_line_one_vs_one(line, weighted_hp)
            else:
                self.__test_line_one_vs_all(line, weighted_hp)

    def plot_linear_data(self):
        """plot the linear data. Plot each hyperplans with each classes it separate. 
        """
        plt.rcParams['figure.figsize'] = [20, 8]
        np_test = np.array(self._test_data)
        plot_count = 1
        for key, value in self.__model.items():
            plt.subplot(2, 5, plot_count)
            full_idx = np.array([])
            for class_nb in key:
                idx = np.where(np_test[:, 0].astype(np.int) == int(class_nb))
                full_idx = np.append(full_idx, idx, )
                plt.scatter(x=np_test[idx, 1].astype(np.float), y=np_test[idx, 2].astype(
                    np.float), c=self._COLORSET[int(class_nb)], label=class_nb)
            plt.legend()
            full_idx = np.array(full_idx).astype(np.int)
            x = np.linspace(start=np.amin(np_test[full_idx, 2].astype(
                np.float)), stop=np.amax(np_test[full_idx, 2].astype(np.float)), num=10)
            plot_count += 1
            y_h = (value[1]*x+value[2])/(-value[0])
            plt.plot(y_h, x, linewidth=2.5)
        plt.show()

    def plot_test_data(self):
        """plot the testing dataset with the hyperplans if the method is one vs all.
        """
        if self.__one_vs_all:
            self._add_test_point_to_plot()
            np_test = np.array(self.test_data)
            for key, value in self.__model.items():
                x = np.linspace(start=np.amin(np_test[:, 2].astype(
                    np.float)), stop=np.amax(np_test[:, 2].astype(np.float)), num=500)
                y_h = (value[1]*x+value[2])/(-value[0])
                max_y = np.amax(np_test[:, 1].astype(np.float))
                min_y = np.amin(np_test[:, 1].astype(np.float))
                clean_x = []
                clean_y = []
                for i in range(len(y_h)):
                    if y_h[i] >= min_y and y_h[i] <= max_y:
                        clean_y.append(y_h[i])
                        clean_x.append(x[i])
                plt.plot(clean_y, clean_x, linewidth=2.5,
                         color=self._COLORSET[int(key)])
        else:
            super().plot_test_data()

    # Private methods
    def __create_hyperplans_classes(self):
        """create the list of hyperplans to compute. 
        """
        self._compute_unique_class_num()
        for i in range(len(self._unique_classes)):
            for j in range(i+1, len(self._unique_classes)):
                self.__hyperplans.append(
                    (self._unique_classes[i], self._unique_classes[j]))

    def __linear_perceptron_two_classes_non_converge(self, train_data: list):
        """Use perceptron method to get the different weights of each class. Non converging method. The number of epochs is based on the attribute.

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist

        Returns:
            dict: class name as key and list of weights as value
        """
        weights = np.zeros(len(train_data.columns))
        final_weights = []
        best_count = None
        for n in range(self.__epochs):
            weights, count = LinearSeparationModel.__linear_perceptron_one_epoch(
                train_data, weights)
            if best_count == None:
                best_count = count
            if best_count >= count:
                final_weights = weights
                best_count = count
        return final_weights, best_count

    def __test_line_one_vs_one(self, line: list, weighted_hp: dict):
        """test the line for each class weights. The weights contain the linear separation between each class. 
        Write the results in the different attributes.

        Args:
            line (list): list containing the class name then the coordinates associate to the point
            weighted_hp (dict): class name as key and list of weights as value
        """
        temp_line = [float(val) for val in line]
        predict_classes = []
        for key, value in weighted_hp.items():
            res = self.__predict_one_line_perceptron(temp_line[1:], value)
            if res >= 0:
                predict_classes.append(key[0])
            else:
                predict_classes.append(key[1])
        grouped_classes = Counter(predict_classes)
        if self._get_top_n_decision(1, int(line[0]), grouped_classes):
            self.count_top1 += 1
        if self._get_top_n_decision(2, int(line[0]), grouped_classes):
            self.count_top2 += 1
        self._update_line(line, grouped_classes, "max")
        self._update_confusion_matrix(
            int(line[0]), grouped_classes, order="max")

    def __test_line_one_vs_all(self, line: list, weighted_hp: dict):
        """test the line for each class weights. The weights contain the linear separation of each class vs all others. 
        Write the results in the different attributes.

        Args:
            line (list): list containing the class name then the coordinates associate to the point
            weighted_hp (dict): class name as key and list of weights as value
        """
        temp_line = [float(val) for val in line]
        class_idx_dict = {}
        i = 0
        for class_name in self._unique_classes:
            class_idx_dict[class_name] = i
            i += 1
        predict_classes = []
        for key, value in weighted_hp.items():
            res = self.__predict_one_line_perceptron(temp_line[1:], value)
            if res >= 0:
                predict_classes.append(key)
        if len(predict_classes) > 1:
            self._error_count += 1
            line.append(None)
        elif len(predict_classes) == 1:
            if predict_classes[0] == line[0]:
                self.count_top1 += 1
            self._conf_matrix[class_idx_dict[predict_classes[0]]
                              ][class_idx_dict[line[0]]] += 1
            line.append(int(predict_classes[0]))
        else:
            self._error_count += 1
            line.append(None)

    @staticmethod
    def __predict_one_line_perceptron(line: list, weights: list):
        """get the result for the perceptron for the line based on the given weights

        Args:
            line (list): list of coordinates of the point
            weights (list): weights of the linear perceptron to test

        Returns:
            float: result of the prediction
        """
        total = 0
        for i in range(len(line)):
            total += line[i] * weights[i]
        total += weights[-1]
        return total

    @staticmethod
    def __linear_perceptron_two_classes_converge(train_data: list):
        """Use perceptron method to get the different weights of each class. Converging method so the model is fixed when no bad classed point left. The number of epochs is based on the attribute.

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist

        Returns:
            dict: class name as key and list of weights as value
        """
        weights = np.zeros(len(train_data.columns))
        count = 1
        while count != 0:
            weights, count = LinearSeparationModel.__linear_perceptron_one_epoch(
                train_data, weights)
        return weights

    @staticmethod
    def __linear_perceptron_one_epoch(train_data: list, weights: dict):
        """Get the new weights after one epoch done and the count of errors found. 

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist
            weights (dict): class name as key and list of weights as value

        Returns:
            dict: class name as key and list of weights as value
        """
        count = 0
        for index, row in train_data.iterrows():
            temp_line = row.tolist()
            line = list(map(float, temp_line))
            # Step 1
            if line[-1] == -1:
                point_list = [-x for x in line[:-1]]
                point_list.append(line[-1])
            else:
                point_list = line
            # Step 2
            if np.dot(weights, point_list) <= 0:
                weights = [weights[i] + point_list[i]
                           for i in range(len(weights))]
                count += 1
        return weights, count
