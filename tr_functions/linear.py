import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from tr_functions.general import GeneralModel


class LinearSeparationModel(GeneralModel):
    def __init__(self):
        self.__converging = False
        self.__hyperplans = []
        self.__epochs = 5
        self.__model = {}
        GeneralModel.__init__(self)
        print("Linear Seaparation Model created")
        print("================================")

    # Getter and setter
    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, nb_epoch: int):
        self.__epochs = nb_epoch

    def print_model(self):
        for key, val in self.__model.items():
            print("The hyperplan between {} and {} is {}".format(
                key[0], key[1], val))
        print("======================")

    def linear_train(self, train_data, is_converging: bool = True, one_vs_all: bool = False):
        print("Begin linear training")
        self.__converging = is_converging
        self._train_data = train_data
        if not one_vs_all:
            self.__create_hyperplans_classes()
        data_pd = pd.DataFrame(train_data)
        results = {}
        for hyper_plan in self.__hyperplans:
            # Subset with data of the classes split by the hyperplans
            if not one_vs_all:
                hyperplan_data = data_pd[(data_pd[0] == hyper_plan[0]) | (
                    data_pd[0] == hyper_plan[1])].copy()
            # Transform the classes to 1 and -1
            hyperplan_data[3] = hyperplan_data[0].apply(
                lambda row: 1 if row == hyper_plan[0] else -1)

            del hyperplan_data[0]
            if is_converging:
                weights = self.__linear_perceptron_two_classes_converge(
                    hyperplan_data)
            else:
                weights = self.__linear_perceptron_two_classes_non_converge(
                    hyperplan_data)
            results[hyper_plan] = weights
            print("hyperplan {} done.".format(hyper_plan))
        print("Training done")
        print("================================")
        self.__model = results
        return results

    def test_linear_model(self, test_data, weighted_hp: dict = None):
        if weighted_hp is None:
            weighted_hp = self.__model
        self._test_data = test_data
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        for line in self._test_data:
            line = [float(val) for val in line]
            predict_classes = []
            for key, value in weighted_hp.items():
                res = self.__predict_one_line_perceptron(line[1:], value)
                if res >= 0:
                    predict_classes.append(key[0])
                else:
                    predict_classes.append(key[1])
            grouped_classes = Counter(predict_classes)
            if self._get_top_n_decision(1, int(line[0]), grouped_classes):
                self.count_top1 += 1
            if self._get_top_n_decision(2, int(line[0]), grouped_classes):
                self.count_top2 += 1
            self._update_confusion_matrix(
                int(line[0]), grouped_classes, order="max")

    def plot_test_data(self):
        self._add_test_point_to_plot()
        for key, value in self.__model.items():
            print(value)
            
            x = np.linspace(-10.,10.)
            y_h = (value[0]*x+value[2])/(-value[1])
            print(y_h)
            plt.plot(y_h, linewidth=2.5)
            
        plt.show()

    def __create_hyperplans_classes(self):
        self._compute_unique_class_num()
        for i in range(len(self._unique_classes)):
            for j in range(i+1, len(self._unique_classes)):
                self.__hyperplans.append(
                    (self._unique_classes[i], self._unique_classes[j]))

    def __linear_perceptron_two_classes_non_converge(self, train_data):
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

    @staticmethod
    def __predict_one_line_perceptron(line, weights):
        total = 0
        for i in range(len(line)):
            total += line[i] * weights[i]
        total += weights[-1]
        return total

    @staticmethod
    def __linear_perceptron_two_classes_converge(train_data):
        weights = np.zeros(len(train_data.columns))
        count = 1
        while count != 0:
            weights, count = LinearSeparationModel.__linear_perceptron_one_epoch(
                train_data, weights)
        return weights

    @staticmethod
    def __linear_perceptron_one_epoch(train_data, weights):
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
