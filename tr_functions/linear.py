import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from tr_functions.general import GeneralModel


class LinearSeparationModel(GeneralModel):
    def __init__(self):
        self.__one_vs_all = False
        self.__hyperplans = []
        self.__epochs = 100
        self.__model = {}
        self.__error = []
        GeneralModel.__init__(self)
        # print("Linear Seaparation Model created")
        # print("================================")

    # Getter and setter
    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, nb_epoch: int):
        self.__epochs = nb_epoch

    # Public methods
    def print_model(self):
        if self.__one_vs_all == False:
            for key, val in self.__model.items():
                print("The hyperplan between {} and {} is {}".format(
                    key[0], key[1], val))
        else:
            for key, val in self.__model.items():
                print("The hyperplan between {} and others is {}".format(
                    key, val))
        print("======================")

    def linear_train(self, train_data, is_converging: bool = True, one_vs_all: bool = False):
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

    def test_linear_model(self, test_data, weighted_hp: dict = None):
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
                plt.plot(clean_y, clean_x, linewidth=2.5, color=self._COLORSET[int(key)])
        else:
            super().plot_test_data()

    # Private methods
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

    def __test_line_one_vs_one(self, line: list, weighted_hp: dict):
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
        temp_line = [float(val) for val in line]
        class_idx_dict = {}
        i = 0
        for class_name in self._unique_classes:
            class_idx_dict[class_name] = i
            i+=1
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
            self._conf_matrix[class_idx_dict[predict_classes[0]]][class_idx_dict[line[0]]] += 1
            line.append(int(predict_classes[0]))
        else:
            self._error_count += 1
            line.append(None)
            
        #print(predict_classes, line[0])
        
    @staticmethod
    def __predict_one_line_perceptron(line, weights):
        total = 0
        for i in range(len(line)):
            # print(i, len(line))
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
