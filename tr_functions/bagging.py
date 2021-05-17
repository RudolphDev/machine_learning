import random
import copy
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tr_functions.general import GeneralModel
from tr_functions.linear import LinearSeparationModel


class BaggingModel(GeneralModel):
    def __init__(self):
        self.__nb_models = 1
        self.__models_results = []
        self.__nb_results_cv = []
        self.__cross_val = 5
        GeneralModel.__init__(self)
        print("Bagging Model created")
        print("================================")

    # Getter and setter
    @property
    def nb_models(self):
        return self.__nb_models

    @nb_models.setter
    def nb_models(self, models: int):
        self.__nb_models = models

    # Public methods
    def test_bagging_linear_model(self, train_data:list, test_data:list, is_converging:bool = True):
        self._train_data = train_data
        self.__prepare_model_results(test_data)
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        linear = LinearSeparationModel()
        for i in range(self.__nb_models):
            # print("Boostrap of linear model {} on {}".format(i+1, self.__nb_models))
            self._test_data = copy.deepcopy(test_data)
            bootstrap_train = random.choices(self._train_data, k=len(self._train_data))
            model = linear.linear_train(bootstrap_train, is_converging=is_converging, one_vs_all=False) 
            linear.test_linear_model(self._test_data, model)
            self.__fill_model_results()
        # print("Bagging computing complete")
        
    def compute_bagging_results(self):
        i = 0
        for result in self.__models_results:
            result_dict = Counter(result[1])
            if self._get_top_n_decision(1, int(result[0]), result_dict):
                self._count_top1 += 1
            self._update_line(self._test_data[i], result_dict, order="max")
            self._update_confusion_matrix(int(result[0]), result_dict, order="max")
            i += 1
        
    def get_nb_bagging_cv(self, app_data:list, cv:int, max_n:int, is_converging: bool = True):
        self.__cross_val = cv
        self._train_data = app_data
        self._compute_unique_class_num()
        best_n = [0, 0]
        df = pd.DataFrame(app_data)
        shuffled = df.sample(frac=1)
        cut_dfs = np.array_split(shuffled, self.__cross_val)  
        for n in range(max_n):
            self.__nb_models = n + 1
            sum_error = 0
            for i in range(self.__cross_val):
                df_cv = cut_dfs[i]
                df_train = df.drop(df_cv.index)
                df_cv = df_cv.values.tolist()
                df_train = df_train.values.tolist()
                count_top_1 = 0
                self.test_bagging_linear_model(df_train, df_cv, is_converging=is_converging)
                for result in self.__models_results:
                    result_dict = Counter(result[1])
                    if self._get_top_n_decision(1, int(result[0]), result_dict, order="max"):
                        count_top_1 += 1
                error_rate = count_top_1/len(df_cv)
                sum_error = sum_error + error_rate
            self.__nb_results_cv.append((self.__nb_models, sum_error/self.__cross_val))
            if best_n[1] < sum_error/self.__cross_val:
                best_n[0] = self.nb_models
                best_n[1] = sum_error/self.__cross_val
        self.__plot_n_error_rate()
        print("The best n found is {} with a error rate = {}".format(best_n[0], 1-best_n[1]))
        self.__nb_models = best_n[0]

    # Private methods
    def __prepare_model_results(self, test_data):
        self.__models_results.clear()
        for i in  range(len(test_data)):
            self.__models_results.append((test_data[i][0], []))
            
    def __fill_model_results(self):
        for i in range(len(self._test_data)):
            self.__models_results[i][1].append(self._test_data[i][3])
            
    def __plot_n_error_rate(self):
        x = [val[0] for val in self.__nb_results_cv]
        y = [1-val[1] for val in self.__nb_results_cv]
        plt.plot(x,y)
        plt.show()