import random
import copy
from collections import Counter
import numpy as np

from tr_functions.general import GeneralModel
from tr_functions.linear import LinearSeparationModel


class BaggingModel(GeneralModel):
    def __init__(self):
        self.__nb_models = 1
        self.__models_results = []
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
            print("Boostrap of linear model {} on {}".format(i+1, self.__nb_models))
            self._test_data = copy.deepcopy(test_data)
            bootstrap_train = random.choices(self._train_data, k=len(self._train_data))
            model = linear.linear_train(bootstrap_train, is_converging=is_converging, one_vs_all=False) 
            linear.test_linear_model(self._test_data, model)
            self.__fill_model_results()
        print("Bagging computing complete")
        
    def compute_bagging_results(self):
        i = 0
        for result in self.__models_results:
            result_dict = Counter(result[1])
            if self._get_top_n_decision(1, int(result[0]), result_dict):
                self._count_top1 += 1
            self._update_line(self._test_data[i], result_dict, order="max")
            self._update_confusion_matrix(int(result[0]), result_dict, order="max")
            i += 1
        

    # Private methods
    def __prepare_model_results(self, test_data):
        for i in  range(len(test_data)):
            self.__models_results.append((test_data[i][0], []))
            
    def __fill_model_results(self):
        for i in range(len(self._test_data)):
            self.__models_results[i][1].append(self._test_data[i][3])