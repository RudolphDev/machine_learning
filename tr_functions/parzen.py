import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tr_functions.general import GeneralModel

class ParzenModel(GeneralModel):
    def __init__(self):
        self.__h = 0
        self.__cross_val = 0
        self.__method = "uniform"
        self.__h_results_cv = []
        GeneralModel.__init__(self)
        
    # Getter and setter
    @property
    def h(self):
        return self.__h
    
    @h.setter
    def h(self, h):
        self.__h = h
        
    @property
    def cross_val(self):
        return self.__cross_val
    
    @cross_val.setter
    def cross_val(self, value):
        self.__cross_val = value
        
    @property
    def method(self):
        return self.__method
    
    @method.setter
    def method(self, value):
        if value in ("gaussian", "uniform"):
            self.__method = value
        else:
            print("Error! The value must be \"gaussian\" or \"uniform\"") 
               
    # Public methods
    def compute_parzen(self, app_data, dec_data):
        self._train_data = app_data
        self._test_data = dec_data
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros((len(self._unique_classes), len(self._unique_classes)))
        
        for line in self._test_data:
            if self.__method == "gaussian":
                results = self.__get_dict_gaussian_value(line[1:])
            else:
                results = self.__get_dict_uniform_value(line[1:])
            if self._get_top_n_decision(1, line[0], results, "max"):
                self._count_top1 +=1
            if self._get_top_n_decision(2, line[0], results, "max"):
                self._count_top2 += 1
            self._update_confusion_matrix(int(line[0]), results, "max")
    
    def get_h_cross_validation(self, app_data, h_list, cv,):
        self.__cross_val = cv
        self._train_data = app_data
        self._compute_unique_class_num()
        best_h = [0, 0]
        df = pd.DataFrame(app_data)
        shuffled = df.sample(frac=1)
        cut_dfs = np.array_split(shuffled, self.__cross_val)  
        for h in h_list:
            self.__h = h
            sum_error = 0
            for i in range(self.__cross_val):
                df_cv = cut_dfs[i]
                df_train = df.drop(df_cv.index)
                df_cv = df_cv.values.tolist()
                df_train = df_train.values.tolist()
                count_top_1 = 0
                for line in df_cv:
                    if self.__method == "uniform":
                        results = self.__get_dict_uniform_value(line[1:], df_train)
                    elif self.__method == "gaussian":
                        results = self.__get_dict_gaussian_value(line[1:], df_train)
                    if self._get_top_n_decision(1, line[0], results, "max"):
                        count_top_1 = count_top_1 + 1
                error_rate = count_top_1/len(df_cv)
                sum_error = sum_error + error_rate
            self.__h_results_cv.append((self.__h, sum_error/self.__cross_val))
            if best_h[1] < sum_error/self.__cross_val:
                best_h[0] = h
                best_h[1] = sum_error/self.__cross_val
        self.__plot_h_error_rate()
        print("The best h found is {} with a error rate = {}".format(best_h[0], 1-best_h[1]))
        self.__h = best_h[0]

    #Private methods
    def __get_dict_uniform_value(self, point, train_data=None):
        if train_data is None:
            train_data = self._train_data
        result_dict = {}
        COUNT_CLASS = 100
        for one_class in self._unique_classes:
            count_uniform = 0
            for app_point in train_data:
                if app_point[0] == one_class:
                    count_uniform = count_uniform + self.__get_uniform_val(app_point[1:], point)
            result_dict[one_class] = count_uniform/COUNT_CLASS
        return result_dict
    
    def __get_uniform_val(self, train_point, test_point):
        result = 0
        dist = self._compute_one_euclidian_dist(train_point, test_point)
        if (dist > -self.__h) & (dist < self.__h):
            result = 1
        return result

    def __get_dict_gaussian_value(self, point, train_data=None):
        if train_data is None:
            train_data = self.__train_data
        result_dict = {}
        COUNT_CLASS = 100
        for one_class in self._unique_classes:
            count_uniform = 0
            for app_point in train_data:
                if app_point[0] == one_class:
                    count_uniform = count_uniform + self.__get_gaussian_val(app_point[1:], point)
            result_dict[one_class] = count_uniform/COUNT_CLASS
        return result_dict

    
    def __get_gaussian_val(self, app_point, dec_point):
        result = 0
        dist = self._compute_one_euclidian_dist(app_point, dec_point)
        result = (1 / (self.__h * math.sqrt(2 * math.pi)))* math.exp((-(dist)**2)/(2*(self.__h**2)))
        return result
    
    def __plot_h_error_rate(self):
        x = [val[0] for val in self.__h_results_cv]
        y = [1-val[1] for val in self.__h_results_cv]
        plt.plot(x,y)
        plt.show()
        print(self.__h_results_cv)