# local functions import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tr_functions.general import GeneralModel

class KppvModel(GeneralModel):
    
    def __init__(self):
        self.__k = 1
        self.__cross_val = 1
        self.__k_results_cv = []
        self.__vote_method = "unanimous"
        GeneralModel.__init__(self)
    
    # Getter and setter
    @property
    def k(self):
        return self.__k
    
    @k.setter
    def k(self, value:int):
        print("K set to {}".format(value))
        self.__k = value
    
    @property
    def cross_val(self):
        return self.__cross_val
    
    @cross_val.setter
    def cross_val(self, value: int):
        if value > 0:
            self.__cross_val = value
        else:
            print("The cross validation value must be > 0!")
    
    @property
    def vote_method(self):
        return self.__vote_method
    
    @vote_method.setter
    def vote_method(self, value: str):
        if value in ("unanimous", "majority"):
            self.__vote_method = value
        else:
            print("Accepted methods are \"unanimous\" and \"majority\"")        

    def print_k_results_cv(self):
        print("Cross Validation results:")
        for k in self.__k_results_cv:
            print("For k = {} the error rate is {}".format(k[0], round(1-k[1], 3)))
    
    #Public methods
    def compute_kppv(self, app_data:list, dec_data:list):
        print("Will use k = {} neighbours".format(self.__k))
        self._train_data = app_data
        self._test_data = dec_data
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros((len(self._unique_classes), len(self._unique_classes)))
        
        for line in dec_data:
            kppv_list = self.__get_kppv_list(line[1:])
            k_list = list(Counter(kppv_list[0]).items())
            self.__compute_vote_kppv(k_list, int(line[0]))
        #     # Conf matrix
            row_num = int(line[0]) - 1
            col_num = int(k_list[0][0]) -1
            self._conf_matrix[row_num, col_num] = self._conf_matrix[row_num, col_num] + 1
    
    def get_k_cross_validation(self,app_data, k_max, cv):
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
            self.__k_results_cv.append((self.__k,sum_error/self.__cross_val))
            if best_k[1] < sum_error/self.__cross_val:
                best_k[0] = self.__k
                best_k[1] = sum_error/self.__cross_val
        self.__plot_k_error_rate()
        print("The best k found is {} with a error rate = {}".format(best_k[0], 1-best_k[1]))
        self.k = best_k[0]

    #Private methods 
    def __get_kppv_list(self, dec_point:list, train_data=None):
        """Compute all distances between dec_point and each training point then keep only the k best distances.

        Args:
            dec_point (list): coordinates of the point

        Returns:
            list: list of lists with the class number and the score in each
        """     
        if train_data == None:
            train_data = self._train_data   
        dist_list = []
        for app_line in train_data:
            dist_list.append([app_line[0], self._compute_one_euclidian_dist(app_line[1:3], dec_point)])
        app_df = pd.DataFrame(dist_list)
        sorted_df = app_df.sort_values(by=1)
        kppv_list = sorted_df.head(self.__k)
        return kppv_list

    def __compute_vote_kppv(self, k_list:list, theo_class:int):
        k_list = [(int(k[0]),k[1]) for k in k_list]
        if self.__vote_method == "unanimous":
            self.__compute_unanimous_vote(k_list, theo_class)    
        else:
            self.__compute_majority_vote(k_list, theo_class)
        
    def __compute_unanimous_vote(self, k_list:list, theo_class:int):
        if len(k_list) != 1:
            self._error_count += 1
        else:
            if k_list[0][0] == theo_class:
                self._count_top1 += 1
            
    def __compute_majority_vote(self, k_list:list, theo_class:int):
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
        x = [val[0] for val in self.__k_results_cv]
        y = [1-val[1] for val in self.__k_results_cv]
        plt.plot(x,y)
        plt.show()
        print(self.__k_results_cv)


   