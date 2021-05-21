import random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tr_functions.general import GeneralModel
from tr_functions.linear import LinearSeparationModel
from tr_functions.kppv import KppvModel


class BaggingModel(GeneralModel):
    def __init__(self):
        """Create the Gaussian model and initialize attributes
        """
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
        """return the number of sub-models used in this model

        Returns:
            int: number of sub-model used in this model
        """
        return self.__nb_models

    @nb_models.setter
    def nb_models(self, models: int):
        """Set the number of sub-models used in this model

        Args:
            value (int): number of sub-models to use in this model
        """
        self.__nb_models = models

    # Public methods
    def test_bagging_linear_model(self, train_data:list, test_data:list, is_converging:bool = True):
        """compute the baggging of linear separation methods. Create the list of results obtained by the different bootstraps.

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist
            test_data (list): 2D nested list of testing data with class name as first element of the sublist
            is_converging (bool, optional): choose if the training dataset separation is converging or if you use 10 epochs. Defaults to True.
        """        
        self._train_data = train_data
        self.__prepare_model_results(test_data)
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        linear = LinearSeparationModel()
        for i in range(self.__nb_models):
            self.test_data = test_data
            bootstrap_train = random.choices(self._train_data, k=len(self._train_data))
            if is_converging == False:
                linear.epochs = 10
            model = linear.linear_train(bootstrap_train, is_converging=is_converging, one_vs_all=False) 
            linear.test_linear_model(self._test_data, model)
            self.__fill_model_results(linear._test_data)
        
    def test_bagging_kppv_model(self, train_data:list, test_data:list, k:int):
        """compute the baggging of kppv methods. Create the list of results obtained by the different bootstraps

        Args:
            train_data (list): 2D nested list of training data with class name as first element of the sublist
            test_data (list): 2D nested list of testing data with class name as first element of the sublist
            k (int): number of neighbours used in sub-models
        """        
        self._train_data = train_data
        self.__prepare_model_results(test_data)
        self._compute_unique_class_num()
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        kppv = KppvModel()
        for i in range(self.__nb_models):
            self.test_data = test_data
            bootstrap_train = random.choices(self._train_data, k=len(self._train_data))
            kppv.k = k
            kppv.compute_kppv(bootstrap_train, self._test_data)
            self.__fill_model_results(kppv._test_data)
        
    def compute_bagging_results(self):
        """Compute the classification rates and confusion matrixes from the results computed.
        """        
        i = 0
        for result in self.__models_results:
            result_dict = Counter(result[1])
            if self._get_top_n_decision(1, int(result[0]), result_dict):
                self._count_top1 += 1
            self._update_line(self._test_data[i], result_dict, order="max")
            self._update_confusion_matrix(int(result[0]), result_dict, order="max")
            i += 1
        
    def get_nb_bagging_cv(self, app_data:list, cv:int, max_n:int, is_converging: bool = True):
        """Cross-validation of the bagging method

        Args:
            app_data (list): 2D nested list of training data with class name as first element of the sublist
            cv (int): number of subpart of the training dataset used in cross validation
            max_n (int): max number of bootstraps to test in the cross-validation
            is_converging (bool, optional): choose if the training dataset separation is converging or if you must set a max number of epochs.. Defaults to True.
        """        
        self.__cross_val = cv
        self._train_data = app_data
        self._compute_unique_class_num()
        best_n = [0, 0]
        df = pd.DataFrame(app_data)
        shuffled = df.sample(frac=1)
        cut_dfs = np.array_split(shuffled, self.__cross_val) 
        for n in range(1,max_n):
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
    def __prepare_model_results(self, test_data:list):
        """prepare the list of results for each line in the testing dataset

        Args:
            test_data (list): 2D nested list of testing data with class name as first element of the sublist
        """        
        self.__models_results.clear()
        for i in  range(len(test_data)):
            self.__models_results.append((test_data[i][0], []))
            
    def __fill_model_results(self, test_data:list):
        """fill the result list from the last testing dataset, the class found column

        Args:
            test_data (list): 2D nested list of testing data with class name as first element of the sublist
        """        
        for i in range(len(test_data)):
            self.__models_results[i][1].append(test_data[i][3])
            
    def __plot_n_error_rate(self):
        """plot the error rate for each number of bootstrap tested by cross validation
        """        
        x = [val[0] for val in self.__nb_results_cv]
        y = [1-val[1] for val in self.__nb_results_cv]
        plt.plot(x,y)
        plt.show()