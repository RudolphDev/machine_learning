# General import
import math
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

# local functions import
from tr_functions.general import GeneralModel


class GaussianModel(GeneralModel):
    def __init__(self):
        self._compute_method = "euclidean"
        self._inv_covmat = None
        self._classes_centers = {}
        GeneralModel.__init__(self)
        print("Gaussian model created")

    @property
    def compute_method(self):
        return self._compute_method

    @compute_method.setter
    def compute_method(self, method):
        if (method == "euclidian") | (method == "mahalanobis"):
            self._compute_method = method
        else:
            print("The availables method are \"euclidian\" or \"mahalanobis\"")

    @property
    def classes_centers(self):
        return self._classes_centers

    def print_classes_centers(self):
        for key, value in self._classes_centers.items():
            print("Class = {} has a center = {}".format(key, value))

    def gaussian_fit_model(self, data):
        self._train_data = data
        self._compute_unique_class_num()
        for class_num in self._unique_classes:
            data_class = self._get_splited_class(int(class_num))
            sum_x = 0
            sum_y = 0
            for line in data_class:
                sum_x += float(line[1])
                sum_y += float(line[2])
            x_center = 1/len(data_class) * sum_x
            y_center = 1/len(data_class) * sum_y
            self._classes_centers[class_num] = [x_center, y_center]
        print("Classes centers created")

    def _compute_euclidian_dists(self, line):
        dists_dict = {}
        for class_center in self._classes_centers:
            dist = GaussianModel.compute_one_euclidian_dist(
                self._classes_centers[class_center], line[1:3])
            dists_dict[class_center] = dist
        return dists_dict

    def _compute_mahalanobis_dists(self, line):
        dists_dict = {}
        for class_center in self.classes_centers:
            dist = self._compute_one_mahalanobis_dist(
                line[1:3], self.classes_centers[class_center])
            dists_dict[class_center] = dist
        return dists_dict

    def _compute_inv_covmat(self, dec_data):
        data = pd.DataFrame(dec_data, dtype=float)[[1, 2]]
        cov = np.cov(data.values.T)
        inv_covmat = linalg.inv(cov)
        self._inv_covmat = inv_covmat

    def test_model(self, dec_data):
        if self._compute_method == "mahalanobis":
            self._compute_inv_covmat(dec_data)
        self._len_test_data = len(dec_data)
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        for line in dec_data:
            if self._compute_method == "mahalanobis":
                dists = self._compute_mahalanobis_dists(line)
            else:
                dists = self._compute_euclidian_dists(line)
            if self._get_top_n_decision(1, line[0], dists, "min"):
                self._count_top1 += 1
            if self._get_top_n_decision(2, line[0], dists, "min"):
                self._count_top2 += 1
            self._update_confusion_matrix(int(line[0]), dists)
       
       
    def _compute_one_mahalanobis_dist(self, first_point: list, second_point: list):
        x_minus_mu = []
        for i in range(0, len(first_point)):
            x_minus_mu.append(float(first_point[i]) - second_point[i])
        x_minus_mu = np.array(x_minus_mu)
        left_term = np.dot(x_minus_mu, self._inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return math.sqrt(mahal)

    def show_train_plot(self):
        self._add_train_point_to_plot()
        pd_centers = pd.DataFrame(self._classes_centers)
        scatter = plt.scatter(x=pd.to_numeric(pd_centers.T[0]), y=pd.to_numeric(
            pd_centers.T[1]), c="black", marker="+", s=100)
        plt.show()
