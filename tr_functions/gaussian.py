# General import
import math
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

# local functions import
from .general import GeneralModel


class GaussianModel(GeneralModel):
    def __init__(self):
        """Create the Gaussian model and initialize attributes
        """
        self._compute_method = "euclidian"
        self._inv_covmat = None
        self._classes_centers = {}
        GeneralModel.__init__(self)
        print("Gaussian model created")

    # Getter and setter
    @property
    def compute_method(self):
        """Return the computed method set or used in the model

        Returns:
            str: Computed method used (could be "euclidian" or "mahalanobis")
        """
        return self._compute_method

    @compute_method.setter
    def compute_method(self, method: str):
        """set the computed method used in the model

        Args:
            method (str): computed method name (must be "euclidian" or "mahalanobis")
        """
        if (method == "euclidian") | (method == "mahalanobis"):
            self._compute_method = method
        else:
            print("The availables method are \"euclidian\" or \"mahalanobis\"")

    @property
    def classes_centers(self):
        """return the coordinates of each class centers

        Returns:
            dict: class center with class name as key and a list of coordinates as value
        """
        return self._classes_centers

    def print_classes_centers(self):
        """print the class center coordinates in a better format
        """
        for key, value in self._classes_centers.items():
            print("Class = {} has a center = {}".format(key, value))

    # Public methods
    def gaussian_fit_model(self, data: list):
        """Train the gaussian model with the given dataset. Get each class and create class center coordinates for each class.

        Args:
            data (list): 2D nested list of data

        Returns:
            dict: dictionary of class center coordinates
        """
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
        return self._classes_centers

    def test_model(self, dec_data: list):
        """Test the model created with the class centers created in the fit_model function

        Args:
            dec_data (list): 2D nested list of data
        """
        if self._compute_method == "mahalanobis":
            self.__compute_inv_covmat(dec_data)
        self.test_data = dec_data
        self._conf_matrix = np.zeros(
            (len(self._unique_classes), len(self._unique_classes)))
        for line in self._test_data:
            if self._compute_method == "mahalanobis":
                dists = self.__compute_mahalanobis_dists(line)
            else:
                dists = self.__compute_euclidian_dists(line)
            if self._get_top_n_decision(1, int(line[0]), dists, "min"):
                self._count_top1 += 1
            if self._get_top_n_decision(2, int(line[0]), dists, "min"):
                self._count_top2 += 1
            self._update_line(line, dists, "min")
            self._update_confusion_matrix(int(line[0]), dists)

    def plot_train_data(self):
        """plot the training dataset with the class centers represented as black crosses
        """
        self._add_train_point_to_plot()
        pd_centers = pd.DataFrame(self._classes_centers)
        scatter = plt.scatter(x=pd.to_numeric(pd_centers.T[0]), y=pd.to_numeric(
            pd_centers.T[1]), c="black", marker="+", s=100)
        plt.show()

    def plot_test_data(self):
        """plot the testing dataset with the class centers represented as black crosses
        """
        self._add_test_point_to_plot()
        pd_centers = pd.DataFrame(self._classes_centers)
        scatter = plt.scatter(x=pd.to_numeric(pd_centers.T[0]), y=pd.to_numeric(
            pd_centers.T[1]), c="black", marker="+", s=100)
        plt.show()

    # Private methods
    def __compute_euclidian_dists(self, line: list):
        """Compute the euclidian distances between the line and each class center then return the dictionnary of distances

        Args:
            line (list): coordinates of the point and the class label in the first position

        Returns:
            dict: dictionnary of the distance between the line and each class center with the class label as key
        """
        dists_dict = {}
        for class_center in self._classes_centers:
            dist = self._compute_one_euclidian_dist(
                self._classes_centers[class_center], line[1:3])
            dists_dict[class_center] = dist
        return dists_dict

    def __compute_mahalanobis_dists(self, line: list):
        """Compute the mahalanobis distances between the line and each class center then return the dictionnary of distances

        Args:
            line (list): coordinates of the point and the class label in the first position

        Returns:
            dict: dictionnary of the distance between the line and each class center with the class label as key
        """
        dists_dict = {}
        for class_center in self.classes_centers:
            dist = self.__compute_one_mahalanobis_dist(
                line[1:3], self.classes_centers[class_center])
            dists_dict[class_center] = dist
        return dists_dict

    def __compute_inv_covmat(self, dec_data: list):
        """compute the inverted covariance matrix from the coordinates of the given dataset. Used for the mahalanobis distance computing

        Args:
            dec_data (list):  2D nested list of data
        """
        data = pd.DataFrame(dec_data, dtype=float)[[1, 2]]
        cov = np.cov(data.values.T)
        inv_covmat = linalg.inv(cov)
        self._inv_covmat = inv_covmat

    def __compute_one_mahalanobis_dist(self, first_point: list, second_point: list):
        """Compute the mahalanobis distance between the two given points

        Args:
            first_point (list): list of coordinates of the first point
            second_point (list): list of coordinates of the second point

        Returns:
            float: mahalanobis distance between the two points
        """
        x_minus_mu = []
        for i in range(0, len(first_point)):
            x_minus_mu.append(float(first_point[i]) - second_point[i])
        x_minus_mu = np.array(x_minus_mu)
        left_term = np.dot(x_minus_mu, self._inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return math.sqrt(mahal)
