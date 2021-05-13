from tr_functions.general import GeneralModel
from tr_functions.linear import LinearSeparationModel


class BaggingModel(GeneralModel):
    def __init__(self):
        self.__nb_models = 1
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
    def train_bagging_model(self):
        return self.__nb_models

    # Private methods
