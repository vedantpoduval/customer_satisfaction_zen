import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self,X_train,y_train):
        """
        Args:
            X_train : Training Data
            y_train : Training Labels
        Return:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    
    """
    def train(self, X_train, y_train):
        try:
            reg = LinearRegression()
            reg.fit(X_train,y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
    