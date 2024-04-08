import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame):
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    
    """
    def handle_data(self, data: pd.DataFrame):
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix", 
                    "order_item_id"
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"].fillna("No review")

            data = data.select_dtypes(include=[np.number])

            return data
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    
    """
    def handle_data(self, data: pd.DataFrame):
        """
        Divide data into train and test split
        """
        try:
            X = data.drop(["review_score"],axis  = 1)
            y = data["review_score"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data and doing train test splits
    
    """
    def __init__(self, data: pd.DataFrame , strategy: DataStrategy) -> None:
        self.df = data
        self.strategy = strategy
   
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Handle Data
        
        """
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e



    
    