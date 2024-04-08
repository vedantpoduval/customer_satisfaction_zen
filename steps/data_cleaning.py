import logging
import pandas as pd
from src import DataCleaning,DataDivideStrategy,DataPreProcessStrategy
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

@step
def cleaning_data(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """
    Cleans the data and divides it into train and test

    """
    try:
        data_cleaning = DataCleaning(data,DataPreProcessStrategy())
        processed_data = data_cleaning.handle_data()
        data_cleaning = DataCleaning(data,DataDivideStrategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
        