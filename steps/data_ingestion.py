import logging
import pandas as pd
from zenml import step

class DataIngestion:
    """
    Data Ingestion Class
    """
    def __init__(self,data_path):
        self.data_path = data_path
   
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path:str) -> pd.DataFrame:
    print("Inside Data Ingestion")
    """
    Function to implement the ingestion by taking the data path as input.

    Args:
        data_path (str): path to the input

    Returns:
        pd.DataFrame: data ingested 
    """
    try:
        ingest = DataIngestion(data_path)
        df = ingest.get_data()
        return df.head()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e

# path = "/Users/hrishikeshpoduval08/Desktop/ai_projects/customer_satisfaction_final/customer_satisfaction_zen/data/olist_customers_dataset.csv"
# print(ingest_data(path))