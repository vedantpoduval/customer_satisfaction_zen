import logging
import pandas as pd
from zenml import step

@step
def cleaning_data(data:pd.DataFrame) -> None:
    print("Inside Cleaning data")