import logging
import pandas as pd
from zenml import step

@step
def train_model(data:pd.DataFrame) -> None:
    print("Inside train model")