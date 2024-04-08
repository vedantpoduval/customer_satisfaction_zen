from zenml import pipeline
from steps import ingest_data,cleaning_data ,train_model,evaluate_model



@pipeline(enable_cache = True)
def train_pipeline(data_path):
    df = ingest_data(data_path)
    cleaning_data(df)
    train_model(df)
    evaluate_model(df)
    
