from zenml import pipeline
from steps import ingest_data,cleaning_data ,train_model,evaluate_model



@pipeline(enable_cache = True)
def train_pipeline(data_path):
    df = ingest_data(data_path)
    X_train,X_test,y_train,y_test = cleaning_data(df)
    model = train_model(X_train,X_test,y_train,y_test)
    r2,rmse = evaluate_model(model,X_test,y_test)
     
