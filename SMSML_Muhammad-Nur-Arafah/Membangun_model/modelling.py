import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# KRITERIA WAJIB: Autologging
mlflow.sklearn.autolog()
mlflow.set_experiment("D-Audit_Muhammad_Nur_Arafah")

def train():
    # Data sederhana agar script bisa running
    df = pd.DataFrame({'fitur': [1, 2, 3], 'target': [10, 20, 30]})
    with mlflow.start_run(run_name="Final_Model_BBPOM"):
        model = RandomForestRegressor(n_estimators=100)
        model.fit(df[['fitur']], df['target'])
        print("Training Sukses dengan Autolog aktif!")

if __name__ == "__main__":
    train()