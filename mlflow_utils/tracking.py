import os
import mlflow

# Set local tracking URI in the data folder
os.makedirs("data", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")

class MLflowTracker:
    def __init__(self, experiment_name="Book_Recommendation_System"):
        self.experiment_name = experiment_name
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"MLflow: Could not set experiment: {e}")
            
    def log_collaborative_run(self, params, metrics):
        print("MLflow: Logging Collaborative Filtering run...")
        try:
            with mlflow.start_run(run_name="Collaborative_Filtering_Surprise"):
                # Log hyperparameters
                mlflow.log_params(params)
                # Log metrics
                mlflow.log_metrics(metrics)
                print("MLflow: Collaborative Filtering logged successfully.")
        except Exception as e:
            print(f"MLflow Logging Error: {e}")
            
    def log_two_tower_run(self, params, epoch_losses):
        print("MLflow: Logging Two-Tower Deep Learning run...")
        try:
            with mlflow.start_run(run_name="Two_Tower_PyTorch"):
                # Log hyperparameters
                mlflow.log_params(params)
                # Log loss curve
                for epoch, losses in enumerate(epoch_losses, 1):
                    mlflow.log_metric("train_loss", losses["train_loss"], step=epoch)
                    mlflow.log_metric("val_loss", losses["val_loss"], step=epoch)
                print("MLflow: Two-Tower model logged successfully.")
        except Exception as e:
            print(f"MLflow Logging Error: {e}")
            
    def log_xgboost_run(self, params, metrics, feature_importances=None):
        print("MLflow: Logging XGBoost Candidate Ranker run...")
        try:
            with mlflow.start_run(run_name="XGBoost_Ranker"):
                # Log hyperparameters
                mlflow.log_params(params)
                # Log evaluation metrics
                mlflow.log_metrics(metrics)
                # Log feature importances as parameters
                if feature_importances:
                    for feat, imp in feature_importances.items():
                        mlflow.log_param(f"imp_{feat}", round(float(imp), 4))
                print("MLflow: XGBoost Ranker logged successfully.")
        except Exception as e:
            print(f"MLflow Logging Error: {e}")
            
if __name__ == "__main__":
    # Test tracking initialization
    tracker = MLflowTracker()
    print("MLflow tracker initialized.")
