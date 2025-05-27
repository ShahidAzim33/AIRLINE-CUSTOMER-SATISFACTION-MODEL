import os
import pandas as pd
import sys
import json
import mlflow
import joblib
import mlflow.sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import lightgbm as lgb
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger=get_logger(__name__)

class Model_Training:
    def __init__(self,data_path,params_path,model_save_path,experiment_name='Model_Training_Experiment'):
        self.data_path=data_path
        self.params_path=params_path
        self.model_save_path=model_save_path
        self.experiment_name=experiment_name

        self.best_model=None
        self.metrics=None

    def load_data(self):
        try:
            logger.info("data loadind for model training")
            data=pd.read_csv(self.data_path)
            logger.info("data loaded successful")
            return data
        except Exception as e:
            raise CustomException("error while loading data", sys)
        
    def split_data(self,data):
        try:
            logger.info("data splitting started")
            x=data.drop(columns='satisfaction')
            y=data['satisfaction']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            logger.info("data splitting done")
            
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise CustomException("error while splitting data", sys)
        
    def train_model(self,x_train,y_train,params):
        try:
            logger.info("training model is started")
            lgbm=lgb.LGBMClassifier()
            grid_search=GridSearchCV(lgbm,param_grid=params,cv=3,scoring='accuracy')
            grid_search.fit(x_train,y_train)
            logger.info("model training is completed")
            self.best_model=grid_search.best_estimator_
            return{
                 "model": self.best_model,
                 "params":grid_search.best_params_
            }
        except Exception as e:
            raise CustomException("error while training the model", sys)
        
    def evaluate_model(self,x_test,y_test):
        try:
            logger.info("model evaluation started")
            y_pred=self.best_model.predict(x_test)
            self.metrics={
                "accuracy":accuracy_score(y_test,y_pred),
                "precision":precision_score(y_test,y_pred,average='weighted'),
                "recall":recall_score(y_test,y_pred,average='weighted'),
                "F1 score":f1_score(y_test,y_pred,average='weighted'),
                "confusion matrix":confusion_matrix(y_test,y_pred).tolist()
               }    
            logger.info(f"Evaluation matrix :{self.metrics}")
            return self.metrics
        except Exception as e:
            raise CustomException("error while evaluating model", sys)
    def save_model(self):
        try:
            logger.info("saving the moedl")
            os.makedirs(os.path.dirname(self.model_save_path),exist_ok=True)
            joblib.dump(self.best_model,self.model_save_path)
            logger.info("model saved successfully")
        except Exception as e:
            raise CustomException("error while saving the data", sys)     
    def run(self):
        try:
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run():
                data=self.load_data()
                x_train,x_test,y_train,y_test=self.split_data(data)
                with open(self.params_path,"r") as f:
                    params=json.load(f)
                logger.info(f"loaded hyperparameter : {params}")
                mlflow.log_params({f"grid_{key}" : value for key, value in params.items()})
                model_data=self.train_model(x_train,y_train,params)
                self.best_model=model_data["model"]
                best_params=model_data["params"]
                logger.info(f"best hyper parameter are : {best_params}")
                mlflow.log_params({f"best_{key}" : value for key, value in best_params.items()})
                metrics=self.evaluate_model(x_test,y_test) 
                for metric,value in metrics.items():
                    if isinstance(value,(int,float)):
                        logger.info(f"logging metric: {metric}={value}")
                        mlflow.log_metric(metric,value)
                    else:
                        logger.warning(f"skipping non-numeric metric : {metric}") 
                self.save_model()
                mlflow.sklearn.log_model(self.best_model,"model")
        except CustomException as ce:
            logger.error(str(ce))
            mlflow.end_run(status="FAILED")
if __name__=="__main__":
    model_trainer=Model_Training(data_path=Engineer_data_path,params_path=PARAMS_PATH,model_save_path=MODEL_SAVE_PATH)
    model_trainer.run()                       
        

                                                           

