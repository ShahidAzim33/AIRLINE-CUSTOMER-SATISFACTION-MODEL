from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb

import pandas as pd
from config.path_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import matplotlib.pyplot as plt
import time
import sys

from torch.utils.tensorboard import SummaryWriter

logger=get_logger(__name__)

class ModelSelection:
    def __init__(self,data_path):
        self.data_path=data_path
        run_id=time.strftime("%Y%m%d-%H%M%S")
        self.writer=SummaryWriter(log_dir=f"tensorboard_logs/run{run_id}")
        self.models={
                      'Logistic Regression': LogisticRegression(),
                      'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                      'Gradient Boosting': GradientBoostingClassifier(n_estimators=50),
                      'AdaBoost': AdaBoostClassifier(n_estimators=50),
                      'support Vector classifier': SVC(),
                      'K-Nearest Neighbors': KNeighborsClassifier(),
                      'Naive Bayes': GaussianNB(),
                      'Decision tress': DecisionTreeClassifier(),
                      'LightGBM': lgb.LGBMClassifier(),
                      'XGBoost': xgb.XGBClassifier(eval_metrics='mlogloss')
        }
        self.results={}
    def load_data(self):
        try:
            logger.info("loading CSV File")
            df=pd.read_csv(self.data_path)
            df_sample=df.sample(frac=0.05,random_state=42)
            x=df_sample.drop(columns='satisfaction') 
            y=df_sample['satisfaction']
            logger.info("data loaded sample successfully")
            return x,y
        except Exception as e:
            raise CustomException("error while laoding data",sys)

    def split_data(self,x,y):
        try:
            logger.info("splitting data")
            return train_test_split(x,y,test_size=0.2,random_state=42)
        except Exception as e:
            raise CustomException("error while laoding data",sys)
        
    def log_confusion_matrix(self,y_true,y_pred,step,model_name):
        cm=confusion_matrix(y_true,y_pred)
        fig, ax=plt.subplots(figsize=(5,5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
              ax.text(x=j,y=i,s=cm[i,j],va="center", ha="center")

        plt.xlabel("predicted label")
        plt.ylabel("True/Actual label")
        plt.title(f"confusuion matrix for {model_name}")
        self.writer.add_figure(f"confusion matrix/{model_name}",fig,global_step=step)
        plt.close(fig)    


    def train_evaluate(self,x_train,x_test,y_train,y_test):
        try:
            logger.info("training and evaluation started")
            for idx, (name,model) in enumerate(self.models.items()):
                model.fit(x_train,y_train)
                y_pred=model.predict(x_test)    
                accuracy=accuracy_score(y_test,y_pred)
                precision=precision_score(y_test,y_pred,average='weighted',zero_division=0)
                recall=recall_score(y_test,y_pred,average='weighted',zero_division=0)
                f1=f1_score(y_test,y_pred,average='weighted',zero_division=0)

                self.results[name]={
                    'accuracy':accuracy,
                    'precision':precision,
                    'recall':recall,
                    'f1_score':f1
                }
                logger.info(f"{name} trained successfully"
                            f"Metrics : Accuracy :{accuracy}, precision :{precision},recall :{recall},f1 score:{f1}")
                self.writer.add_scalar(f"Accuracy/{name}" , accuracy, idx )
                self.writer.add_scalar(f"Precision/{name}" , precision, idx )
                self.writer.add_scalar(f"Recall/{name}" , recall, idx )
                self.writer.add_scalar(f"F1_score/{name}" , f1, idx )
                
                self.writer.add_text('Model Details', f"Metrics : Accuracy :{accuracy}, precision :{precision},recall :{recall},f1 score:{f1}")
                self.log_confusion_matrix(y_test,y_pred,idx,name)
            self.writer.close()
        except Exception as e:
            raise CustomException("error while training and evaluation",sys)
    def run(self):
        try:
            logger.info("starting model pipeline ")
            x,y=self.load_data()
            x_train,x_test,y_train,y_test=self.split_data(x,y)
            self.train_evaluate(x_train,x_test,y_train,y_test)
            logger.info("moddel selection pipeline completed")
        except Exception as e:
            logger.error("error in pipeline")
            raise CustomException("error in the pipeline",sys)        

if __name__=="__main__":
    model_selection=ModelSelection(Engineer_data_path)     
    model_selection.run()                       
