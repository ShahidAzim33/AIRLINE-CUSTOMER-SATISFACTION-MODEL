import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
import sys
from utils.helpers import *

logger=get_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.data_path=PROCESSED_DATA_PATH
        self.df=None
        self.label_mapping={}

    def load_data(self):
        try:
            logger.info("loading data")
            self.df=pd.read_csv(self.data_path)
            logger.info("data loaded successfully")
        except CustomException as e:
            logger.error(f"error while loading data {e}")
            raise CustomException("error while loading data ", sys)
    def feature_construction(self):
        try:
            logger.info("starting feature construction")
            self.df['Arrival Delay in Minutes']=self.df['Arrival Delay in Minutes'].fillna(self.df['Arrival Delay in Minutes'].median())
            logger.info("feature construction completed")
        except CustomException as e:
            logger.error(f"error while Feature construction {e}")
            raise CustomException("error while feature construction ", sys)
    def bin_age(self):
        try:
            logger.info("starting binning of age columns")
            self.df["Age Group"]=pd.cut(self.df['Age'], bins=[0,18,30,50,100], labels=['child','youngster','Adult','senior'])
            logger.info("binning of age columns successful")
        except CustomException as e:
            logger.error(f"error while binning {e}")
            raise CustomException("error while binning ", sys)    
    def label_encoding(self):
        try:
            columns_to_encode=['Gender','Customer Type', 'Type of Travel','Class','satisfaction','Age Group']
            logger.info(f"Performing label encoding on {columns_to_encode}")
            self.df,self.label_mapping=label_encode(self.df,columns_to_encode)
            
            for col,mapping in self.label_mapping.items():
                logger.info(f"Mapping for {col} : {mapping}")
                logger.info("label encoding successfully")
        except CustomException as e:
            logger.error(f"error while label encoding {e}")
            raise CustomException("error while label encoding ", sys)        
        
    def feature_selection(self):
        try:
            logger.info("trying feature selection") 
            X=self.df.drop(columns='satisfaction')
            y=self.df['satisfaction'] 
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            mutual_info=mutual_info_classif(X_train,y_train,discrete_features=True)
            mutual_info_df=pd.DataFrame({
                                'Feature':X.columns,
                                'mutual_information':mutual_info,
                                   }).sort_values(by='mutual_information', ascending=False)
            logger.info(f"mutual information table is :{mutual_info_df}")
            top_features=mutual_info_df.head(12)['Feature'].tolist()
            self.df=self.df[top_features + ['satisfaction']]
            logger.info(f"top feature : {top_features}")
            logger.info("feature selection successfully")
        except CustomException as e:
            logger.error(f"error while feature selection {e}")
            raise CustomException("error while feature selection", sys)     
    def save_data(self):
        try:
            os.makedirs(Engineer_DIR,exist_ok=True)
            self.df.to_csv(Engineer_data_path,index=False)
            logger.info(f"data saved successfully, {Engineer_data_path}")
        except CustomException as e:
            logger.error(f"error while feature saving data {e}")
            raise CustomException("error while saving data", sys) 
    def run(self):
        try:
            logger.info("starting your feature engineer pipeline")
            self.load_data()
            self.feature_construction()
            self.bin_age()
            self.label_encoding()
            self.feature_selection()
            self.save_data()
            logger.info("your FE pipline successfully done")
        except CustomException as e:
            logger.error(f"error while FE pipleline{e}")
            raise CustomException("error while FE pipeline", sys)
        finally:
            logger.info("enf of FE pipliene")

if __name__=="__main__":
    feature_engineer=FeatureEngineer()
    feature_engineer.run()                         

