import pandas as pd
from config.path_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import sys
 
logger=get_logger(__name__)

class DataProcessor:
    def __init__(self):
        self.train_path=TRAIN_DATA_PATH
        self.processed_data_path=PROCESSED_DATA_PATH
   
    def load_data(self):
        try:
            logger.info("data processing started")
            df=pd.read_csv(self.train_path)
            logger.info("data read successfully")
            return df
        except Exception as e:
            logger.error("problem while loading data")
            raise CustomException("error while loading data :", sys)

    def drop_unneccesary_columns(self, df, columns):
        try:
            logger.info(f"dropping unneccessary column: {columns} ")
            df=df.drop(columns=columns, axis=1)
            logger.info("columns dropped successfully")
            return df
        except Exception as e:
            logger.error("problem while dropping columns")
            raise CustomException("dropping error :", sys)
    def handle_outlier(self,df,columns):
        try:
            logger.info("handling outlier")
            for column in columns:
                 Q1=df[column].quantile(0.25)
                 Q3=df[column].quantile(0.75)
                 IQR=Q3-Q1

                 lower_bound=Q1-1.5*IQR
                 upper_bound=Q3+1.5*IQR
                 df[column]=df[column].clip(lower=lower_bound, upper=upper_bound)
            logger.info("outlier handle successfully")
            return df
        except Exception as e:
            logger.error("problem while handling outlier")
            raise CustomException("problem while outlier handler :", sys)
    def handle_null_value(self,df,columns):
        try:
            logger.info("handle null value")
            df[columns]=df[columns].fillna(df[columns].median())
            return df
        except Exception as e:
            logger.error("problem while handling null value")
            raise CustomException("problem while handling null value :", sys)
    def save_data(self,df):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
        logger.info("processed data saved successfully")

    def run(self):
        try:

            logger.info("starting the pipeline of data processing")
            df=self.load_data()
            df=self.drop_unneccesary_columns(df,["MyUnknownColumn","id"])
            column_to_handle=['Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes','Checkin service']
            df=self.handle_outlier(df, column_to_handle)  
            df=self.handle_null_value(df,"Arrival Delay in Minutes")      
            self.save_data(df)
            logger.info("data processing completred successfully")
        except CustomException as ce:
            logger.error(f"error occured in data processing pipeline")
        

if __name__== "__main__":
    processor=DataProcessor()
    processor.run()
