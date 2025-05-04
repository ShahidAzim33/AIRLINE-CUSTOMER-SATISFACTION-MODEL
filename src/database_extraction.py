import os
import csv
import mysql.connector
from mysql.connector import Error
from config.db_config import DB_CONFIG
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)
class MySQLDataExtractor:
    def __init__(self, db_config):
        self.host=db_config["host"]
        self.user=db_config["user"]
        self.password=db_config["password"]
        self.database=db_config["database"]
        self.table_name=db_config["table_name"]
        self.connection=None

        logger.info("your databse configuration has been setup")

    def connect(self):
        try:
            self.connection=mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
                  )
            if self.connection.is_connected():
                logger.info("succesfully connected to the database")
        except Error as e:
            raise CustomException("error while connection to the database,{e}")
    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("disconnect to the database")

    def extract_to_csv(self,output_folder="./artifacts/raw"):
        try:
        
            if not self.connection or not self.is_connected():
                self.connect()
  
            cursor=self.connection.cursor()
            query=f"SELECT * FROM {self.table_name}"
            cursor.execute(query)
            rows=cursor.fetchall()
            columns=[desc[0] for desc in cursor.description]
            logger.info("data fetch successfully")
            os.makedirs(output_folder, exist_ok=True)
            csv_file_path=os.path.join(output_folder,"data.csv")

            with open(csv_file_path,mode='w', newline="", encoding="utf-8") as file:
                writer=csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)
                logger.info("data successfully saved to {csv_file_path}")
        except Error as e:
            raise CustomException("error in extracting db due to error {e}")
        except CustomException as ce:
            logger.error(str(ce))
        finally:
            if 'cursor' in locals():
                cursor.close()
            self.disconnect()

if __name__== '__main__':   
    try:
        extractor=MySQLDataExtractor(DB_CONFIG)
        extractor.extract_to_csv()
    except CustomException as ce:
        logger.error(str(ce))                             

              

                           
         