from pyspark.sql import SparkSession
import sys
APP_NAME = "CS5052 P1 - Apache Spark"

def main(spark, output):
    ratings = spark.read.csv("ml-latest-small/ratings.csv", header=True)
    ratings.show()




if __name__ == "__main__":
        
    #Configure Spark Job
    spark = SparkSession.builder.master("local").appName(APP_NAME).getOrCreate()
    #if using filename parameter in main method then i can use
    output = sys.argv[1]

    #Execute the program
    main(spark,output)
