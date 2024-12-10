1. Prepare AWS Environment
Launch EC2 Instances:

Launch 4 EC2 instances with Ubuntu Linux AMI.
Ensure the instances are in the same security group with proper networking rules (open required ports such as 22 for SSH, 8080 for Spark, etc.).
Install Java, Python, and Apache Spark on all instances.
Set Up a Spark Cluster:

Designate one EC2 instance as the master node and others as worker nodes.
Configure Spark (conf/spark-env.sh and conf/slaves) to set up a multi-node cluster.
Upload Data to S3:

Upload TrainingDataset.csv and ValidationDataset.csv to an S3 bucket.
2. Train the ML Model with Spark MLlib
Write the Training Script: Save the following script as train_model.py.

python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load Training Data
training_data = spark.read.csv("s3://your-bucket/TrainingDataset.csv", header=True, inferSchema=True)

# Prepare Features
feature_columns = training_data.columns[:-1]  # Exclude target column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
training_data = assembler.transform(training_data).select("features", "quality")

# Train Model
logistic_regression = LogisticRegression(featuresCol="features", labelCol="quality", maxIter=10)
model = logistic_regression.fit(training_data)

# Save Model
model.save("s3://your-bucket/models/wine_quality_model")

spark.stop()
Run Training Script: Submit the job to the Spark cluster using spark-submit:


spark-submit --master spark://<master-node-ip>:7077 train_model.py
3. Validate and Optimize the Model
Write the Validation Script: Save the following script as validate_model.py.

python
Copy code
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityValidation").getOrCreate()

# Load Validation Data
validation_data = spark.read.csv("s3://your-bucket/ValidationDataset.csv", header=True, inferSchema=True)

# Prepare Features
feature_columns = validation_data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
validation_data = assembler.transform(validation_data).select("features", "quality")

# Load Model
model = LogisticRegressionModel.load("s3://your-bucket/models/wine_quality_model")

# Predict and Evaluate
predictions = model.transform(validation_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

print(f"F1 Score: {f1_score}")

spark.stop()
Run Validation Script:


spark-submit --master spark://<master-node-ip>:7077 validate_model.py
4. Dockerize the Prediction Application
Create a Prediction Script: Save the following script as predict_model.py.


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load Test Data
test_data = spark.read.csv("/app/TestDataset.csv", header=True, inferSchema=True)

# Prepare Features
feature_columns = test_data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_data = assembler.transform(test_data).select("features")

# Load Model
model = LogisticRegressionModel.load("/app/models/wine_quality_model")

# Predict
predictions = model.transform(test_data)
predictions.select("prediction").show()

spark.stop()
Create a Dockerfile: Save the following as Dockerfile.

dockerfile

FROM openjdk:11
COPY . /app
WORKDIR /app
CMD ["spark-submit", "predict_model.py"]
Build and Push Docker Image:


docker build -t wine-quality-prediction .
docker tag wine-quality-prediction:latest <your-dockerhub-username>/wine-quality-prediction:latest
docker push <your-dockerhub-username>/wine-quality-prediction:latest
Run Docker Container:

bash
Copy code
docker run -v /path/to/models:/app/models -v /path/to/TestDataset.csv:/app/TestDataset.csv wine-quality-prediction
5. Test on a Single EC2 Instance
Use the Test Dataset (TestDataset.csv) to ensure the application runs correctly and outputs the F1 score.
