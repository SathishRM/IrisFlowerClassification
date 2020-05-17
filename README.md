### IrisFlowerClassification
Trains a logistic regression model using the Iris dataset.

Use spark-submit to run the script, provided adding the pyspark installation directory to PATH variable and have the soure code path in the variable PYTHON_PATH

### Model Training ###
Command: spark-submit --master spark://[hostname]:[port#] irisdatatraining.py

Reads the data in the CSV file from the input path, tune it with the parameters configured then the best model is saved for further predictions.
Features used are sepal_length, sepal_width, petal_length and petal_width.
This contians 3 labels Iris-setosa, Iris-versicolor and Iris-virginica to predict.

Pipeline is used to build the transformation of data which will be fed to the model creation. The model is tuned with the set of parameters and choosen the best one using an evaluator with the metric configured.

###Predict Flower###
command: spark-submit --master spark://[hostname]:[port#] irisflowerpredict.py [sepalLength] [sepalWidth] [petalLength] [petalWidth]
Argument values are the measurements in centimeters 

