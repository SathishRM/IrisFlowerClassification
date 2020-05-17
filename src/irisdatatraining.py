from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from util.applogger import getAppLogger
from util.appconfigreader import AppConfigReader
import os

def loadCsvFiles(spark, schema, csvDir):
    '''Loads CSV files from the given path and returns a dataframe
    Args: spark - Active spark session
          schema - Schema of the data in csv files
          csvDir - Path of csv files
    '''
    try:
        loadedData = spark.read.format('csv').schema(schema).load(csvDir)
        return loadedData
    except Exception as error:
        logger.exception(f"Failed to load csv files {error}")
        raise
    else:
        logger.info("CSV files are loaded successfully")

def createPipeline(lrElasticNetParam, lrRegParam):
    '''Creates a pipeline for coverting the data into features and label with the required format
    '''
    strIndexer = StringIndexer().setInputCol('species').setOutputCol('label')
    va = VectorAssembler(inputCols=['sepal_length','sepal_width','petal_length','petal_width'], outputCol='vec_features')
    ss = StandardScaler().setInputCol(va.getOutputCol()).setOutputCol('features')
    lr = LogisticRegression().setFeaturesCol(ss.getOutputCol())
    stages = [strIndexer, va, ss, lr]
    pipeline = Pipeline().setStages(stages)

    params = ParamGridBuilder().addGrid(lr.elasticNetParam,lrElasticNetParam).addGrid(lr.regParam,lrRegParam).build()
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName=lrMetric)

    return pipeline, params, evaluator

def trainLRModel(trainData, pipeline, params, evaluator, numFolds=3):
    '''Trains the model and select the best model based on the evaluation choosen
    Args: trainData - Data to train LR model
          pipeline - Pipeline with set of transformers and estimators
          params - List of parameters used for the model tuning
          evaluator - Evaluates the model
          numFolds - Number of splitting data into a set of folds
    '''
    crossValidator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=numFolds)
    lrModels = crossValidator.fit(trainData)
    return lrModels.bestModel.stages[3]

if __name__ == '__main__':
    try:
        logger = getAppLogger(__name__)

        # Read application CFG file
        logger.info('Read application parameters from config file')
        appConfigReader = AppConfigReader()
        print(appConfigReader.config)
        if 'APP' in appConfigReader.config:
            appCfg = appConfigReader.config['APP']
            inputDir = appCfg['CSV_DIR']
            lrElasticNetParam = [float(n) for n in appCfg['LR_ELASTIC_NET_PARAM'].split(',')]
            lrRegParam = [float(n) for n in appCfg['LR_REG_PARAM'].split(',')]
            lrMetric = appCfg['LR_METRIC_NAME']
            numFolds = int(appCfg['DATA_SPLIT_COUNT'])
            modelDir = appCfg['LR_MODEL_DIR']
        else:
            logger.error('Application details are missed out to configure')
            raise SystemExit(1)

        logger.info('Create a spark session')
        spark = SparkSession.builder.appName('IrisFlowerClassification').getOrCreate()
        if spark:
            manualSchema = StructType([StructField('sepal_length', DoubleType(), False), StructField('sepal_width', DoubleType(), False),
                                       StructField('petal_length', DoubleType(), False), StructField('petal_width', DoubleType(), False),
                                       StructField('species', StringType(), False)])
            logger.info('Load the csv files form the input directory')
            irisData= loadCsvFiles(spark, manualSchema, inputDir)
            irisColumns = irisData.columns
            irisDataTrain, irisDataTest = irisData.randomSplit([0.7,0.3])

            logger.info('Create a pipeline with the set of transformer and estimators')
            pipeline, params, evaluator = createPipeline(lrElasticNetParam, lrRegParam)

            logger.info('Train and tune the mode with the parameters configured')
            lrModel = trainLRModel(irisDataTrain, pipeline, params, evaluator, numFolds)
            logger.info('Save the trained model')
            lrModel.write().overwrite().save(modelDir)

            logger.info('Details about the trained model')
            # logger.info(f'Evaluation based on the metric {lrMetric} is {evaluator.evaluate(lrModel.transform(irisDataTest))}')
            logger.info(f'Precision: {lrModel.summary.weightedPrecision}')
            logger.info(f'Recall: {lrModel.summary.weightedRecall}')
            logger.info(f'Weighted F Score: {lrModel.summary.weightedFMeasure()}')
            logger.info(f'True Positive Rate: {lrModel.summary.weightedTruePositiveRate}')
            logger.info(f'False Positive Rate: {lrModel.summary.weightedFalsePositiveRate}')
            logger.info(f'Parameters used: {lrModel.extractParamMap()}')



    except Exception as error:
        logger.exception(f'Something went wrong here {error}')
    else:
        logger.info('Model has been trained and ready for any new prediction')
    finally:
        spark.stop()
