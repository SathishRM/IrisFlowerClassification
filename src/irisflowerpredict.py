from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import argparse
from util.applogger import getAppLogger
from util.appconfigreader import AppConfigReader

def flowerPrediction(sepalLength, sepalWidth, petalLength, petalWidth):
    irisData = spark.createDataFrame([(sepalLength,sepalWidth,petalLength,petalWidth)],['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    # vaData = VectorAssembler(inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], outputCol='vecFeatures').transform(irisData)
    # modelData = StandardScaler(inputCol= 'vecFeatures', outputCol='features').fit(vaData).transform(vaData)
    pipelineModel = PipelineModel.load(modelDir)
    if pipelineModel:
        vaData = pipelineModel.stages[1].transform(irisData)
        modelData = pipelineModel.stages[2].transform(vaData)
        lrModel = pipelineModel.stages[3]
        indexString = pipelineModel.stages[4]
        if vaData and modelData and lrModel and indexString:
            logger.info('Model has been loaded and going to predict for the values passed')
            prediction = lrModel.transform(modelData).select('prediction')
            predictedValue = indexString.transform(prediction).collect()
            return predictedValue[0].predictedLabel
        else:
            logger.error(f'Problem in loading the one of the stages from the model saved in the location {modelDir}')
            raise SystemExit(3)
    else:
        logger.error(f'Problem in loading the model from the directory {modelDir}')
        raise SystemExit(3)



if __name__ == '__main__':
    try:
        logger = getAppLogger(__name__)

        # Read application CFG file
        logger.info('Read application parameters from config file')
        appConfigReader = AppConfigReader()
        if 'APP' in appConfigReader.config:
            appCfg = appConfigReader.config['APP']
            modelDir = appCfg['LR_MODEL_DIR']
        else:
            logger.error('Application details are missed out to configure')
            raise SystemExit(1)

        #Parse the arguments passed
        argParser = argparse.ArgumentParser()
        argParser.add_argument('sepalLength', type=float, help="Length of sepal in CM")
        argParser.add_argument('sepalWidth', type=float, help="Width of sepal in CM")
        argParser.add_argument('petalLength', type=float, help="Length of petal in CM")
        argParser.add_argument('petalWidth', type=float, help="Width of petal in CM")
        args = argParser.parse_args()
        logger.info(f'Arguments received are SepalLength:{args.sepalLength}, SepalWidth:{args.sepalWidth}, PetalLength:{args.petalLength}, PetalWidth:{args.petalWidth}')

        if args.sepalLength and args.sepalWidth and args.petalLength and args.petalWidth:
            logger.info('Create a spark session')
            spark = SparkSession.builder.appName('IrisFlowerPrediction').getOrCreate()
            if spark:
                logger.info('Predicts the flower for the given values...')
                predictedValue = flowerPrediction(args.sepalLength,args.sepalWidth,args.petalLength,args.petalWidth)
                logger.info(f'Value predicted is {predictedValue}')
        else:
            logger.error(f'Missing some required parameters, please checkt the script help for the list of arguments required')
            raise SystemExit(2)
    except Exception as error:
        logger.exception(f'Something went wrong here {error}')
    else:
        logger.info('Prediction has completed')
    finally:
        spark.stop()
