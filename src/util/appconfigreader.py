from configparser import ConfigParser
import os


class AppConfigReader():
    '''Loads the config file values'''

    def __init__(self):
        self.config = ConfigParser()
        # Get the config file path from environmental variable PY_APP_CONFIG
        cfgDir = os.environ.get('CFG_DIR')
        if cfgDir:
            cfgFile = cfgDir + "\\irisflowerclassification.properties"
        else:
            cfgFile = "E:\\Spark\\github\\IrisFlowerClassification\\conf\\irisflowerclassification.properties"

        # Load the CFG file
        self.config.read(cfgFile)
