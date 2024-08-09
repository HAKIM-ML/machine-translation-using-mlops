from machinetranslation.config.configuration import ConfigurationManager
from machinetranslation.conponents.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_dataset, eval_dataset = data_transformation.transform_data()
        data_transformation.save_datasets(train_dataset, eval_dataset)