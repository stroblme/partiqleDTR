"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from partiqlegan.pipelines import data_generation as dg
from partiqlegan.pipelines import data_processing as dp
from partiqlegan.pipelines import data_science as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_generation_artificial_pipeline = dg.create_artificial_pipeline()
    data_generation_belleII_pipeline = dg.create_belleII_pipeline()
    data_processing_artificial_pipeline = dp.create_artificial_pipeline()
    data_processing_artificial_pipeline_no_shuffle = dp.create_artificial_pipeline_no_shuffle()
    data_processing_belleII_pipeline = dp.create_belleII_pipeline()
    training_qgnn_pipeline = ds.create_training_qgnn_pipeline()
    training_qgnn_pipeline_no_param_log = ds.create_training_qgnn_pipeline_no_param_log()

    return {
        "__default__": data_generation_artificial_pipeline+data_processing_artificial_pipeline+training_qgnn_pipeline_no_param_log,
        "default": data_generation_artificial_pipeline+data_processing_artificial_pipeline+training_qgnn_pipeline,
        "data_generation_artificial_pipeline": data_generation_artificial_pipeline,
        "data_generation_belleII_pipeline": data_generation_belleII_pipeline,
        "data_processing_artificial_pipeline": data_processing_artificial_pipeline,
        "data_processing_artificial_pipeline_no_shuffle": data_processing_artificial_pipeline_no_shuffle,
        "data_processing_belleII_pipeline": data_processing_belleII_pipeline,
        "training_qgnn_pipeline": training_qgnn_pipeline,
    }
