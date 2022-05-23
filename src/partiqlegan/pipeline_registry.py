"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from partiqlegan.pipelines import data_generation as dg
from partiqlegan.pipelines import data_processing as dp

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_generation_artificial_pipeline = dg.create_artificial_pipeline()
    data_generation_belleII_pipeline = dg.create_belleII_pipeline()
    data_processing_pipeline = dp.create_pipeline()

    return {
        "__default__": data_generation_artificial_pipeline+data_processing_pipeline,
        "data_generation_artificial_pipeline": data_generation_artificial_pipeline,
        "data_generation_belleII_pipeline": data_generation_belleII_pipeline,
        "data_processing_pipeline": data_processing_pipeline,
    }
