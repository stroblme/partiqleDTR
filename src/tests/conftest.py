import pytest

@pytest.fixture
def data_generation_parameters():
    parameters = {
        "masses": [100, 90, 80, 70, 50, 20, 25, 10],
        "fsp_masses": [1, 2, 3, 5, 12],
        "n_topologies": 5,
        "max_depth": 3,
        "max_children": 3,
        "min_children": 2,
        "isp_weight": 1.0,
        "iso_retries": 0,
        "seed": 1117,
    }
    return parameters