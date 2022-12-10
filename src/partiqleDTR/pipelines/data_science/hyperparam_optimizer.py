import optuna as o
from typing import List, Dict
import time


class Hyperparam_Optimizer:
    def __init__(
        self, name: str, id: int, host: str, port: int, path: str, password: str
    ):
        storage = self.initialize_storage(host, port, path, password)

        pruner = o.pruners.NopPruner()
        # pruner = o.pruners.PercentilePruner(10.0, n_warmup_steps=2, n_startup_trials=20)

        sampler = o.samplers.TPESampler(seed=id, multivariate=True)

        self.study = o.create_study(
            pruner=pruner,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
            study_name=name,
            storage=storage,
        )

    def initialize_storage(self, host: str, port: int, path: str, password: str):
        """
        Storage intialization
        """
        storage = o.storages.RedisStorage(
            url=f"redis://{password}@{host}:{port}/{path}",
        )
        return storage

    def set_variable_parameters(self, model_parameters, instructor_parameters):
        assert isinstance(model_parameters, Dict)
        assert isinstance(instructor_parameters, Dict)

        self.variable_model_parameters = model_parameters
        self.variable_instructor_parameters = instructor_parameters

    def set_fixed_parameters(self, model_parameters, instructor_parameters):
        assert isinstance(model_parameters, Dict)
        assert isinstance(instructor_parameters, Dict)

        self.fixed_model_parameters = model_parameters
        self.fixed_instructor_parameters = instructor_parameters

    @staticmethod
    def create_model():
        raise NotImplementedError("Create Model method must be set!")

    @staticmethod
    def create_instructor():
        raise NotImplementedError("Create Instructor method must be set!")

    @staticmethod
    def objective():
        raise NotImplementedError("Objective method must be set!")

    def early_stop_callback(self):
        # TODO: Implement early stopping
        return False

    def update_variable_parameters(self, trial, parameters):
        updated_variable_parameters = dict()
        for parameter, value in parameters.items():
            assert isinstance(value, List)

            # if we have three values (-> no bool) and they are not categorical (str)
            if (
                len(value) == 3
                and not isinstance(value[0], str)
                and not isinstance(value[1], str)
            ):
                low = value[0]
                high = value[1]

                # if the third parameter specifies the scale
                if isinstance(value[2]) == str:
                    log = value[2] == "log"
                    step = None
                else:
                    log = False
                    step = value[2]

                #
                if isinstance(low, float) and isinstance(high, float):
                    updated_variable_parameters[parameter] = trial.suggest_float(
                        parameter, *value[1, 2], step=step, log=log
                    )
                elif isinstance(low, int) and isinstance(high, int):
                    updated_variable_parameters[parameter] = trial.suggest_categorical(
                        parameter, *value[1, 2], step=step, log=log
                    )
                else:
                    raise ValueError(
                        f"Unexpected type of range for trial suggestion for parameter {parameter}. Expected one of 'float' or 'int', got [{type(low)}, {type(high)}]."
                    )

            else:
                updated_variable_parameters[parameter] = trial.suggest_categorical(
                    parameter, *value
                )

        return updated_variable_parameters

    def minimize(self):
        startTime = time.time()

        while (time.time() - startTime) < self.duration:
            self.run_trial()

    def run_trial(self):
        trial = self.study.ask()

        updated_variable_model_parameters = self.update_variable_parameters(
            trial, self.variable_model_parameters
        )
        model_parameters = (
            self.fixed_model_parameters + updated_variable_model_parameters
        )
        model = self.create_model(*model_parameters)

        # gather all parameters and create instructor
        updated_variable_instructor_parameters = self.update_variable_parameters(
            trial, self.variable_instructor_parameters
        )
        instructor_parameters = (
            self.fixed_instructor_parameters + updated_variable_instructor_parameters
        )
        instructor_parameters[
            "model"
        ] = model  # update this single parameter using the returned model
        instructor_parameters["report_callback"] = trial.report
        instructor_parameters["early_stop_callback"] = self.early_stop_callback
        instructor = self.create_instructor(*instructor_parameters)

        artifact = self.objective(instructor)
