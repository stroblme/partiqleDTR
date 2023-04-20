import optuna as o
from optuna_dashboard import set_objective_names
from typing import List, Dict
import time
import mlflow


class Hyperparam_Optimizer:
    def __init__(
        self, name: str, seed: int, path:str, n_trials: int, timeout: int, selective_optimization:bool=False, toggle_classical_quant:bool=False, resume_study:bool=False
    ):
        # storage = self.initialize_storage(host, port, path, password)

        pruner = o.pruners.NopPruner()
        # pruner = o.pruners.PercentilePruner(10.0, n_warmup_steps=2, n_startup_trials=20)

        sampler = o.samplers.TPESampler(seed=seed, multivariate=True)

        self.n_trials = n_trials
        self.timeout = timeout
        self.toggle_classical_quant = toggle_classical_quant
        self.selective_optimization = selective_optimization

        self.study = o.create_study(
            pruner=pruner,
            sampler=sampler,
            directions=["maximize", "minimize", "maximize"],
            load_if_exists=resume_study,
            study_name=name,
            storage=f"sqlite:///{path}",
        )
        set_objective_names(self.study, ["Accuracy", "Loss", "Perfect LCAG"])

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

    def log_study(self):
        for trial in self.study.best_trials:
            mlflow.log_params({f"trial_{trial.number}_{k}":v for k, v in trial.params.items()})

            mlflow.log_metric(f"trial_{trial.number}_accuracy", trial.values[0])
            mlflow.log_metric(f"trial_{trial.number}_loss", trial.values[1])
            mlflow.log_metric(f"trial_{trial.number}_perfect_lcag", trial.values[2])


    def update_variable_parameters(self, trial, parameters):
        updated_variable_parameters = dict()
        for parameter, value in parameters.items():
            if "_range_quant" in parameter:
                if not self.toggle_classical_quant and self.selective_optimization:
                    continue # continue if its a quantum parameter and we are classical
                param_name = parameter.replace("_range_quant", "")
            else:
                if self.toggle_classical_quant and self.selective_optimization:
                    continue # continue if its a classical parameter and we are quantum
                param_name = parameter.replace("_range", "")


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
                if isinstance(value[2], str):
                    if value[2] == "log":
                        log = True
                        step = None
                    else:
                        log = False
                        step = value[0]
                else:
                    log = False
                    step = value[2]

                #
                if isinstance(low, float) and isinstance(high, float):
                    updated_variable_parameters[param_name] = trial.suggest_float(
                        param_name, value[0], value[1], step=step, log=log
                    )
                elif isinstance(low, int) and isinstance(high, int):
                    updated_variable_parameters[param_name] = trial.suggest_int(
                        param_name, value[0], value[1], step=1, log=log
                    )
                else:
                    raise ValueError(
                        f"Unexpected type of range for trial suggestion for parameter {param_name}. Expected one of 'float' or 'int', got [{type(low)}, {type(high)}]."
                    )

            else:
                updated_variable_parameters[param_name] = trial.suggest_categorical(
                    param_name, value
                )

        return updated_variable_parameters

    def minimize(self):
        self.study.optimize(self.run_trial, n_trials=self.n_trials)

    def run_trial(self, trial=None):
        if trial is None:
            trial = self.study.ask()

        updated_variable_model_parameters = self.update_variable_parameters(
            trial, self.variable_model_parameters
        )
        model_parameters = (
            self.fixed_model_parameters | updated_variable_model_parameters
        )
        model = self.create_model(**model_parameters)["model"]

        # gather all parameters and create instructor
        updated_variable_instructor_parameters = self.update_variable_parameters(
            trial, self.variable_instructor_parameters
        )
        instructor_parameters = (
            self.fixed_instructor_parameters | updated_variable_instructor_parameters
        )
        instructor_parameters[
            "model"
        ] = model  # update this single parameter using the returned model
        instructor_parameters["report_callback"] = trial.report
        instructor_parameters["early_stop_callback"] = self.early_stop_callback
        instructor = self.create_instructor(**instructor_parameters)["instructor"]

        metrics = self.objective(instructor)["metrics"]

        return [metrics["accuracy"], metrics["loss"], metrics["perfect_lcag"]]
