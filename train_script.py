"""
Script to start a training run.
"""
import os
import sys
import argparse
import logging
import json
import datetime
import mlflow

from common.common import recursive_dict_filling, flatten_dict, load_and_merge_parameters
from trainers.trainers import initialize_trainer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("log.log"),
        logging.StreamHandler()
    ]
)


def main(arguments):
    # Parameters
    model_parameters = arguments["model_parameters"]
    data_parameters = arguments["data_parameters"]
    train_parameters = arguments["train_parameters"]

    # Train context
    use_mlflow = arguments["use_mlflow"]
    experiment_name = arguments["experiment_name"]
    run_name = arguments["run_name"]

    unique_name = run_name + '-' + datetime.datetime.now().strftime('%H%M%S')

    # Initiate trainer
    trainer, parameters = initialize_trainer(unique_name, model_parameters, data_parameters, **train_parameters)

    # Log parameters to mlflow
    if use_mlflow:
        LOGGER.info(f"Using MLFlow with experiment name {experiment_name}")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=unique_name)
        arguments_to_show = {
            "model": model_parameters,
            "train": train_parameters,
            "data": data_parameters
        }
        mlflow.log_params(dict(flatten_dict(arguments_to_show)))

    # Start training
    try:
        output_folder, metrics = trainer.train()

        # Log metrics and artifacts to mlflow
        if use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(output_folder)
    except AssertionError:
        pass
    finally:
        if use_mlflow:
            mlflow.log_artifact('log.log')
            mlflow.end_run()


if __name__ == '__main__':
    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_parameters', type=json.loads, default=dict())
    parser.add_argument('--train_parameters', type=json.loads, default=dict())
    parser.add_argument('--data_parameters', type=json.loads, default=dict())
    parser.add_argument('--parameter_path', type=str, default=None)

    parser.add_argument('--use_mlflow', type=json.loads, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default='run_name')

    args, unknown = parser.parse_known_args()
    argument_dict = vars(args)

    # Parse arguments with . notation format
    for arg in unknown:
        if arg.startswith("--"):
            parser.add_argument(arg, type=json.loads, default=None)
    unknown_argument_dict = vars(parser.parse_args())
    for name, value in unknown_argument_dict.items():
        keys = name.split('.')
        if keys[0] in argument_dict.keys():
            recursive_dict_filling(argument_dict, keys, value)

    # Merge parameters from file with argument dict, argument dict is leading
    parameter_path = argument_dict['parameter_path']
    if parameter_path is not None:
        assert os.path.isfile(parameter_path), 'Parameter path not found!'
        argument_dict = load_and_merge_parameters(argument_dict, parameter_path, ignore_unused_parameters=True)

    main(argument_dict)
