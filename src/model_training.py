from .torch_datasets import generate_datasets
from .trainer import Trainer
from .models import (LSTM_MDN, GRU_MDN, RNN_MDN, TCN_MDN, Transformer_MDN)
from .loss import mdn_loss
import json
import hashlib
import torch
import optuna

import typing
import pandas as pd

from src.constants import MODELS_DIR

def lstm_mdn_objective(trial, dataset_df : pd.DataFrame, num_epochs : int = 30):
    try:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        num_mixtures = trial.suggest_int("num_mixtures", 3, 8)
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lookback_days = trial.suggest_categorical("lookback_days", [1, 2, 4])
        step = trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        lookback_hours = lookback_days * 24

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=lookback_hours,
            delay=24,   # predict 1 day ahead (constant)
            step=step
        )

        model = LSTM_MDN(
            input_dim=21,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            output_dim=7,
            dropout=dropout
            # fc_dim=fc_dim
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        param_str = json.dumps(trial.params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        model_pth = f"transformer_mdn_{param_hash}_checkpoint.pth"
        trainer = Trainer(
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            model_pth=model_pth,
            batch_size=256
        )

        trainer.train(num_epochs=num_epochs)

        return trainer.history['val_loss'][-1]

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")
    
def gru_mdn_objective(trial, dataset_df : pd.DataFrame, num_epochs : int = 30):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    num_mixtures = trial.suggest_int("num_mixtures", 3, 8)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lookback_days = trial.suggest_categorical("lookback_days", [1, 2, 4])
    step = trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    lookback_hours = lookback_days * 24

    training_dataset, validation_dataset = generate_datasets(
        dataset_df,
        lookback=lookback_hours,
        delay=24,  # 1-day ahead prediction
        step=step
    )

    model = GRU_MDN(
        input_dim=21,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mixtures=num_mixtures,
        output_dim=7,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    param_str = json.dumps(trial.params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    model_pth = f"transformer_mdn_{param_hash}_checkpoint.pth"
    trainer = Trainer(
        model=model,
        criterion=mdn_loss,
        optimizer=optimizer,
        train_dataset=training_dataset,
        val_dataset=validation_dataset,
        model_pth=model_pth,
        batch_size=256
    )

    trainer.train(num_epochs=num_epochs)

    return trainer.history['val_loss'][-1]

def rnn_mdn_objective(trial, dataset_df : pd.DataFrame, num_epochs : int = 30):
    try:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        num_mixtures = trial.suggest_int("num_mixtures", 3, 8)
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lookback_days = trial.suggest_categorical("lookback_days", [1, 2, 4])
        step = trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        lookback_hours = lookback_days * 24

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=lookback_hours,
            delay=24,  # 1-day ahead forecast
            step=step
        )

        model = RNN_MDN(
            input_dim=21,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            output_dim=7,
            dropout=dropout
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        param_str = json.dumps(trial.params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        model_pth = f"transformer_mdn_{param_hash}_checkpoint.pth"
        
        trainer = Trainer(
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            model_pth=model_pth,
            batch_size=256
        )

        trainer.train(num_epochs=num_epochs)

        return trainer.history['val_loss'][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")

def tcn_mdn_objective(trial, dataset_df : pd.DataFrame, num_epochs : int = 30):
    try:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 5)
        num_mixtures = trial.suggest_int("num_mixtures", 3, 8)
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lookback_days = trial.suggest_categorical("lookback_days", [1, 2, 4])
        step = trial.suggest_categorical("step", [1, 2, 3, 4, 6])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        lookback_hours = lookback_days * 24

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=lookback_hours,
            delay=24,  # 1-day ahead prediction
            step=step
        )

        model = TCN_MDN(
            input_dim=21,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            output_dim=7,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        param_str = json.dumps(trial.params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        model_pth = f"transformer_mdn_{param_hash}_checkpoint.pth"
        trainer = Trainer(
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            model_pth=model_pth,
            batch_size=256
        )

        trainer.train(num_epochs=num_epochs)

        return trainer.history['val_loss'][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")
    
def transformer_mdn_objective(trial, dataset_df : pd.DataFrame, num_epochs : int = 30):
    try:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_mixtures = trial.suggest_int("num_mixtures", 3, 8)
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lookback_days = trial.suggest_categorical("lookback_days", [1, 2, 4])
        step = trial.suggest_categorical("step", [1, 2, 3, 4, 6])

        lookback_hours = lookback_days * 24

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=lookback_hours,
            delay=24,  # predict 1 day ahead
            step=step
        )

        model = Transformer_MDN(
            input_dim=21,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            output_dim=7,
            dropout=dropout
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        param_str = json.dumps(trial.params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        model_pth = f"transformer_mdn_{param_hash}_checkpoint.pth"

        trainer = Trainer(
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            model_pth=model_pth,
            batch_size=256
        )

        trainer.train(num_epochs=num_epochs)

        return trainer.history['val_loss'][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")
    
def run_training(dataset_df : pd.DataFrame, num_trials: int = 30, num_epochs: int = 30):
    gru_study = optuna.create_study(
        direction="minimize",
        study_name="gru_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True
    )
    gru_study.optimize(lambda trial: gru_mdn_objective(trial, dataset_df, num_epochs), n_trials=num_trials)

    rnn_study = optuna.create_study(
        direction="minimize",
        study_name="rnn_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True
    )
    rnn_study.optimize(lambda trial: rnn_mdn_objective(trial, dataset_df, num_epochs), n_trials=num_trials)

    tcn_study = optuna.create_study(
        direction="minimize",
        study_name="tcn_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True
    )
    tcn_study.optimize(lambda trial: tcn_mdn_objective(trial, dataset_df, num_epochs), n_trials=num_trials)

    transformer_study = optuna.create_study(
        direction="minimize",
        study_name="transformer_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True
    )
    transformer_study.optimize(lambda trial: transformer_mdn_objective(trial, dataset_df, num_epochs), n_trials=num_trials)
