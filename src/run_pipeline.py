import argparse
from .data_preprocessing import get_preprocessed_data, get_raw_data
from .feature_engineering import get_feature_engineered_data
from .model_training import run_training

def main():
    parser = argparse.ArgumentParser(description="Run the data preprocessing and model training pipeline.")
    parser.add_argument('--dry-run', action='store_true', help='Run the script in dry run mode (1 epoch and 1 tries per model).')
    parser.add_argument('--num-trials', type=int, default=30,
                        help='Number of trials to run (default: 30)')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of epochs to train each model (default: 30)')
    
    args = parser.parse_args()

    data = get_raw_data()
    data = get_preprocessed_data(data)
    data = get_feature_engineered_data(data)

    if args.dry_run:
        print("Running in dry run mode. Only 1 epoch and 1 trial will be executed.")
        run_training(data, num_trials=1, num_epochs=1)
    else:
        run_training(data, num_trials=args.num_trials, num_epochs=args.num_epochs)

    print(args)
    
if __name__ == "__main__":
    main()