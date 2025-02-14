import argparse
import os
import matplotlib.pyplot as plt
from networks.ResNetwork import ResNetwork
import torch

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from networks.ResNetwork import ResNetwork
from train_functions.starting_train import starting_train

SUMMARIES_PATH = "training_summaries"

def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary_path = None
    if args.logdir is not None:
        summary_path = f"{SUMMARIES_PATH}/{args.logdir}"
        os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    dataset = StartingDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [round(0.8*dataset.__len__()), round(0.2*dataset.__len__())])
    #model = StartingNetwork()
    model = ResNetwork()
    model = model.to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--logdir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
