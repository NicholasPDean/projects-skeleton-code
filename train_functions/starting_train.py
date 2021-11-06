from networks.StartingNetwork import StartingNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter as SummaryWriter
from tqdm import tqdm as tqdm
import os

def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = SummaryWriter(summary_path)    

    step = 0
    validate_runs = 0
    best_val_acc = 0
    for epoch in range(epochs):
        loss_sum = 0
        num_correct = 0
        num_total = 0

        # Loop over each batch in the dataset
        loop = tqdm(train_loader, desc=f"Train {epoch}")
        for batch in loop:
            # Backpropagation and gradient descent

            images, labels = batch
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 50:
                val_accuracy, val_loss = evaluate(val_loader, model, loss_fn, validate_runs)
                best_val_acc = max(best_val_acc, val_accuracy)

                writer.add_scalar('acc/val', val_accuracy, validate_runs)
                writer.add_scalar('loss/val', val_loss, validate_runs)
                writer.add_scalar('acc/best_val', best_val_acc, validate_runs)
                writer.flush()

                validate_runs += 1

            # per batch accuracy
            batch_accuracy, batch_correct, batch_total = compute_accuracy(outputs, labels)

            num_correct += batch_correct
            num_total += batch_total

            # update tqdm string
            loop.set_postfix({"acc:": f"{batch_accuracy : .03f}"})
            
            step += 1
        
        train_accuracy = num_correct / num_total
        train_loss = loss_sum / len(train_loader)

        writer.add_scalar('acc/train', train_accuracy, epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.flush()

        print()

    writer.close()

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    n_correct = (outputs.argmax(dim=1) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total, n_correct, n_total

def evaluate(val_loader, model, loss_fn, validate_runs):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    loss_sum = 0
    num_correct = 0
    num_total = 0

    model.eval()
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Valid {validate_runs}")
        for batch in loop:

            images, labels = batch
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)

            loss_sum += loss.item()

            batch_accuracy, batch_correct, batch_total = compute_accuracy(outputs, labels)

            num_correct += batch_correct
            num_total += batch_total

            loop.set_postfix({"acc:": f"{batch_accuracy : .03f}"})

    model.train()

    validation_accuracy = num_correct / num_total
    validation_loss = loss_sum / len(val_loader)

    return validation_accuracy, validation_loss
