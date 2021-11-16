from PIL.Image import LINEAR
from networks.StartingNetwork import StartingNetwork
import torch
import torch.nn as nn
import torch.optim as optim
# tensorboard 
from torch.utils.tensorboard import SummaryWriter as SummaryWriter
from tqdm import tqdm as tqdm
import time
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
    name = time.ctime()
    if summary_path is not None:
        writer = SummaryWriter(os.path.join(summary_path, name))    

    step = 0
    best_val_acc = 0
    for epoch in range(epochs):
        #print(f"Epoch {epoch + 1} of {epochs}")

        loss_sum = 0
        validate_runs = 0

        # Loop over each batch in the dataset
        loop = tqdm(train_loader, desc=f"Train {epoch}")
        for batch in loop:
            #print(f"\rIteration {i + 1} of {len(train_loader)} for training loop...", end="")

            # TODO: Backpropagation and gradient descent
            images, labels = batch

            outputs = model.forward(images)
            
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()

            # Periodically evaluate our model + log to Tensorboard
            #if step % n_eval == 50:
                # TODO:
                # Compute training loss and accuracy.

                #print('Training accuracy: ' + str(train_accuracy))
                # Log the results to Tensorboard.
                
                # TODO: split the dataset into training and validation: (we did the splitting in main.py)
                # 80% training, 20% validation
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!


            train_accuracy = compute_accuracy(outputs, labels)
            train_loss = loss_sum / len(outputs)
            # update loop
            loop.set_postfix({"acc:": f"{train_accuracy : .03f}", "loss:": f"{loss.item() : .03f}"})
            writer.flush() 

            step += 1

        # go through the whole training loop first, and then validate
        val_accuracy, val_loss = evaluate(val_loader, model, loss_fn, validate_runs)
        best_val_acc = max(best_val_acc, val_accuracy)

        writer.add_scalar('acc/val', val_accuracy, step)
        writer.add_scalar('loss/val', val_loss, step)
        writer.add_scalar('acc/best_val', best_val_acc, step)

        validate_runs += 1
        writer.add_scalar('acc/train', train_accuracy, epoch)
        writer.add_scalar('loss/train', train_loss, epoch) 

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
    #print("n_corrrect: ", n_correct) # just for debugging, delete later
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, validate_runs):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    loss_sum = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Valid {validate_runs}")
        for batch in loop:
            #print(f"\rIteration {i + 1} of {len(val_loader)} for validation loop...", end="")

            images, labels = batch

            outputs = model.forward(images)

            loss = loss_fn(outputs, labels)
            loss_sum += loss.item()
            validation_accuracy = compute_accuracy(outputs, labels)
            #print('Validation accuracy: ' + str(compute_accuracy(outputs, labels)))
            loop.set_postfix({"acc:": f"{validation_accuracy : .03f}", "loss:": f"{loss_sum / len(outputs) : .03f}"})
    model.train()
    return validation_accuracy, loss_sum / len(outputs)
