import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.misc import log_message, set_random_seed


# Unified training function
def train_model(
    model_class, 
    train_loader, 
    valid_loader, 
    lr=0.001, 
    num_epochs=10, 
    batch_size=32, 
    kl_weight=1e-3, 
    use_pretrained=False, 
    checkpoint_path=None, 
    device="cuda",
    save_dir=os.getcwd(),
    random_seed=42
):
    """
    Train a Bayesian model with given hyperparameters.
    
    Args:
        model_class: The model class to be trained (e.g., BayesianResNet).
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        lr: Learning rate.
        kl_weight: Weight for KL divergence loss.
        num_epochs: Number of training epochs.
        batch_size: Batch size for the training and validation.
        use_pretrained: Flag to load pre-trained weights.
        checkpoint_path: Path to model checkpoint (if any).
        device: The device to run the model on ('cuda' or 'cpu').
        save_dir: Directory to save model checkpoints, logs, and plots.

    Returns:
        None: The function trains the model, logs progress, and saves the best configuration.
    """
    # Set seed for reproducibility
    set_random_seed(random_seed)

    # Ensure directories exist
    plots_path = os.path.join(save_dir, "plots")
    log_path = os.path.join(save_dir, "logs")
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model = model_class(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses, val_losses = [], []
    
    if use_pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch
        print(f"Loading pre-trained weights and optimizer state from {checkpoint_path}")
        log_message(f"Loaded pre-trained weights and optimizer state from {checkpoint_path}", log_path=log_path)
    else:
        print('Training model from scratch...')
        best_val_loss = float("inf")
        best_epoch = -1
        start_epoch = 0

    # Update total number of epochs if pretrained model loaded
    # E.g. model loaded at epoch 5 with num_epochs 10 will train for 15 epochs in total
    total_epochs = start_epoch + num_epochs

    # Training and validation loop
    for epoch in range(start_epoch, total_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs} - Training", unit="batch"
        ):
            images, labels = images.to(device), labels.to(device).long()

            # Forward pass
            outputs = model(images)
            ce_loss = nn.CrossEntropyLoss()(outputs, labels)
            kl_div = model.kl_divergence()
            loss = ce_loss + kl_weight * kl_div

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(
                valid_loader, desc=f"Epoch {epoch+1}/{total_epochs} - Validation", unit="batch"
            ):
                images, labels = images.to(device), labels.to(device).long()

                # Forward pass
                outputs = model(images)
                ce_loss = nn.CrossEntropyLoss()(outputs, labels)
                kl_div = model.kl_divergence()
                loss = ce_loss + kl_weight * kl_div
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(valid_loader)

        # Logging losses and params
        print(f"Epoch [{epoch+1}/{total_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        log_message(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}", log_path=log_path)

        # Keep tracking losses for each epoch
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Scheduler step after validation in this epoch
        scheduler.step(epoch_val_loss)

        # Logging learning rate after scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate after epoch {epoch+1}: {current_lr:.6f}")
        log_message(f"Epoch {epoch+1}: Learning rate = {current_lr:.6f}", log_path=log_path)

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_train_loss = epoch_train_loss  # Save the training loss for this epoch
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            checkpoint_save_path = os.path.join(save_dir, f"best_model_lr_{lr}_batch_{batch_size}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr": lr,
                "batch_size": batch_size,
                "epoch": best_epoch,
                "train_loss": best_train_loss,
                "val_loss": best_val_loss
            }, checkpoint_save_path)
            print(f"Checkpoint saved: {checkpoint_save_path}")

    
    # Save configuration metrics
    config_metrics = {
        "lr": lr,
        "batch_size": batch_size,
        "kl_weight": kl_weight,
        "best_epoch": best_epoch,
        "train_loss": best_train_loss, # Training loss in epoch with best val loss
        "best_val_loss": best_val_loss,
        "random_seed": random_seed,
        "checkpoint_path": checkpoint_save_path
    }

    # Append to central results log
    results_log = os.path.join(save_dir, "results.json")
    if os.path.exists(results_log):
        with open(results_log, "r") as f:
            results = json.load(f)
    else:
        results = []
    results.append(config_metrics)
    with open(results_log, "w") as f:
        json.dump(results, f, indent=4)
            
    # Save loss plots
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(total_epochs-start_epoch), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LR={lr}, Batch size = {batch_size}")
    plt.legend()
    plt.savefig(f"plots/loss_plot_LR_{lr}_batch_{batch_size}.png")
    plt.close()