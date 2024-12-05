import torch
import numpy as np

def generate_bayesian_predictions(model, test_loader, device, num_samples=200):
    """
    Generate predictions using a Bayesian model with variational inference.

    This function performs multiple stochastic forward passes through the model to sample from 
    the posterior distribution of the weights. It also computes the KL divergence during each 
    pass (if the model supports it). Predictions for all test samples across multiple passes 
    are stored for downstream analysis.

    Args:
        model (torch.nn.Module): The Bayesian neural network model to evaluate. 
            Must implement a `kl_divergence` method for KL divergence computation.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset, 
            providing batches of images and labels.
        device (torch.device): The device to run the inference on (e.g., 'cuda' or 'cpu').
        num_samples (int): Number of forward passes through the model to perform. 
            Each pass samples from the posterior distribution of the model weights. 
            Default is 5.

    Returns:
        tuple: A tuple containing:
            - all_preds (list of np.ndarray): A list where each element is an array of predictions 
              for all test images from one forward pass. The shape of each array is 
              `(num_images, num_classes)`.
            - all_kl_divs (list of float): A list of KL divergence values for each forward pass. 
              The length of this list is equal to `num_samples`.

    Note:
        - The model should be in evaluation mode (`model.eval()`) before calling this function.
        - Ensure that the `kl_divergence` method of the model is implemented to return a scalar 
          value representing the KL divergence.

    Example:
        >>> model.eval()
        >>> all_preds, all_kl_divs = generate_bayesian_predictions(model, test_loader, device, num_samples=10)
    """

    # To store all predictions for each image
    all_preds = []  # This will hold predictions for all images, for all samples

    # To store KL divergences
    all_kl_divs = []

    with torch.no_grad():  # disable gradient calculation for inference
        for _ in range(num_samples):
            # List to hold model predictions for this forward pass
            batch_preds = []

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                output = model(images)

                # Store predictions for this batch (each batch contains images)
                batch_preds.append(output.cpu().numpy())  # Store predictions for this batch

                # Compute KL divergence for this batch
                kl_div = model.kl_divergence()  # Get KL divergence for the current pass
                all_kl_divs.append(kl_div.item())

            # After processing the batches, append the predictions for this forward pass
            all_preds.append(np.concatenate(batch_preds, axis=0))  # Concatenate batch predictions across the batch dimension and add to all_preds
    
    all_preds = torch.from_numpy(np.stack(all_preds, axis=0)) # logit tensor shape [num_samples x num_images x num_classes]
    
    return all_preds, all_kl_divs