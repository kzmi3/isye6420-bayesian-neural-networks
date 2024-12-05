import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import random
import os


def get_normalization_params(transform):
    """
    Extract mean and std from the transform, if they exist. Defaults to ImageNet if not found.
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Default ImageNet values

    if transform:
        # Check if the transform includes Normalize
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                mean = np.array(t.mean)
                std = np.array(t.std)
                break
    
    return mean, std


def imshow(ax, img, transform=None):
    """Utility function to unnormalize and display a tensor image."""
    # Extract mean and std from the transform if available
    mean, std = get_normalization_params(transform)
    
    # Reverse normalization
    img = img.numpy()  # Convert tensor to NumPy array
    img = np.transpose(img, (1, 2, 0))  # Convert from Tensor (C x H x W) to (H x W x C) expected in matplotlib

    # Handle image normalization depending on channels (grayscale or RGB)
    if img.shape[2] == 1:  # Grayscale image
        img = img.squeeze(axis=-1)  # Remove the single channel dimension
    
    img = img * std + mean  # Unnormalize each channel
    img = np.clip(img, 0, 1)  # Clip values to [0, 1] for valid display
    
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.axis('off')


def plot_example_images(df, transform=None, seed=None, dpi=150):
    """
    Plot a single example image from each unique class in the dataframe.

    Parameters:
    df (pandas.DataFrame): Dataframe containing image paths and labels
    transform (torchvision.transforms.Compose): Custom transform to apply on images (default None)
    seed (int): Seed for random selection of images (default None)
    """
    
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)  # For random selection
        np.random.seed(seed)  # For numpy random selection
        torch.manual_seed(seed)  # For PyTorch operations

    # Extract unique classes from the dataframe
    unique_classes = df['encoded_label'].unique()

    # Set up the figure for plotting
    fig, axes = plt.subplots(len(unique_classes), 1, figsize=(5, 5 * len(unique_classes)), dpi=dpi)
    
    # Ensure axes is always an array, even if there's only one class
    if len(unique_classes) == 1:
        axes = [axes]

    # Loop through each unique class and plot one example image
    for idx, class_id in enumerate(unique_classes):
       
        # Filter rows for the current class
        class_df = df[df['encoded_label'] == class_id]
        
        # Randomly sample one image from the current class
        selected_row = class_df.sample(n=1, random_state=seed).iloc[0]
        image_path = selected_row['filepath']
        
        img = Image.open(image_path)  # Open the image
        
        # Convert image to RGB if it is grayscale or other modes
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply the optional transform if provided
        if transform:
            img = transform(img)
        
        # Ensure the image is a tensor (if it's a PIL Image, it needs conversion)
        if isinstance(img, Image.Image):
            img_tensor = transforms.ToTensor()(img)  # Convert to tensor
        else:
            img_tensor = img  # It's already a tensor if previously transformed
        
        # Get the image filename without extension for the title
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Plot using the imshow function
        ax = axes[idx]  # 1D array indexing
        
        imshow(ax, img_tensor, transform=transform)
        ax.set_title(f'{img_name}')  # Add image name without extension to title
    
    plt.tight_layout()
    plt.show()


