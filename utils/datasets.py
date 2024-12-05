from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filepath' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx]['encoded_label'] # 'encoded_label' column
        img_path = self.dataframe.iloc[idx]['filepath']  
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format for compatibility with models (instead of modifying first layer)

        if self.transform:
            # Apply transformations if any
            image = self.transform(image)
        
        return image, label


def calculate_mean_std(dataset):
    """
    Function to calculate the mean and std of the dataset.
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    # Iterate through the dataset
    for image, _ in dataset:
        # Convert image to tensor
        image = transforms.ToTensor()(image)

        # Compute mean and std over each image
        mean += image.mean([1, 2])  # Mean over height and width (spatial dimensions)
        std += image.std([1, 2])    # Std over height and width (spatial dimensions)
        total_images += 1

    # Divide by total number of images to get the mean and std for each channel
    mean /= total_images
    std /= total_images

    return mean, std


def get_standard_transform(mean=None, std=None):
    """
    Standard transformation pipeline for training data.
    This will not include any OOD-specific augmentations.
    If mean and std are not provided, use ImageNet statistics for normalization.
    """
    if mean is None or std is None:
        # Default to ImageNet normalization if mean and std are not provided
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((200, 200)),  # Resize images 
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize using ImageNet stats as efficientnet was trained on this data (and a pretrained model will be used to start with)
    ])


def get_ood_transform(mean=None, std=None):
    """
    Transformation pipeline specifically for out-of-distribution (OOD) data.
    This may include augmentations such as random rotations, cropping, or other variations.
    If mean and std are not provided, use ImageNet statistics for normalization.
    """
    if mean is None or std is None:
        # Default to ImageNet normalization if mean and std are not provided
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.RandomRotation(45), # Randomly rotate between -45 and 45 degrees
        transforms.RandomResizedCrop(size=(200, 200), scale=(0.8, 1.2)), # Randomly crop 80-120% of image area
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), # Random color augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
    ])

