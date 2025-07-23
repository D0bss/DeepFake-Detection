import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # Real or Fake

def create_model():
    """
    Creates the VGG16 model with the specified modifications for classification.
    
    Returns:
    - model: The VGG16 model.
    """
    model = models.vgg16(weights='DEFAULT')

    # Freeze layers (optional for fine-tuning)
    for param in model.features[:-15].parameters():
        param.requires_grad = False

    # Modify classifier
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, NUM_CLASSES)
    )

    return model.to(DEVICE)

def load_model(model_path):
    """
    Load the trained model weights from a .pth file.
    
    Args:
    - model_path: Path to the model weights file (.pth)
    
    Returns:
    - model: The loaded model
    """
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set to evaluation mode
    return model
