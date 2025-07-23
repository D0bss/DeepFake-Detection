import torch
from torchvision import transforms

# Define preprocessing transforms (resize, normalize, etc.)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224 for VGG16
    transforms.ToTensor(),              # Convert image to tensor
    transforms.Normalize(              # Normalize using VGG16's mean and std
        mean=[0.485, 0.456, 0.406],     # Mean of ImageNet
        std=[0.229, 0.224, 0.225]       # Std of ImageNet
    )
])

def preprocess_face(face_pil):
    return preprocess(face_pil).unsqueeze(0).to('cuda')  # Add batch dimension and move to DEVICE (GPU/CPU)


