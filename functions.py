import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

from PIL import Image

def create_resnet50_model(num_class, freeze=False):
  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

  if freeze:
    # Freeze all layers
    for param in model.parameters():
      param.requires_grad = False

    # Replace the Fully Connected layers
    model.fc = nn.Linear(model.fc.in_features, num_class)

    # Unfreeze the last residual block (layer4) and the new fc layer
    for name, param in model.named_parameters():
      if "layer4" in name or "fc" in name:
        param.requires_grad = True

  else:
    # No freezing, all layers are trainable
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_class)
        )

  return model


def make_prediction(model, image):
  # Return (class, confidence)
  class_name = {
    0: "Normal",
    1: "Pneumonia"
  }
  softmax = nn.Softmax(dim=1)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  transformed_image = transform(image).unsqueeze(0) # Add 1 more dimension (batch, c, w, h)

  with torch.no_grad():
    outputs = model(transformed_image)
    outputs = softmax(outputs)
    confidence, prediction = torch.max(outputs, dim=1)
    return {"class": class_name[int(prediction.numpy()[0])],
            "prob": float(confidence.numpy()[0])}

'''
model_path = "models/best_resnet50_model_updated.pth"
model = create_resnet50_model(num_class=2, freeze=False)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

image_path = "images/normal_4.jpeg"
image = Image.open(image_path).convert("RGB")

print(make_prediction(model, image))
'''