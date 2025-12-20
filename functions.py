import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

import matplotlib.cm as cm

import numpy as np

from PIL import Image

import base64
from io import BytesIO

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

class GradCAM:
  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None
    self._register_hooks()

  def _register_hooks(self):
    def forward_hook(module, input, output):
      self.activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
      self.gradients = grad_output[0].detach()

    self.target_layer.register_forward_hook(forward_hook)
    self.target_layer.register_full_backward_hook(backward_hook)

  def __call__(self, input_tensor, class_idx=None):
    self.model.zero_grad()
    output = self.model(input_tensor)

    if class_idx is None:
      class_idx = output.argmax(dim=1).item()

    score = output[:, class_idx]
    score.backward() # Computing gradient on logit (result), no the Loss

    #print(self.gradients is None)

    weights = self.gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * self.activations).sum(dim=1)
    cam = F.relu(cam)

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam

def numpy_to_base64(heatmap_array):
  # assume shape = (H, W) or (H, W, 3)
  img = Image.fromarray(heatmap_array.astype("uint8"))

  buffered = BytesIO()
  img.save(buffered, format="PNG")
  img_bytes = buffered.getvalue()

  return base64.b64encode(img_bytes).decode("utf-8")


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

  '''GRAD CAM'''
  target_layer = model.layer4[-1] # Last Convolutional Layer
  grad_cam = GradCAM(model, target_layer)

  #print("Grad exists?", grad_cam.gradients is not None)

  cam = grad_cam(transformed_image)
  cam = cam[0].cpu().numpy()

  '''Transform to Image for resizing than back to Numpy array to overlay it on the original image'''
  cam = Image.fromarray(cam)
  cam = cam.resize((224, 224), resample=Image.BILINEAR)
  cam = np.array(cam)

  heatmap = cm.jet(cam)[..., :3]

  img_vis = transformed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
  img_vis = img_vis.astype(np.float32)

  overlay = heatmap + 0.2 * img_vis
  overlay = np.clip(overlay, 0, 1)

  heatmap_base64 = numpy_to_base64((overlay*255))

  with torch.no_grad():
    outputs = model(transformed_image)
    outputs = softmax(outputs)
    confidence, prediction = torch.max(outputs, dim=1)

    return {"class": class_name[int(prediction.numpy()[0])],
            "prob": float(confidence.numpy()[0]),
            "heatmap": heatmap_base64}










'''
TEST CODE
model_path = "models/best_resnet50_model_updated.pth"
model = create_resnet50_model(num_class=2, freeze=False)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

image_path = "images/normal_4.jpeg"
image = Image.open(image_path).convert("RGB")

print(make_prediction(model, image))
'''