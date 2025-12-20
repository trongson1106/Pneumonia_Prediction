import torch
import torch.nn as nn
import torchvision.transforms as transforms
from functions import create_resnet50_model, make_prediction
from huggingface_hub import hf_hub_download
import os
import io

from PIL import Image

from flask import Flask, request, jsonify, render_template

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

model_path = "models/best_resnet50_model_updated.pth"
if os.path.exists(model_path):
    print("Model is already downloaded!")

else:
    print("Downloading model")

    # noinspection PyArgumentList
    model_path = hf_hub_download(
        repo_id="sonh1106/pneumonia_resnet50",
        filename="best_resnet50_model_updated.pth",
        local_dir="models",
    )

    print(f"Model is downloaded at {model_path}")  # full path to saved model

model = create_resnet50_model(num_class=2, freeze=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"})

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = make_prediction(model, img) # return {"class": ..., "prob":..., "heatmap":...}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)


