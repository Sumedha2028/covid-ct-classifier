
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "mobilenet_v3_large.pth"

DEVICE = torch.device("cpu")


class Classifier(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=(1280,),
        drop_p=0.2,
        use_log_softmax=True,
    ):
        super().__init__()

        layers = []
        previous = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_p))
            previous = hidden_size

        layers.append(nn.Linear(previous, output_size))

        self.net = nn.Sequential(*layers)
        self.use_log_softmax = use_log_softmax

    def forward(self, x):
        x = self.net(x)
        if self.use_log_softmax:
            return F.log_softmax(x, dim=1)
        return x


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["arch"] != "mobilenet_v3_large":
        raise ValueError("Unexpected model architecture")

    model = models.mobilenet_v3_large(weights=None)
    model.classifier = Classifier(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
        hidden_layers=checkpoint["hidden_layers"],
        drop_p=checkpoint["drop_p"],
    )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    idx_to_name = {
        index: name
        for name, index in checkpoint["class_to_idx"].items()
    }

    return model, idx_to_name


PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image: Image.Image, model, idx_to_name):
    image = image.convert("RGB")
    input_tensor = PREPROCESS(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.exp(output)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "predicted_class": idx_to_name[predicted_index],
        "probabilities": {
            idx_to_name[index]: round(
                probabilities[index].item() * 100, 2
            )
            for index in range(len(probabilities))
        },
    }
