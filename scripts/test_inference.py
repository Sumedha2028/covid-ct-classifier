
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "sample" / "NiCT1.jpg"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "mobilenet_v3_large.pth"

CLASSES = ["NiCT", "nCT", "pCT"]
device = torch.device("cpu")


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
        prev = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_p))
            prev = hidden_size

        layers.append(nn.Linear(prev, output_size))

        self.net = nn.Sequential(*layers)
        self.use_log_softmax = use_log_softmax

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1) if self.use_log_softmax else x


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

if checkpoint["arch"] != "mobilenet_v3_large":
    raise ValueError(f"Unexpected architecture: {checkpoint['arch']}")

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
    index: name for name, index in checkpoint["class_to_idx"].items()
}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

with Image.open(IMAGE_PATH) as image:
    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

with torch.inference_mode():
    output = model(input_tensor)
    probabilities = torch.exp(output)[0]

predicted_index = int(torch.argmax(probabilities).item())

print("Checkpoint loaded successfully.")
print("Image:", IMAGE_PATH.name)
print("Predicted class:", idx_to_name[predicted_index])
print("Class probabilities:")

for index in range(len(probabilities)):
    class_name = idx_to_name[index]
    percentage = probabilities[index].item() * 100
    print(f"  {class_name}: {percentage:.2f}%")
