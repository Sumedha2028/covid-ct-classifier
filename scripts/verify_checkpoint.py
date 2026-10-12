
from pathlib import Path

import torch
from torch import nn
from torchvision import models


CHECKPOINT_PATH = Path("outputs/checkpoints/mobilenet_v3_large.pth")


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()

        layers = []
        prev = input_size

        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=drop_p))
            prev = hidden

        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.net(x), dim=1)


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    print("PyTorch:", torch.__version__)
    print("Loading checkpoint:", CHECKPOINT_PATH.resolve())

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("arch") != "mobilenet_v3_large":
        raise ValueError(
            f"Unexpected architecture: {checkpoint.get('arch')}"
        )

    model = models.mobilenet_v3_large(weights=None)

    model.classifier = Classifier(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
        hidden_layers=checkpoint["hidden_layers"],
        drop_p=checkpoint["drop_p"],
    )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classes = checkpoint["classes"]
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Architecture:", checkpoint["arch"])
    print("Number of outputs:", checkpoint["output_size"])
    print("Class mapping:", model.class_to_idx)
    print("Saved classes:", model.classes)
    print("Evaluation mode:", not model.training)
    print("\nCheckpoint verification PASSED.")


if __name__ == "__main__":
    main()
