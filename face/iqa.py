import torch
import random

from pyiqa import create_metric
from torchvision import transforms


class IQAQualityAssessor(torch.nn.Module):
    def __init__(
        self,
        metrics=("hyperiqa", "clipiqa+"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device

        self.metrics = {name: create_metric(name).to(device).eval() for name in metrics}
        for metric in self.metrics.values():
            for param in metric.parameters():
                param.requires_grad = False

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ]
        )

    def forward(self, images, k=4):
        if len(images) < k:
            raise ValueError(
                f"Not enough frames to sample {k} frames. Found {len(images)}."
            )

        sampled_images = random.sample(images, k)

        processed_images = torch.stack([self.preprocess(img) for img in sampled_images])
        processed_images = processed_images.to(self.device)
        processed_images = torch.clamp(processed_images, min=0, max=1)

        scores = {}
        with torch.no_grad():
            for name, metric in self.metrics.items():
                metric_scores = metric(processed_images).squeeze()
                scores[name] = metric_scores.mean().item()

        return scores
