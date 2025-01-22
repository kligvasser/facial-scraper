import torch
import clip
from torchvision import transforms
from itertools import combinations


CLIP_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLIPImageEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        chunk_size=8,
    ):
        super().__init__()
        self.device = device
        self.chunk_size = chunk_size

        self.clip, _ = clip.load(model_name)
        self.clip.eval()
        self.clip = self.clip.to(device)

        for param in self.clip.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((CLIP_SIZE, CLIP_SIZE)),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def forward(self, images):
        processed_images = torch.stack([self.preprocess(img) for img in images])
        processed_images = processed_images.to(self.device)

        with torch.no_grad():
            embeddings = [
                self.clip.encode_image(processed_images[i : i + self.chunk_size])
                for i in range(0, processed_images.size(0), self.chunk_size)
            ]
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def compute_clip_score(self, face_frames, k=16):
        if len(face_frames) < k:
            raise ValueError(f"Not enough frames to sample {k} frames.")

        sampled_frames = torch.utils.data.Subset(face_frames, range(k))
        embeddings = self.forward(sampled_frames)

        distances = [
            1 - torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            for i, j in combinations(range(k), 2)
        ]

        return max(distances).item()
