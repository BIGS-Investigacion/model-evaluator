from collections.abc import Callable
from typing import Any

import numpy as np
import open_clip
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from tti_eval.common import ClassArray, EmbeddingArray
from tti_eval.dataset import Dataset
from tti_eval.model.types.hugging_face import HFModel
from tti_eval.model.types.open_clip_model import OpenCLIPModel


class UniModel(HFModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir)
        self._setup(**kwargs)

    def _setup(self, **kwargs) -> None:
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        self.model = timm.create_model(self.title_in_source, pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.transform(i).to(self.device).squeeze() for i in images
            ]
            return batch

        return process_fn

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        all_image_embeddings = []
        all_labels = []
        with torch.inference_mode():
            _dataset: Dataset = dataloader.dataset
            for batch in tqdm(
                dataloader,
                desc=f"Embedding ({_dataset.split}) {_dataset.title} dataset with {self.title}",
            ):
                image_features = self.model(batch["pixel_values"].to(self.device))
                normalized_image_features = (image_features / image_features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                all_image_embeddings.append(normalized_image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, image_embeddings, labels

class CONCHModel(OpenCLIPModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str,
        pretrained: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.pretrained = pretrained
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir, **kwargs)
        self._setup(**kwargs)

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [self.processor(i) for i in images]
            return batch

        return process_fn

    def get_collate_fn(self) -> Callable[[Any], Any]:
        def collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(example["label"])

            torch_images = torch.stack(images)
            labels = torch.tensor(labels)
            return {"image": torch_images, "labels": labels}

        return collate_fn

    def _setup(self, **kwargs) -> None:
        self.model, self.processor = open_clip.create_model_from_pretrained(self.title_in_source)
        self.tokenizer = open_clip.get_tokenizer(model_name=self.title_in_source)

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        all_image_embeddings = []
        all_labels = []
        with torch.inference_mode():
            _dataset: Dataset = dataloader.dataset
            text = self.tokenizer(_dataset.text_queries).to(self.device)
            class_embeddings = self.model.encode_text(text, normalize=True).numpy(force=True)
            for batch in tqdm(
                dataloader,
                desc=f"Embedding ({_dataset.split}) {_dataset.title} dataset with {self.title}",
            ):
                image_features = self.model.encode_image(batch["image"].to(self.device), normalize=True)
                all_image_embeddings.append(image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, class_embeddings, labels
