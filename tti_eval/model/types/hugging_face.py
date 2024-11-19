from collections.abc import Callable
from typing import Any

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel as HF_AutoModel
from transformers import AutoProcessor as HF_AutoProcessor
from transformers import AutoTokenizer as HF_AutoTokenizer

from tti_eval.common import ClassArray, EmbeddingArray
from tti_eval.dataset import Dataset
from tti_eval.model import Model
from tti_eval.model.types.hugging_face_aux import ConvStem


class HFModel(Model):
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

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def get_collate_fn(self) -> Callable[[Any], Any]:
        def collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(example["label"])

            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return {"pixel_values": pixel_values, "labels": labels}

        return collate_fn

    def _setup(self, **kwargs) -> None:
        self.model = HF_AutoModel.from_pretrained(self.title_in_source, cache_dir=self._cache_dir).to(self.device)
        load_result = HF_AutoProcessor.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)
        self.processor = load_result[0] if isinstance(load_result, tuple) else load_result
        self.tokenizer = HF_AutoTokenizer.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        all_image_embeddings = []
        all_labels = []
        with torch.inference_mode():
            _dataset: Dataset = dataloader.dataset
            inputs = self.tokenizer(_dataset.text_queries, padding=True, return_tensors="pt").to(self.device)
            class_features = self.model.get_text_features(**inputs)
            normalized_class_features = class_features / class_features.norm(p=2, dim=-1, keepdim=True)
            class_embeddings = normalized_class_features.numpy(force=True)
            for batch in tqdm(
                dataloader,
                desc=f"Embedding ({_dataset.split}) {_dataset.title} dataset with {self.title}",
            ):
                image_features = self.model.get_image_features(pixel_values=batch["pixel_values"].to(self.device))
                normalized_image_features = (image_features / image_features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                all_image_embeddings.append(normalized_image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, class_embeddings, labels

class VisualHFModel(HFModel):
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
        self.model = HF_AutoModel.from_pretrained(self.title_in_source, cache_dir=self._cache_dir).to(self.device)
        load_result = HF_AutoProcessor.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)
        self.processor = load_result[0] if isinstance(load_result, tuple) else load_result
        #self.tokenizer = HF_AutoTokenizer.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        all_image_embeddings = []
        all_labels = []
        with torch.inference_mode():
            _dataset: Dataset = dataloader.dataset
            #inputs = self.tokenizer(_dataset.text_queries, padding=True, return_tensors="pt").to(self.device)
            #class_features = self.model.get_text_features(**inputs)
            #normalized_class_features = class_features / class_features.norm(p=2, dim=-1, keepdim=True)
            #class_embeddings = normalized_class_features.numpy(force=True)
            for batch in tqdm(
                dataloader,
                desc=f"Embedding ({_dataset.split}) {_dataset.title} dataset with {self.title}",
            ):
                image_features = self.model(pixel_values=batch["pixel_values"].to(self.device)).pooler_output.squeeze()
                normalized_image_features = (image_features / image_features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                all_image_embeddings.append(normalized_image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, image_embeddings, labels

class GigaPathModel(VisualHFModel):
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
        # load model from the hub
        self.model = timm.create_model(
        model_name=self.title_in_source,
        pretrained=True,
        )
        self.transform = transforms.Compose(
            [ transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.transform(i).to(self.device).unsqueeze(0) for i in images
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
                image_features = [self.model(image.to(self.device)).squeeze()
                                  for image in batch["pixel_values"].unbind()]
                image_features = torch.stack(image_features)
                normalized_image_features = (image_features / image_features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                all_image_embeddings.append(normalized_image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, image_embeddings, labels

class CTransPathModel(GigaPathModel):
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
        # load model from the hub
        self.model = timm.create_model(
        model_name=self.title_in_source,
        embed_layer=ConvStem, #  defined above
        pretrained=True,
        )
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
