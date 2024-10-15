import open_clip

from tti_eval.model.types.visual_only_hugging_face import VisualHFModel
from transformers import AutoConfig

class UNIModel(VisualHFModel):
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

    def _setup(self, **kwargs) -> None:
        config = AutoConfig.from_pretrained('vit')
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name=self.title_in_source,
            pretrained=self.pretrained,
            cache_dir=self._cache_dir.as_posix(),
            device=self.device,
            **kwargs,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name=self.title_in_source)


