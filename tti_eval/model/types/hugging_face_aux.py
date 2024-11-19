import torch.nn as nn
from timm.layers.helpers import to_2tuple

class ConvStem(nn.Module):
    """Custom Patch Embed Layer.

    Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py#L6-L44
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()

        # Check input constraints
        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Create stem network
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        # Apply normalization layer (if provided)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Check input image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x