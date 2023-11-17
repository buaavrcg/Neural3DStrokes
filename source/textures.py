import abc
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from source import configs
from source.utils import misc


def get_stroke_texture(config: configs.Config):
    if config.stroke_texture == 'none':
        return None
    if config.stroke_texture == 'image':
        return ImageTexture(config.texture_image_path, config.texture_image_size)
    else:
        assert 0, f'Unknown stroke texture {config.stroke_texture}'


class StrokeTexture(nn.Module):
    @abc.abstractmethod
    def forward(self, texcoords: torch.Tensor, colors: torch.Tensor, alphas: torch.Tensor):
        """
        Query a stroke 3d texture from the given texcoord.
        Args:
            texcoords: UV texture coordinates in [0,1] of shape (..., 2).
            colors: RGB colors of strokes of shape (..., 3).
            alphas: Alpha values of strokes of shape (...,).
        Returns:
            new_colors: textured RGB colors of shape (..., 3).
            new_alphas: textured Alpha values of shape (...,).
        """
        raise NotImplementedError
    

@gin.configurable
class ImageTexture(StrokeTexture):
    modulate_alpha: bool = False  # Whether to modulate alpha with texture
    tint_ratio: float = 1.0  # The amount of original color to tint with
    
    def __init__(self, tint_image_path: str, texture_size: tuple[int, int], **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        image = torch.from_numpy(misc.load_img(tint_image_path))
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        image = (image  / 255.).permute(2, 0, 1).unsqueeze(0)  # [N, C, H, W]
        image = F.interpolate(image, texture_size, mode='area')
        self.register_buffer('image_texture', image, persistent=False)
        
    def sample_texture(self, texcoords: torch.Tensor):
        """Sample texture from the given texcoords.
        Args:
            texcoords: UV texture coordinates in [0,1] of shape (N, 2).
        Returns:
            texvalues: RGB colors of shape (N, C).
        """
        uvs = texcoords.reshape(1, 1, -1, 2) * 2 - 1  # [1, 1, N, 2]
        texvalues = F.grid_sample(self.image_texture, uvs, align_corners=True)  # [1, C, 1, N]
        texvalues = texvalues[0, :, 0, :].permute(1, 0)  # [N, C]
        return texvalues
        
    def forward(self, texcoords: torch.Tensor, colors: torch.Tensor, alphas: torch.Tensor):
        assert colors.shape[-1] == 3, 'Colors should be RGB for ImageTexture'
        pre_shape = colors.shape[:-1]
        texcoords = texcoords.reshape(-1, 2)  # [N, 2]
        colors = colors.reshape(-1, 3)  # [N, 3]
        alphas = alphas.reshape(-1)  # [N]
        
        tint = self.sample_texture(texcoords)  # [N, C]
        assert tint.shape[-1] == 3 or tint.shape[-1] == 1, 'Texture should be RGB or grayscale'
        
        # Interpolate between pure white and texture
        # tint = (1 - alphas.unsqueeze(1)) * tint + alphas.unsqueeze(1)
        
        colors = colors * self.tint_ratio + (1.0 - self.tint_ratio)
        colors = colors * tint
        if self.modulate_alpha:
            alphas = alphas * tint.mean(-1)
        
        return colors.reshape(*pre_shape, 3), alphas.reshape(*pre_shape)
    
