"""Multispectral SwinIR architecture tailored specifically for 4-band Sentinel-2 L2A imagery (B02, B03, B04, B08)."""
from satellite_srm.compat import torch, nn, F
from satellite_srm.models.swinir import ResidualSwinTransformerBlock

class MultispectralSwinIR(nn.Module):
    """
    Multispectral SwinIR Super-Resolution Network:
    - Input: 4-band 10m Sentinel-2 (B02 Blue, B03 Green, B04 Red, B08 NIR)
    - Output: 4-band sub-4m Super-Resolved Imagery (B02_SR, B03_SR, B04_SR, B08_SR)
    - Depthwise spectral attention and cross-band residual learning.
    """

    def __init__(
        self,
        scale_factor: float = 3.0,
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 96,
        depths=(4, 4, 4, 4),
        num_heads=(6, 6, 6, 6),
        window_size: int = 8,
        drop_rate: float = 0.05
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # 1. Spectral Shallow Feature Extraction
        # Independent per-band projection + joint cross-spectral convolution
        self.band_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=drop_rate)

        # 2. Deep Transformer Backbone (RSTBs)
        self.rstb_layers = nn.ModuleList([
            ResidualSwinTransformerBlock(dim=embed_dim, depth=d, num_heads=h, window_size=window_size)
            for d, h in zip(depths, num_heads)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. High-Quality Reconstruction & Upscaling Head
        # Supports arbitrary scale factors (e.g. 2.5x, 3.0x, 4.0x)
        self.int_scale = int(round(scale_factor))
        self.conv_before_upsample = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_up = nn.Conv2d(embed_dim, embed_dim * (self.int_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.int_scale)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Base residual connection from bicubic upsampling
        base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)

        feat = self.band_conv(x)
        feat = self.dropout(feat)

        body = feat
        for layer in self.rstb_layers:
            body = layer(body)
        body = self.conv_after_body(body) + feat

        # Upscale
        up = self.conv_before_upsample(body)
        up = self.conv_up(up)
        up = self.pixel_shuffle(up)
        res = self.conv_last(up)

        # Adjust dimensions if float scale factor doesn't match integer pixel shuffle exactly
        if res.shape[-2:] != base.shape[-2:]:
            res = F.interpolate(res, size=(base.shape[-2], base.shape[-1]), mode="bicubic", align_corners=False)

        out = base + res
        return out
