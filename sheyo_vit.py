import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and embeds them.
    Input: (B, C, H, W) -> Output: (B, N_Patches, Embed_Dim)
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # We use a Conv2d layer to do the patching and linear projection in one step
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class VisionTransformer(nn.Module):
    """
    Sheyo OS Vision Core v2.1 (RGB Enabled)
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=64, depth=3, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding (Defaults to 3 channels for RGB)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 2. Learnable Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        # 4. Transformer Encoder Layers (Pre-Norm for RL stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 5. Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Input: Batch of Images (B, 3, 64, 64)
        Output: Latent Feature Vector (B, Embed_Dim)
        """
        B = x.shape[0]
        
        # 1. Create Patches
        x = self.patch_embed(x) # (B, N, E)
        
        # 2. Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, E)
        
        # 3. Add Position Embedding
        x = x + self.pos_embed
        
        # 4. Transformer Magic
        x = self.encoder(x)
        
        # 5. Extract only the CLS token output
        cls_out = x[:, 0] 
        cls_out = self.norm(cls_out)
        
        return cls_out 

if __name__ == "__main__":
    # Quick Test for RGB
    try:
        model = VisionTransformer(in_channels=3)
        dummy_img = torch.randn(1, 3, 64, 64) # Batch 1, 3 Channels (RGB), 64x64
        output = model(dummy_img)
        print(f"ViT v2.1 Output Shape: {output.shape}") 
        print("Visual Core v2.1 (RGB) Online: SUCCESS")
    except Exception as e:
        print(f"Visual Core FAILED: {e}")