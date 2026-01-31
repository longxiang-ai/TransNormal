# Model Weights

This directory should contain the model weights for TransNormal inference.

## Required Weights

You need to download the following model weights:

### 1. TransNormal UNet Weights

The fine-tuned UNet model for surface normal estimation.

**Download from HuggingFace:**
```bash
# Option 1: Using git lfs
git lfs install
git clone https://huggingface.co/Longxiang-ai/TransNormal ./weights/transnormal

# Option 2: Using huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Longxiang-ai/TransNormal', local_dir='./weights/transnormal')"
```

### 2. DINOv3 Pretrained Weights

The DINOv3 ViT-H+/16 distilled encoder for semantic feature extraction.

> **⚠️ Important:** DINOv3 weights require access approval from Meta AI.

**Step 1: Request Access**
1. Visit [Meta AI DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
2. Fill in the access request form
3. Once approved, you will receive an email with download URLs for all model weights

**Step 2: Download ViT-H+/16 distilled**
```bash
# After receiving the download URL via email, use wget to download
wget <YOUR_DINOV3_VITH16PLUS_URL> -O dinov3_vith16plus.pth

# Or download from HuggingFace (if available)
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov3-vith16plus-pretrain-lvd1689m', local_dir='./dinov3_vith16plus')"
```

**Alternative: Using HuggingFace Transformers**

DINOv3 is also available via HuggingFace Transformers (version >= 4.56.0):
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
```

For more details, see the official [DINOv3 GitHub Repository](https://github.com/facebookresearch/dinov3)


## Directory Structure

After downloading, your weights directory should look like:

```
weights/
├── README.md                          # This file
├── transnormal/                       # TransNormal model
│   ├── model_index.json
│   ├── cross_attention_projector.pt   # Projector weights (included)
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── vae/
└── dinov3_vith16plus/                 # DINOv3 encoder (download separately)
    ├── config.json
    └── model.safetensors
```

## Verify Installation

After downloading, verify your installation:

```python
import os

weights_dir = "./weights"
required = [
    "transnormal/unet/diffusion_pytorch_model.safetensors",
    "transnormal/cross_attention_projector.pt",
    "dinov3_vith16plus/model.safetensors",
]

for path in required:
    full_path = os.path.join(weights_dir, path)
    if os.path.exists(full_path):
        print(f"✓ Found: {path}")
    else:
        print(f"✗ Missing: {path}")
```

## Notes

- **Storage:** ~8GB total
- **GPU Memory:** ~11GB VRAM (Peak), ~7.5GB for model loading
- **Precision:** BF16 recommended (avoid NaN issues with DINOv3)
