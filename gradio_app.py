#!/usr/bin/env python
"""
TransNormal Gradio Web Interface

Launch with: python gradio_app.py
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transnormal import TransNormalPipeline, create_dino_encoder, normal_to_rgb

# ============== 配置路径 ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Model paths (can be overridden via environment variables)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(WEIGHTS_DIR, "transnormal"))
DINO_WEIGHTS_PATH = os.environ.get("DINO_WEIGHTS_PATH", os.path.join(WEIGHTS_DIR, "dinov3_vith16plus"))
# Projector is included in the transnormal model folder
PROJECTOR_PATH = os.environ.get("PROJECTOR_PATH", os.path.join(WEIGHTS_DIR, "transnormal", "cross_attention_projector.pt"))
# =====================================

# Global pipeline (loaded once)
pipe = None


def load_pipeline():
    """Load the TransNormal pipeline."""
    global pipe
    
    if pipe is not None:
        return pipe
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print("[TransNormal] Loading model components...")
    print(f"  - Model path: {MODEL_PATH}")
    print(f"  - DINO path: {DINO_WEIGHTS_PATH}")
    print(f"  - Projector path: {PROJECTOR_PATH}")
    
    # Load DINO encoder
    dino_encoder = create_dino_encoder(
        model_name="dinov3_vith16plus",
        cross_attention_dim=1024,
        weights_path=DINO_WEIGHTS_PATH,
        projector_path=PROJECTOR_PATH,
        device=device,
        dtype=dtype,
        freeze_encoder=True,
    )
    
    # Load pipeline from pretrained
    pipe = TransNormalPipeline.from_pretrained(
        MODEL_PATH,
        dino_encoder=dino_encoder,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    
    print("[TransNormal] Model loaded successfully!")
    return pipe


def predict_normal(image: Image.Image, processing_res: int = 768) -> Image.Image:
    """
    Predict surface normal from input image.
    
    Args:
        image: Input RGB image
        processing_res: Processing resolution
    
    Returns:
        Normal map as PIL Image
    """
    if image is None:
        return None
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Run inference
    with torch.no_grad():
        normal_map = pipeline(
            image=image,
            processing_res=processing_res,
            output_type="pil",
        )
    
    return normal_map


def create_demo():
    """Create Gradio demo interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    }
    h1 {
        font-weight: 600 !important;
    }
    .markdown-text {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    """
    
    with gr.Blocks(
        title="TransNormal",
        theme=gr.themes.Base(),
        css=custom_css,
    ) as demo:
        
        gr.Markdown(
            """
            # TransNormal
            ### Surface Normal Estimation for Transparent Objects
            
            Upload an image to estimate surface normals. Particularly effective for **transparent objects** like glass and plastic.
            
            **Normal Convention:** Red=X (Left) | Green=Y (Up) | Blue=Z (Out)
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400,
                )
                
                processing_res = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64,
                    label="Processing Resolution",
                )
                
                submit_btn = gr.Button("Estimate Normal", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="Normal Map",
                    type="pil",
                    height=400,
                )
        
        # Examples
        example_dir = os.path.join(os.path.dirname(__file__), "examples", "input")
        if os.path.exists(example_dir):
            example_images = [
                os.path.join(example_dir, f) 
                for f in os.listdir(example_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            if example_images:
                gr.Examples(
                    examples=[[img, 768] for img in example_images[:4]],
                    inputs=[input_image, processing_res],
                    outputs=output_image,
                    fn=predict_normal,
                    cache_examples=False,
                )
        
        # Event handlers
        submit_btn.click(
            fn=predict_normal,
            inputs=[input_image, processing_res],
            outputs=output_image,
        )
        
        gr.Markdown(
            """
            ---
            **Paper:** [TransNormal: Dense Visual Semantics for Diffusion-based Transparent Object Normal Estimation](https://longxiang-ai.github.io/TransNormal/)
            
            **Authors:** Mingwei Li, Hehe Fan, Yi Yang (Zhejiang University)
            """
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TransNormal Gradio Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    # Pre-load model
    print("[TransNormal] Pre-loading model...")
    load_pipeline()
    
    # Launch demo
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
