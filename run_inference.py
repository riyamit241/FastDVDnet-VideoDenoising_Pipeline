import torch
import cv2
import numpy as np
import os
from collections import OrderedDict
from models import FastDVDnet

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Path to your input video
INPUT_VIDEO = "20251123_001022.mp4" 

# Path where the cleaned video will be saved
OUTPUT_VIDEO = "cleaned_output.mp4"

# Denoising Strength (10-50). 
# 30 is a good balance. Higher = smoother but less detail.
NOISE_SIGMA = 30 

# Set to False if you don't have an NVIDIA GPU
USE_GPU = True 

# Target width for processing (Height is calculated automatically)
# 960 is a good balance between speed and quality.
TARGET_WIDTH = 960
# ==========================================

def load_model(weights_path):
    """
    Loads the FastDVDnet model and handles the 'module.' prefix mismatch
    common with models trained on multi-GPU setups.
    """
    print(f"Loading model weights from {weights_path}...")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: Weights file '{weights_path}' not found!")

    model = FastDVDnet(num_input_frames=5)
    
    # Load weights
    state_temp_dict = torch.load(weights_path, map_location='cuda' if USE_GPU else 'cpu')
    if 'state_dict' in state_temp_dict:
        state_temp_dict = state_temp_dict['state_dict']
    
    # Create new dictionary without "module." prefix
    new_state_dict = OrderedDict()
    for k, v in state_temp_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    if USE_GPU and torch.cuda.is_available():
        model.cuda()
        
    return model

def get_safe_dimensions(width, height, target_w=960):
    """
    Calculates new height to preserve aspect ratio while ensuring
    dimensions are divisible by 16 (required for U-Net architecture).
    """
    aspect_ratio = height / width
    target_h = int(target_w * aspect_ratio)
    
    # Round to nearest multiple of 16
    if target_h % 16 != 0:
        target_h = round(target_h / 16) * 16
        
    return (target_w, target_h)

def process_video():
    # 1. Check Input
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video '{INPUT_VIDEO}' not found.")
        return

    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(INPUT_VIDEO)
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Calculate Safe Dimensions
    resize_dim = get_safe_dimensions(orig_w, orig_h, TARGET_WIDTH)
    print(f"Processing Video: {INPUT_VIDEO}")
    print(f"Original: {orig_w}x{orig_h} | Resizing to: {resize_dim} for inference")

    # 4. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, orig_fps, resize_dim)

    # 5. Load Model
    # Expects 'model_clipped_noise.pth' in the same folder
    model = load_model('model_clipped_noise.pth')
    
    # 6. Create Noise Map
    # Normalized sigma: value / 255.0
    noise_map = torch.FloatTensor([NOISE_SIGMA / 255.0]).repeat(1, 1, resize_dim[1], resize_dim[0])
    if USE_GPU: noise_map = noise_map.cuda()

    print("Starting Denoising Pipeline...")
    
    frames_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break 
        
        # Resize
        frame = cv2.resize(frame, resize_dim)
        
        # Preprocess: BGR -> RGB, Normalize 0-1, Channel First
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        
        # Update Buffer (Sliding Window)
        frames_buffer.append(frame_tensor)
        if len(frames_buffer) > 5: frames_buffer.pop(0)
        
        # Need 5 frames to process the middle one
        if len(frames_buffer) < 5: continue 

        # Inference
        input_stack = torch.stack(frames_buffer)
        # Flatten: [Batch, Channels, H, W] -> [1, 15, H, W]
        input_flat = input_stack.view(1, -1, resize_dim[1], resize_dim[0])
        if USE_GPU: input_flat = input_flat.cuda()

        with torch.no_grad():
            clean_tensor = model(input_flat, noise_map)
            
        # Post-process: Normalize back to 0-255, RGB -> BGR
        clean_arr = clean_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        clean_arr = np.clip(clean_arr * 255.0, 0, 255).astype(np.uint8)
        clean_bgr = cv2.cvtColor(clean_arr, cv2.COLOR_RGB2BGR)
        
        out.write(clean_bgr)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"✨ Processed {frame_count}/{total_frames} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video()