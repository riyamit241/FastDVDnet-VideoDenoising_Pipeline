# FastDVDnet Video Denoising Inference Pipeline

## Project Overview
This repository implements a production-ready inference pipeline for **FastDVDnet**, a state-of-the-art Deep Learning video denoising algorithm. Unlike methods requiring optical flow, this project uses a sliding-window U-Net architecture to remove Gaussian noise from video feeds in real-time.

## Key Features
* [cite_start]**Sliding Window Buffer:** Implements a 5-frame temporal buffer to leverage spatio-temporal coherence .
* **Smart Resizing:** Automatically calculates aspect-ratio-safe dimensions divisible by 16 for U-Net compatibility.
* **Optimized Inference:** Flattens temporal inputs into 15-channel tensors for efficient batch processing.

## Technical Details
* **Language:** Python 3.x
* **Libraries:** PyTorch, OpenCV, NumPy
* **Model:** FastDVDnet (Cascaded U-Net)

## How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Get the Weights:**
    This project requires the pre-trained model weights. Download `model_clipped_noise.pth` from the [original FastDVDnet repository](https://github.com/m-tassano/fastdvdnet) and place it in the root folder.
3.  **Run Inference:**
    Update the `INPUT_VIDEO` path in `run_inference.py` and execute:
    ```bash
    python run_inference.py
    ```

## References
* *FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation* - Tassano et al. (CVPR 2020) [cite_start][cite: 2-4]
