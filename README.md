# Parallax Simulator RGB–T

This project simulates the **parallax effect** between paired **RGB and thermal images** that are originally registered.  
Through an interactive Gradio interface, users can adjust parameters to deform one of the images and reproduce the differences that typically occur when using physical stereo-mounted cameras.

---

## How to Use

1. Create and activate a virtual environment (example with `venv`):

    ```bash

    macOS
    python3.10 -m venv .venv-parallax
    source .venv-parallax/bin/activate
    ```

    ```bash
    Windows
    python -m venv .venv-parallax
    .\.venv-parallax\Scripts\Activate
    ```

2. Install dependencies:

    ```bash
    pip install "gradio==4.44.1" "gradio-client==1.3.0" "fastapi==0.115.0" "starlette==0.38.5" "pydantic==2.9.2" "uvicorn==0.30.6" opencv-python numpy tqdm uvicorn
    ```

3. Run the script:

    ```bash
    python simulate_parallax.py
    ```

4. A local or public link will open in your browser with the interactive interface.


## Parameters

- **dx / dy (px):** Global displacement (baseline) in pixels. Simulates the positional offset between cameras.

- **Perspective (0–1):** Small corner deformations to simulate different viewpoints. Higher values generate stronger distortions.

- **Rotation (°):** Small pitch, yaw, or roll misalignments.

- **Scale & Blur:** Simulates resolution differences between sensors:

- **Scale:** downscale / upscale.

- **Blur:** additional Gaussian blur.

- **Apply to:** Select which image to transform:

    - thermal (default)
    - rgb
    - both

- **Relative:** Interprets dx/dy as a fraction of the image width/height instead of absolute pixels.

- **Seed:** Fixes the randomness of the perspective transformation for reproducibility.

## Batch Processing
By clicking Process all:

- Applies the selected parameters to all image pairs in the dataset.

- Saves the results in:

    - ./rgb_out

    - ./thermal_out

- Optionally generates a _parallax_params.json file with the parameters used for each pair.

## Notes
- The project expects two input folders:

    - ./rgb

    - ./thermal
containing images with the same base filename (for example, img001.png in both folders).

- Output images are saved in .png format by default.
