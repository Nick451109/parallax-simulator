# Parallax Simulator RGB–T

This project simulates the **parallax effect** between paired **RGB and thermal images** that are originally registered.  
Through an interactive Gradio interface, users can adjust parameters to deform one of the images and reproduce the differences that typically occur when using physical stereo-mounted cameras.

---

## How to Use

1. Create and activate a virtual environment (example with `venv`):

    ```bash
    python3 -m venv parallax-env
    source parallax-env/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install opencv-python numpy tqdm gradio fastapi starlette pydantic uvicorn
    ```

3. Run the script:

    ```bash
    python simulate_parallax.py
    ```

4. A local or public link will open in your browser with the interactive interface.


## Parameters

* dx / dy (px): Global displacement (baseline) in pixels. Simulates the positional offset between cameras.

* Perspective (0–1): Small corner deformations to simulate different viewpoints. Higher values generate stronger distortions.

* Rotation (°): Small pitch, yaw, or roll misalignments.

* Scale & Blur: Simulates resolution differences between sensors:

* Scale: downscale / upscale.

* Blur: additional Gaussian blur.

- Apply to: Select which image to transform:

    - thermal (default)

    - rgb

    -  both

* Relative: Interprets dx/dy as a fraction of the image width/height instead of absolute pixels.

* Seed: Fixes the randomness of the perspective transformation for reproducibility.

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
