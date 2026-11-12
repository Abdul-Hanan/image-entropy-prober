# Image Entropy Prober

Calculates and displays Shannon entropy for RGB channels of an image.

This project generates a synthetic color image, computes the Shannon entropy for
its Red, Green, and Blue color channels independently, and then saves an
output image that includes the original image alongside the calculated entropy
values. This is designed to run in headless environments without requiring
a display or user interaction.

## Requirements

- Python 3.7+
- `numpy`
- `opencv-python`

## How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/image-entropy-prober.git
    cd image-entropy-prober
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the main script:
    ```bash
    python main.py
    ```

Upon successful execution, a file named `entropy_analysis_output.png` will be
created in the same directory. This file contains the synthetic input image
at the top and the calculated entropy values for each RGB channel displayed
as text below it.