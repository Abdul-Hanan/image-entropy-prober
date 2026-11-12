import numpy as np
import cv2

def generate_synthetic_image(width=400, height=300):
    """
    Generates a synthetic color image with distinct regions
    to demonstrate varying channel entropies.
    Image is divided into four quadrants:
    - Top-left: Solid Red
    - Top-right: Solid Green
    - Bottom-left: Solid Blue
    - Bottom-right: Grayscale random noise (high entropy for all channels)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    h_half, w_half = height // 2, width // 2

    # Top-left: Solid Red (BGR: 0,0,255)
    img[0:h_half, 0:w_half] = [0, 0, 255]

    # Top-right: Solid Green (BGR: 0,255,0)
    img[0:h_half, w_half:width] = [0, 255, 0]

    # Bottom-left: Solid Blue (BGR: 255,0,0)
    img[h_half:height, 0:w_half] = [255, 0, 0]

    # Bottom-right: Grayscale random noise
    # Generate a single channel of noise and assign it to all BGR channels
    # for a grayscale noise effect. This ensures all three channels in this
    # quadrant contribute to high entropy.
    noise_block = np.random.randint(0, 256, size=(h_half, w_half), dtype=np.uint8)
    img[h_half:height, w_half:width, 0] = noise_block # Blue channel
    img[h_half:height, w_half:width, 1] = noise_block # Green channel
    img[h_half:height, w_half:width, 2] = noise_block # Red channel

    return img

def calculate_channel_entropy(channel_data):
    """
    Calculates the Shannon entropy for a single grayscale channel.
    Args:
        channel_data (numpy.ndarray): A 2D numpy array representing a single
                                      color channel (e.g., Red, Green, or Blue).
                                      Pixel values are expected to be 0-255.
    Returns:
        float: The Shannon entropy of the channel.
    """
    # Ensure channel data is 8-bit unsigned integer
    channel_data = channel_data.astype(np.uint8)

    # Calculate histogram
    # cv2.calcHist expects `images` as a list, and `channels` as a list of channel indices.
    # For a single channel image (like one of the R, G, B planes), images=[channel_data]
    # and channels=[0] (as it's the first and only channel in this context).
    hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])

    # Normalize histogram to get probabilities
    total_pixels = channel_data.size
    if total_pixels == 0:
        return 0.0 # Handle empty channel case

    hist_normalized = hist.ravel() / total_pixels

    # Remove zero probabilities to avoid log(0) which is undefined
    probabilities = hist_normalized[hist_normalized > 0]

    # Calculate Shannon entropy: H = -sum(p * log2(p))
    # Using base 2 logarithm for Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def main():
    print("Starting Image Entropy Prober...")

    # 1. Generate a synthetic test image
    image = generate_synthetic_image()
    print(f"Generated synthetic image with shape: {image.shape}")

    # 2. Split the image into B, G, R channels
    # OpenCV uses BGR order by default when reading/splitting color images
    b_channel, g_channel, r_channel = cv2.split(image)

    # 3. Calculate Shannon entropy for each channel
    r_entropy = calculate_channel_entropy(r_channel)
    g_entropy = calculate_channel_entropy(g_channel)
    b_entropy = calculate_channel_entropy(b_channel)

    print(f"\nCalculated Channel Entropies:")
    print(f"  Red Channel Entropy:   {r_entropy:.2f}")
    print(f"  Green Channel Entropy: {g_entropy:.2f}")
    print(f"  Blue Channel Entropy:  {b_entropy:.2f}")

    # 4. Prepare output image to display results visually
    # Constants for text rendering
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)  # White text in BGR format
    BG_COLOR = (0, 0, 0)          # Black background for the text area in BGR
    LINE_HEIGHT = 25              # Pixels per line of text
    TEXT_PADDING_TOP = 20         # Padding between image bottom and first text line

    text_lines = [
        "Channel Entropies:",
        f"  Red:   {r_entropy:.2f}",
        f"  Green: {g_entropy:.2f}",
        f"  Blue:  {b_entropy:.2f}"
    ]

    img_height, img_width, _ = image.shape
    text_area_height = len(text_lines) * LINE_HEIGHT + TEXT_PADDING_TOP

    # Create a new image canvas large enough for the original image + text
    output_img_height = img_height + text_area_height
    output_img = np.zeros((output_img_height, img_width, 3), dtype=np.uint8)
    output_img.fill(BG_COLOR[0]) # Fill the entire canvas with the background color

    # Place the original image at the top of the new canvas
    output_img[0:img_height, 0:img_width] = image

    # Add entropy text below the image
    current_y = img_height + TEXT_PADDING_TOP
    for line in text_lines:
        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
        cv2.putText(output_img, line, (10, current_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        current_y += LINE_HEIGHT

    # 5. Save the combined output image to a file
    output_filename = "entropy_analysis_output.png"
    cv2.imwrite(output_filename, output_img)
    print(f"Analysis complete. Visual output saved to '{output_filename}'")

if __name__ == "__main__":
    main()