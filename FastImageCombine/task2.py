from PIL import Image
import cv2
import numpy as np
def blend_images(image_path1, image_path2, output_path):
    """
    Blend two images into one and save the result.
    """
    image1 = Image.open(image_path1).convert('RGBA')
    image2 = Image.open(image_path2).convert('RGBA')

    # Prepare an empty canvas with the same size as the images
    result_image = Image.new('RGBA', image1.size)

    # Paste the background image into the canvas
    result_image.paste(image2, (0, 0))

    # Paste the foreground image into the canvas, considering alpha transparency
    result_image.paste(image1, (0, 0), image1)

    # Save the resulting blended image
    result_image.save(output_path, 'PNG')
    return f"Blended image saved to {output_path}"


def stack_images_horizontally(image_path1, image_path2, output_path):
    """
    Stack two images horizontally and save the result.
    """
    # Load the images
    image1 = Image.open(image_path1).convert('RGBA')
    image2 = Image.open(image_path2).convert('RGBA')

    # Calculate dimensions for the new image
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)

    # Create a new image with appropriate dimensions
    new_image = Image.new('RGBA', (total_width, max_height))

    # Paste images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    # Save the new image
    new_image.save(output_path, 'PNG')
    return f"Stacked image saved to {output_path}"


def outline_black_areas(image_path, output_path):
    """
    Find all black areas in an image, including nested shapes, and outline them with a red line,
    while preserving transparency.
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with outlined black areas.
    """
    # Load the image ensuring alpha channel is considered if present
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Make sure the image has four channels (RGBA)
    if image.shape[2] == 3:  # It's RGB, need to add alpha channel
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255  # Fully opaque
        image = np.dstack([image, alpha_channel])
    elif image.shape[2] == 1:  # It's grayscale, convert to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

    # Create a mask for black (and near-black) areas, considering opacity
    color_channels = image[:, :, :3]
    alpha_channel = image[:, :, 3]
    mask = (alpha_channel == 255) & np.all(color_channels <= 50, axis=2)

    # Convert mask to uint8 type
    mask = np.uint8(mask) * 255

    # Find contours using RETR_TREE to capture all hierarchical contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red lines around the black areas on a transparent layer
    line_layer = np.zeros_like(image)
    cv2.drawContours(line_layer, contours, -1, (0, 0, 255, 255), 1)  # Red color with full opacity

    # Combine the line layer with the original image
    result_image = cv2.addWeighted(image, 1, line_layer, 1, 0)

    # Save the image ensuring the alpha channel is preserved
    cv2.imwrite(output_path, result_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    return f"Image with outlined black areas saved to {output_path}"


def save_enhanced_grayscale(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:  # RGBA
        alpha_channel = image[:, :, 3]  # Extract alpha channel before processing
        image = image[:, :, :3]  # Remove alpha for processing

    image_float = image.astype(np.float32)
    b, g, r = cv2.split(image_float)

    r = cv2.add(r, 50)  # Enhance red channel
    b = cv2.subtract(b, 50)  # Suppress blue channel
    g = cv2.subtract(g, 50)  # Suppress green channel

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    enhanced_image = cv2.merge((b, g, r)).astype(np.uint8)
    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR to match alpha channel size

    if 'alpha_channel' in locals():  # Check if alpha channel was originally present
        final_image = cv2.merge((gray_image[:, :, 0], gray_image[:, :, 1], gray_image[:, :, 2], alpha_channel))
    else:
        final_image = gray_image

    cv2.imwrite(output_path, final_image)
    return f"Enhanced grayscale image saved to {output_path}"


if __name__ == "__main__":
    result = blend_images("Layer1_transformed.png", "Layer2_transformed.png", "Blended2.png")
    result2 = stack_images_horizontally("Layer1_transformed.png", "Layer2_transformed.png", "Stacked.png")
    result3 = outline_black_areas("Blended2.png", "Outlined_Black_Areas.png")
    save_enhanced_grayscale("Outlined_Black_Areas.png", "Enhanced_Grayscale_Output.png")

