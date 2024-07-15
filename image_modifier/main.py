import numpy as np
from PIL import Image
import cv2
import multiprocessing
import concurrent.futures


def process_image(image_path):
    """
    1) Transform an image by making green or shades of green -> transparent
    2) Turning blueish shades of pixels -> white.
    """
    input_path, output_path = image_path
    image = Image.open(input_path).convert('RGBA')
    arr = np.array(image)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # Define ranges for greenish and blueish colors in HSV
    lower_green = np.array([40, 40, 40])  # Lower bound
    upper_green = np.array([80, 255, 255])  # Upper bound

    lower_blue = np.array([100, 40, 40])  # Lower bound
    upper_blue = np.array([140, 255, 255])  # Upper bound

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    transparent_mask = (green_mask > 0)
    arr[transparent_mask, 3] = 0
    arr[transparent_mask, :3] = 0

    # Blue range pixels to white
    arr[blue_mask > 0, :3] = 255

    transformed_image = Image.fromarray(arr)
    transformed_image.save(output_path, format='PNG')
    return f"Processed and saved: {output_path}"


def blend_images(image_path1, image_path2, output_path):
    """
    Blend two images into one and save the result.
    """
    # Load the images
    image1 = Image.open(image_path1).convert('RGBA')
    image2 = Image.open(image_path2).convert('RGBA')

    blended_image = Image.blend(image1, image2, alpha=0.5)
    blended_image.save(output_path, 'PNG')
    return f"Blended image saved to {output_path}"


def main():
    # List of image files to process. Each element is a tuple (input_path, output_path).
    image_paths = [("Layer1.bmp", "Layer1_transformed.png"), ("Layer2.bmp", "Layer2_transformed.png")]

    # Determine the number of workers based on available CPU cores.
    num_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(num_cpus * 0.7))

    # Using ProcessPoolExecutor to handle image processing in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, paths) for paths in image_paths]

        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
