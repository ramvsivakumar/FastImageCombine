import cv2
import numpy as np
from PIL import Image
import multiprocessing
import concurrent.futures


class ImageProcessor:
    def __init__(self):
        self.num_cpus = multiprocessing.cpu_count()
        self.num_workers = max(1, int(self.num_cpus * 0.7))

    def process_image(self, image_paths):
        """
        1) Transform an image by making green or shades of green -> transparent
        2) Turning blueish shades of pixels -> white.
        """
        input_path, output_path = image_paths
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

    def blend_images(self, image_paths):
        """
        Blend two images into one and save the result.
        """
        image_path1, image_path2, output_path = image_paths
        image1 = Image.open(image_path1).convert('RGBA')
        image2 = Image.open(image_path2).convert('RGBA')

        # empty canvas
        result_image = Image.new('RGBA', image1.size)

        result_image.paste(image2, (0, 0))
        result_image.paste(image1, (0, 0), image1)

        result_image.save(output_path, 'PNG')
        return f"Blended image saved to {output_path}"

    def stack_images_horizontally(self, image_paths):
        """
        Stack two images horizontally and save the result.
        """

        image_path1, image_path2, output_path = image_paths
        image1 = Image.open(image_path1).convert('RGBA')
        image2 = Image.open(image_path2).convert('RGBA')

        total_width = image1.width + image2.width
        max_height = max(image1.height, image2.height)

        new_image = Image.new('RGBA', (total_width, max_height))

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

        new_image.save(output_path, 'PNG')
        return f"Stacked image saved to {output_path}"

    def outline_black_areas(self, image_paths):
        """
        Find all black areas in an image, including nested areas, and outline them with a red line (1px width),
        """
        image_path, output_path = image_paths
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image.shape[2] == 3:
            alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255  # Fully opaque
            image = np.dstack([image, alpha_channel])
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

        # Create a mask for black (and near-black) areas, considering opacity
        color_channels = image[:, :, :3]
        alpha_channel = image[:, :, 3]
        mask = (alpha_channel == 255) & np.all(color_channels <= 50, axis=2)
        mask = np.uint8(mask) * 255

        # Find contours using RETR_TREE
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw red lines around the black areas on a transparent layer
        line_layer = np.zeros_like(image)
        cv2.drawContours(line_layer, contours, -1, (0, 0, 255, 255), 1)  # Red color with full opacity

        result_image = cv2.addWeighted(image, 1, line_layer, 1, 0)

        cv2.imwrite(output_path, result_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        return f"Image with outlined black areas saved to {output_path}"

    def save_enhanced_grayscale(self, image_paths):
        image_path, output_path = image_paths
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

    def process_in_parallel(self, function, paths):
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(function, paths))
        return results

    def main(self):
        # Add paths and call functions for processing
        image_paths = [("Layer1.bmp", "Layer1_transformed.png"), ("Layer2.bmp", "Layer2_transformed.png")]
        blend_paths = [("Layer1_transformed.png", "Layer2_transformed.png", "Blended.png")]
        stack_paths = [("Layer1_transformed.png", "Layer2_transformed.png", "Stacked.png")]
        outline_paths = [("Blended.png", "Outlined_Black_Areas.png")]
        grayscale_paths = [("Outlined_Black_Areas.png", "Enhanced_Grayscale_Output.png")]

        process_results = self.process_in_parallel(self.process_image, image_paths)
        blend_results = self.process_in_parallel(self.blend_images, blend_paths)
        stack_results = self.process_in_parallel(self.stack_images_horizontally, stack_paths)
        outline_results = self.process_in_parallel(self.outline_black_areas, outline_paths)
        grayscale_results = self.process_in_parallel(self.save_enhanced_grayscale, grayscale_paths)

        print(process_results, blend_results, stack_results, outline_results, grayscale_results)


if __name__ == "__main__":
    processor = ImageProcessor()
    processor.main()