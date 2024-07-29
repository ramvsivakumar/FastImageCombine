import cv2
import numpy as np
from PIL import Image
import gradio as gr
import multiprocessing
import concurrent.futures


class ImageProcessor:
    def __init__(self):
        self.num_cpus = multiprocessing.cpu_count()
        self.num_workers = max(1, int(self.num_cpus * 0.7))

    def read_and_combine_images(self, image_path1, image_path2):
        image1 = Image.open(image_path1).convert('RGBA')
        image2 = Image.open(image_path2).convert('RGBA')

        result_image = Image.new('RGBA', image1.size)

        result_image.paste(image2, (0, 0))
        result_image.paste(image1, (0, 0), image1)

        return result_image

    def custom_color_process(self, image, color_to_transparent, color_to_white):
        arr = np.array(image.convert('RGBA'))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # Convert hex colors to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        # Convert RGB to HSV
        def rgb_to_hsv(rgb_color):
            return cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]

        # Define ranges based on selected colors
        target_transparent_hsv = rgb_to_hsv(hex_to_rgb(color_to_transparent))
        target_white_hsv = rgb_to_hsv(hex_to_rgb(color_to_white))

        # Creating masks
        transparent_mask = cv2.inRange(hsv, target_transparent_hsv - np.array([10, 100, 100]),
                                       target_transparent_hsv + np.array([10, 100, 100]))
        white_mask = cv2.inRange(hsv, target_white_hsv - np.array([10, 100, 100]),
                                 target_white_hsv + np.array([10, 100, 100]))

        # Apply masks
        arr[transparent_mask > 0, 3] = 0
        arr[white_mask > 0, :3] = 255

        return Image.fromarray(arr)

    def outline_black_areas(self, image, line_color, line_thickness):
        image = np.array(image)

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
        cv2.drawContours(line_layer, contours, -1, tuple(int(line_color[i:i + 2], 16) for i in (1, 3, 5)), int(line_thickness))
        result_image = cv2.addWeighted(image, 1, line_layer, 1, 0)

        return Image.fromarray(result_image)

    def enhance_to_grayscale(self, image):
        image = np.array(image)

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

        return Image.fromarray(final_image)

    def save_image(self, image, path):
        image = np.array(image)
        global alpha_channel

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

        cv2.imwrite(path, final_image)
        return f"Image successfully saved to {path}"


def create_interface(combined_image):
    processor = ImageProcessor()

    def update_image(color_to_transparent, color_to_white, line_color, line_thickness):
        processed_image = processor.custom_color_process(combined_image, color_to_transparent, color_to_white)
        outlined_image = processor.outline_black_areas(processed_image, line_color, line_thickness)
        grayscale_image = processor.enhance_to_grayscale(outlined_image)
        return grayscale_image

    def save_result(image, path):
        img = Image.fromarray(image)
        return processor.save_image(img, path)

    with gr.Blocks() as demo:
        with gr.Row():
            color_to_transparent = gr.ColorPicker(value="#00FF00", label="Color to Make Transparent")
            color_to_white = gr.ColorPicker(value="#0000FF", label="Color to Convert to White")
            line_color = gr.ColorPicker(value="#FF0000", label="Outline Color")
            line_thickness = gr.Slider(minimum=1, maximum=10, value=1, label="Outline Thickness")
        with gr.Row():
            submit_button = gr.Button("Process Image")
            image_output = gr.Image(label="Processed Image")
            path_input = gr.Textbox(value="output_image.png", label="Save Path")
            save_button = gr.Button("Save Image")

        submit_button.click(
            fn=update_image,
            inputs=[color_to_transparent, color_to_white, line_color, line_thickness],
            outputs=image_output
        )

        save_button.click(
            fn=save_result,
            inputs=[image_output, path_input],
            outputs=gr.Text(label="Save Result")
        )

    return demo


if __name__ == "__main__":
    processor = ImageProcessor()
    combined_image = processor.read_and_combine_images("Layer1.bmp", "Layer2.bmp")
    interface = create_interface(combined_image)
    interface.launch(server_name='0.0.0.0', server_port=8190, share=True)
