# FastImageCombine

## **Task 1:**
Convert these 2 images using the following rules
- Keep all black or white pixels
- Green should become fully transparent
- Blue should become white

## **Task 2:**
- Combine both images to one.

## **Task 3:**
- Now find all black areas and draw a red line of 1 pixel thickness around them

## **Task 4:**
- Store the result into a 8bit Grayscale image. You can freely choose any image format you are
comfortable with.

## **Task 5:**
- Increase your performance by using multi threads or cores to solve Tasks 1 to 3

# **Optional Task**
## **Task 6:**
Prepare a User Interface where the User can choose one or more of the following:
-  Thickness of the red outline from Task 3
-  Color of the outline from task 3
-  Color conversion rules from task 1
-  Output format for task 4

# Requirements
- NumPy
- Pillow
- OpenCV-Python


# imageprocessor.py

This Python-based utility leverages OpenCV, NumPy, and PIL to perform advanced image-processing tasks. It provides functionalities such as color transformations, image blending, horizontal stacking, outlining black areas, and enhancing grayscale images. The utility employs multiprocessing for efficient handling of image processing tasks.

## Features

- **Color Transformation:** Converts specific color shades (green to transparent, blue to white).
- **Image Blending:** Merges two images together.
- **Horizontal Stacking:** Joins two images side by side.
- **Outlining Black Areas:** Identifies and outlines black areas in images with a red line.
- **Enhanced Grayscale:** Converts images to an enhanced grayscale format.


# gradio_image_processor.py
This Python application uses image processing techniques such as OpenCV and PIL, integrated with a Gradio interface for enhanced user interaction. The application allows users to combine images, customize colors, outline specific areas, and perform other image enhancements in real-time.

## Features

- **Image Combination**: Combines two images into one using alpha compositing techniques.
- **Color Processing**: Custom functions to alter image colors—making specified colors transparent or white.
- **Outlining**: Detect and outline specific color areas within the image.
- **Enhancement**: Enhance and convert images to grayscale while tweaking specific color channels.
- **Image Saving**: Option for users to save the processed images locally.


# Gradio Interface Components

## Inputs
- **Color to Make Transparent:** Choose a color via a color picker to be made transparent in the image.
- **Color to Convert to White:** Select a color that will be converted to white in the processed image.
- **Outline Color:** Pick a color for the outline of specified areas within the image.
- **Outline Thickness:** Use a slider to adjust the thickness of the outline.

## Outputs
- **Processed Image:** Displays the image after processing steps have been applied.
- **Save Result:** A text output that indicates the successful saving of the processed image.

## Controls
- **Color Pickers:** For selecting transparent and white conversion colors, and the outline color.
- **Slider:** To adjust the thickness of outlines.
- **Process Image Button:** Applies the selected image processing techniques.
- **Save Image Button:** Saves the processed image to a specified directory.

## Configuration
The server is configured to run on all network interfaces (0.0.0.0) using port 8190. Adjust the server address or port as needed based on your deployment requirements.
