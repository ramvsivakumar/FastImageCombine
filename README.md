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



