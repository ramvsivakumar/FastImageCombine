#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <exception>

class ImageProcessor {
public:
    cv::Mat loadImage(const std::string& path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image at " << path << std::endl;
            throw std::runtime_error("Failed to load image.");
        }
        return img;
    }

 
    void saveImage(const cv::Mat& img, const std::string& path) {
        if (!cv::imwrite(path, img)) {
            std::cerr << "Failed to save image to " << path << std::endl;
            throw std::runtime_error("Failed to save image.");
        }
        std::cout << "Image saved to " << path << std::endl;
    }

    void processImage(const std::string& inputPath, const std::string& outputPath) {
        cv::Mat image = loadImage(inputPath);
        cv::Mat hsvImage;
        cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
        applyColorMask(hsvImage, image, cv::Scalar(40, 40, 40), cv::Scalar(80, 255, 255), true);
        applyColorMask(hsvImage, image, cv::Scalar(100, 40, 40), cv::Scalar(140, 255, 255), false, cv::Scalar(255, 255, 255, 255));
        saveImage(image, outputPath);
    }
	
    void applyColorMask(cv::Mat& hsvImage, cv::Mat& image, cv::Scalar low, cv::Scalar high, bool transparent, cv::Scalar fillColor=cv::Scalar(0, 0, 0, 0)) {
        cv::Mat mask;
        cv::inRange(hsvImage, low, high, mask);
        if (transparent) {
            makeTransparent(image, mask);
        } else {
            image.setTo(fillColor, mask);
        }
    }

    void makeTransparent(cv::Mat& image, const cv::Mat& mask) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                if (mask.at<uint8_t>(i, j)) {
                    image.at<cv::Vec4b>(i, j)[3] = 0;
                }
            }
        }
    }

    void blendImages(const std::string& path1, const std::string& path2, const std::string& outputPath) {
        cv::Mat image1 = loadImage(path1);
        cv::Mat image2 = loadImage(path2);
        cv::Mat result;
        cv::addWeighted(image1, 0.5, image2, 0.5, 0, result);
        saveImage(result, outputPath);
    }

    void stackImagesHorizontally(const std::string& path1, const std::string& path2, const std::string& outputPath) {
        cv::Mat image1 = loadImage(path1);
        cv::Mat image2 = loadImage(path2);
        cv::Mat result(cv::max(image1.rows, image2.rows), image1.cols + image2.cols, image1.type());
        image1.copyTo(result(cv::Rect(0, 0, image1.cols, image1.rows)));
        image2.copyTo(result(cv::Rect(image1.cols, 0, image2.cols, image2.rows)));
        saveImage(result, outputPath);
    }

    void outlineBlackAreas(const std::string& inputPath, const std::string& outputPath) {
        cv::Mat image = loadImage(inputPath);
        cv::Mat gray, thresh;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, thresh, 15, 255, cv::THRESH_BINARY_INV);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            cv::drawContours(image, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 255), 2);
        }
        saveImage(image, outputPath);
    }

    void saveEnhancedGrayscale(const std::string& inputPath, const std::string& outputPath) {
        cv::Mat image = loadImage(inputPath);
        cv::Mat gray, enhanced;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, enhanced);
        saveImage(enhanced, outputPath);
    }
};

int main() {
    ImageProcessor processor;
    try {
        processor.processImage("Layer1.bmp", "Layer1_transformed.png");
        processor.processImage("Layer2.bmp", "Layer2_transformed.png");
        processor.blendImages("Layer1_transformed.png", "Layer2_transformed.png", "Blended.png");
        processor.stackImagesHorizontally("Layer1_transformed.png", "Layer2_transformed.png", "Stacked.png");
        processor.outlineBlackAreas("Blended.png", "Outlined_Black_Areas.png");
        processor.saveEnhancedGrayscale("Outlined_Black_Areas.png", "Enhanced_Grayscale_Output.png");
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    return 0;
}

