
# 10XLabel

10XLabel is a Python script designed to enrich image datasets through various augmentation techniques. It leverages existing image and label files to apply transformations such as rotation, brightness enhancement, Gaussian noise addition, shadow effect creation, and horizontal and vertical flipping, while appropriately transforming the associated object labels in the images. This script is particularly useful for image processing and machine learning projects where augmented data can significantly improve model performance.

## Getting Started

This section explains how to set up and run the project.

### Prerequisites

- Python 3.x
- OpenCV library (cv2)
- NumPy library
- Math library

### Installation

Clone or download the project:

```bash
git clone https://github.com/yourusername/10XLabel.git
```

Install the required dependencies:

```bash
pip install opencv-python numpy
```

### Usage

Navigate to the terminal or command line and run the following command:

```bash
python3 image_augmentation_script.py
```

## Features

- Rotate images at specified angles and update labels.
- Increase image brightness without changing labels.
- Distort images by adding Gaussian noise while keeping labels unchanged.
- Add shadow effects to images without altering labels.
- Flip images horizontally and vertically and update labels accordingly.
- Zoom out images and adjust label coordinates.

## Contributing

Guidelines for those who wish to contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact

Yavuz Selim - yavuzselim.xyz@gmail.com
LinkedIn: [Yavuz Selim](https://www.linkedin.com/in/yavuzselimxyz/)
Project Link: https://github.com/yavuzxyz/10XLabel
