# HIVE Data Collection - Photo Capture Application

## Overview
This repository contains a Python application for capturing photos from a webcam with various image adjustments and geometric transformations. Built using `customtkinter` for a modern GUI, `OpenCV` for image processing, and `PIL` for image handling, this tool is designed for data collection in AI vision projects, such as the Quantum AI Vision Centre of Excellence.

## Features
- Real-time webcam preview with adjustable resolution.
- Automated and manual photo capture with customizable intervals and limits.
- Image adjustments: brightness, contrast, and color mode (RGB/Grayscale).
- Filters: blur, sharpen, and noise.
- Geometric transformations: scaling, translation, affine, perspective, cropping, padding, flipping, rotation, shearing, and warping.
- Face detection option to filter captures.
- Metadata logging for all captured images.
- Toggleable transformation controls with reset buttons for sliders.
- Responsive UI with a scrollable options panel.

## Prerequisites
- Python 3.7 or higher.
- Required packages: `customtkinter`, `opencv-python`, `Pillow`, `numpy`.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/photo-capture-app.git
   cd photo-capture-app
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Create a `requirements.txt` file with the following content and include it in the repository:
   ```
   customtkinter
   opencv-python
   Pillow
   numpy
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. The intro screen will appear. Click "Start Application" to proceed.
3. Configure capture settings (camera, interval, folder, max images) and image adjustments/filters.
4. Use the "Start Capture" button for automated captures or "Manual Capture" for single shots.
5. Toggle "Enable Geometric Transformations" to show/hide advanced options and use reset buttons to revert sliders.
6. Captured images are saved in the specified folder with metadata in a `metadata.csv` file.

## Configuration
- Adjust sliders and checkboxes in the UI to modify image properties.
- Enter crop coordinates, padding values, and labels as needed.
- The status bar below the video feed updates with capture progress.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request with a description of your changes.

Please ensure your code follows the existing style and includes appropriate comments.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the `customtkinter`, `OpenCV`, and `Pillow` communities for their powerful tools.
- Inspired by the need for efficient data collection in AI vision research.