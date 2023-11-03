import sys
import cv2
import numpy as np
import os
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def correct_distortion(img, k1, k2, p1, p2, k3, rotation_angle):
    h, w = img.shape[:2]
    camera_matrix = np.array([[766.14, 0, w / 2.0], [0, 766.14, h / 2.0], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_angle, 1)
    img_rotated = cv2.warpAffine(img_undistorted, M, (w, h))
    return img_rotated

def process_images(input_folder, output_folder):
    image_files = glob.glob(os.path.join(input_folder, "*.JPG"))
    k1, k2, p1, p2, k3, rotation_angle = -0.35245767, 0.13652994, 0.00065283, 0.00075075, 0.1339468, -2
    for file in image_files:
        img = cv2.imread(file)
        img_corrected = correct_distortion(img, k1, k2, p1, p2, k3, rotation_angle)
        filename = os.path.basename(file)
        output_file_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_file_path, img_corrected, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Successfully wrote {output_file_path}")

def get_gps_coordinates(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if not exif_data:
        return None
    geo_data = {}
    for tag, value in TAGS.items():
        if tag in exif_data:
            if TAGS[tag] == "GPSInfo":
                for tag_value in exif_data[tag]:
                    if tag_value in GPSTAGS:
                        geo_data[GPSTAGS[tag_value]] = exif_data[tag][tag_value]

    lat_data = geo_data.get("GPSLatitude", None)
    lon_data = geo_data.get("GPSLongitude", None)

    if not lat_data or not lon_data:
        return None

    lat = sum(float(x)/float(y) for x, y in zip(lat_data, (1, 60.0, 3600.0)))
    lon = sum(float(x)/float(y) for x, y in zip(lon_data, (1, 60.0, 3600.0)))

    lat = lat if geo_data["GPSLatitudeRef"] == "N" else -lat
    lon = lon if geo_data["GPSLongitudeRef"] == "E" else -lon

    return lat, lon

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processor')
        self.image_list = []
        self.current_index = 0
        self.input_folder_path = "T_images"
        self.output_folder_path = "output"

        layout = QVBoxLayout()

        # Add a new QLabel to show the image name
        self.image_name_label = QLabel(self)
        layout.addWidget(self.image_name_label, alignment=QtCore.Qt.AlignCenter)

        self.image_label = QLabel(self)
        pixmap = QPixmap(400, 300)  # Sample size placeholder
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)
        btn_select_input = QPushButton('Select Input Folder', self)
        btn_select_input.clicked.connect(self.set_input_folder)
        layout.addWidget(btn_select_input, alignment=QtCore.Qt.AlignCenter)
        btn_select_output = QPushButton('Select Output Folder', self)
        btn_select_output.clicked.connect(self.set_output_folder)
        layout.addWidget(btn_select_output, alignment=QtCore.Qt.AlignCenter)
        btn_process = QPushButton('Process Images', self)
        btn_process.clicked.connect(self.process_and_load)
        layout.addWidget(btn_process, alignment=QtCore.Qt.AlignCenter)
        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.prev_image)
        layout.addWidget(self.prev_button, alignment=QtCore.Qt.AlignCenter)
        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.next_image)
        layout.addWidget(self.next_button, alignment=QtCore.Qt.AlignCenter)
        self.input_path_label = QLabel(self)
        self.output_path_label = QLabel(self)
        layout.addWidget(self.input_path_label)
        layout.addWidget(self.output_path_label)

        self.setLayout(layout)
        self.update_path_labels()

    def update_path_labels(self):
        self.input_path_label.setText(f"Input Folder: {self.input_folder_path}")
        self.output_path_label.setText(f"Output Folder: {self.output_folder_path}")

    def set_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path:
            self.input_folder_path = path
            self.update_path_labels()

    def set_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder_path = path
            self.update_path_labels()

    def process_and_load(self):
        process_images(self.input_folder_path, self.output_folder_path)
        self.load_original_images()  # Load original images for extracting GPS data

    def load_original_images(self):
        self.image_list = glob.glob(os.path.join(self.input_folder_path, "*.JPG"))  # Load original images
        if self.image_list:
            self.show_image(self.image_list[0],
                            self.output_folder_path)  # Show processed image but get GPS from the original

    def show_image(self, original_path, output_folder):
        # Use the original image path to get the processed image path
        filename = os.path.basename(original_path)
        processed_path = os.path.join(output_folder, filename)

        image = cv2.imread(processed_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        filename = os.path.basename(original_path)
        self.image_name_label.setText(f"Image Name: {filename}")
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        coords = get_gps_coordinates(original_path)  # Extract GPS from the original image
        if coords:
            lat, lon = coords
            self.input_path_label.setText(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}")
        else:
            self.input_path_label.setText("GPS coordinates not found.")

    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.image_list[self.current_index], self.output_folder_path)

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image(self.image_list[self.current_index], self.output_folder_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.setGeometry(100, 100, 600, 600)
    window.show()
    sys.exit(app.exec_())