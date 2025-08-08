import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import os
from datetime import datetime
import threading
import time
import numpy as np
import csv
import random


class PhotoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HIVE Data Collection - Photo Capture")
        self.root.geometry("900x700")

        # Set CustomTkinter appearance and theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("green")

        # Initialize rotation angle, lock, and camera
        self.current_rotation_angle = 0
        self.save_lock = threading.Lock()
        self.camera_index = 0
        self.available_cameras = self.detect_cameras()

        # Variables
        self.is_capturing = False
        self.save_folder = "captured_photos"
        self.capture_interval = 1
        self.max_images = 1000
        self.image_count = 0
        self.sequence_number = 0  # For unique filenames
        self.brightness = 0
        self.contrast = 1.0
        self.resolution = (640, 480)
        self.color_mode = "RGB"
        self.apply_blur = False
        self.apply_sharpen = False
        self.apply_noise = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.rotate = False
        self.face_detection = False
        self.label = ""
        # Transformation variables
        self.scale_factor = 1.0
        self.interpolation_mode = "Bilinear"
        self.translation_x = 0
        self.translation_y = 0
        self.affine_transform = False
        self.perspective_transform = False
        self.crop_x1 = 0
        self.crop_y1 = 0
        self.crop_x2 = 640
        self.crop_y2 = 480
        self.padding = 0
        self.shear_x = 0.0
        self.warp = False
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Show intro screen
        self.show_intro_screen()

    def detect_cameras(self):
        """Detect available cameras (up to index 3)."""
        available = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(str(i))
                cap.release()
        return available if available else ["0"]

    def show_intro_screen(self):
        """Display the introductory screen with title and start button."""
        self.intro_frame = ctk.CTkFrame(self.root)
        self.intro_frame.pack(fill="both", expand=True)

        title_label = ctk.CTkLabel(
            self.intro_frame,
            text="Quantum AI Vision Centre of Excellence",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=100)

        start_button = ctk.CTkButton(
            self.intro_frame,
            text="Start Application",
            command=self.show_main_screen,
            width=200,
            height=40,
            font=ctk.CTkFont(size=16)
        )
        start_button.pack(pady=20)

    def show_main_screen(self):
        """Set up the main application screen."""
        self.intro_frame.destroy()

        # Initialize webcam
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.quit()
            return

        # Main frame with scrollable sections
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Video display
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=10)

        # Scrollable control frame
        self.control_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.control_frame.pack(fill="x", padx=10, pady=5)

        # Section 1: Capture Controls
        capture_frame = ctk.CTkFrame(self.control_frame)
        capture_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(capture_frame, text="Capture Controls", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5)

        self.start_button = ctk.CTkButton(capture_frame, text="Start Capture", command=self.start_capture, width=120)
        self.start_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ctk.CTkButton(capture_frame, text="Stop Capture", command=self.stop_capture, state="disabled", width=120)
        self.stop_button.pack(side="left", padx=5, pady=5)
        self.manual_button = ctk.CTkButton(capture_frame, text="Manual Capture", command=self.manual_capture, width=120)
        self.manual_button.pack(side="left", padx=5, pady=5)

        # Section 2: Capture Settings
        settings_frame = ctk.CTkFrame(self.control_frame)
        settings_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(settings_frame, text="Capture Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5)

        ctk.CTkLabel(settings_frame, text="Camera Selection:").pack(anchor="w", padx=5)
        self.camera_var = ctk.StringVar(value=str(self.camera_index))
        ctk.CTkOptionMenu(settings_frame, variable=self.camera_var, values=self.available_cameras, command=self.change_camera, width=150).pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(settings_frame, text="Capture Interval (s):").pack(anchor="w", padx=5)
        self.interval_entry = ctk.CTkEntry(settings_frame, width=150)
        self.interval_entry.insert(0, str(self.capture_interval))
        self.interval_entry.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(settings_frame, text="Save Folder:").pack(anchor="w", padx=5)
        self.folder_entry = ctk.CTkEntry(settings_frame, width=150)
        self.folder_entry.insert(0, self.save_folder)
        self.folder_entry.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(settings_frame, text="Max Images:").pack(anchor="w", padx=5)
        self.max_images_entry = ctk.CTkEntry(settings_frame, width=150)
        self.max_images_entry.insert(0, str(self.max_images))
        self.max_images_entry.pack(anchor="w", padx=5, pady=2)

        # Section 3: Image Adjustments
        adjust_frame = ctk.CTkFrame(self.control_frame)
        adjust_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(adjust_frame, text="Image Adjustments", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5)

        ctk.CTkLabel(adjust_frame, text="Brightness:").pack(anchor="w", padx=5)
        self.brightness_scale = ctk.CTkSlider(adjust_frame, from_=-100, to=100, command=self.update_brightness, width=150)
        self.brightness_scale.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(adjust_frame, text="Contrast:").pack(anchor="w", padx=5)
        self.contrast_scale = ctk.CTkSlider(adjust_frame, from_=0.1, to=2.0, command=self.update_contrast, width=150)
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(adjust_frame, text="Resolution:").pack(anchor="w", padx=5)
        self.resolution_var = ctk.StringVar(value="640x480")
        resolutions = ["224x224", "512x512", "640x480", "1280x720"]
        ctk.CTkOptionMenu(adjust_frame, variable=self.resolution_var, values=resolutions, command=self.update_resolution, width=150).pack(anchor="w", padx=5, pady=2)

        # Section 4: Filters
        filter_frame = ctk.CTkFrame(self.control_frame)
        filter_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(filter_frame, text="Filters", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5)

        self.blur_var = ctk.BooleanVar()
        ctk.CTkCheckBox(filter_frame, text="Apply Blur", variable=self.blur_var).pack(anchor="w", padx=5, pady=2)
        self.sharpen_var = ctk.BooleanVar()
        ctk.CTkCheckBox(filter_frame, text="Apply Sharpen", variable=self.sharpen_var).pack(anchor="w", padx=5, pady=2)
        self.noise_var = ctk.BooleanVar()
        ctk.CTkCheckBox(filter_frame, text="Apply Noise", variable=self.noise_var).pack(anchor="w", padx=5, pady=2)

        # Color mode
        self.color_mode_var = ctk.StringVar(value="RGB")
        ctk.CTkLabel(filter_frame, text="Color Mode:").pack(anchor="w", padx=5)
        ctk.CTkRadioButton(filter_frame, text="RGB", variable=self.color_mode_var, value="RGB").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(filter_frame, text="Grayscale", variable=self.color_mode_var, value="Grayscale").pack(anchor="w", padx=10, pady=2)

        # Section 5: Transformations
        transform_frame = ctk.CTkFrame(self.control_frame)
        transform_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(transform_frame, text="Geometric Transformations", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=5)

        # Scale and Interpolation
        ctk.CTkLabel(transform_frame, text="Scale Factor:").pack(anchor="w", padx=5)
        self.scale_slider = ctk.CTkSlider(transform_frame, from_=0.1, to=1.5, command=self.update_scale, width=150)
        self.scale_slider.set(1.0)
        self.scale_slider.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(transform_frame, text="Interpolation:").pack(anchor="w", padx=5)
        self.interpolation_var = ctk.StringVar(value="Bilinear")
        ctk.CTkOptionMenu(transform_frame, variable=self.interpolation_var, values=["Nearest", "Bilinear", "Bicubic"],
                          command=self.update_interpolation, width=150).pack(anchor="w", padx=5, pady=2)

        # Translation
        ctk.CTkLabel(transform_frame, text="Translation X:").pack(anchor="w", padx=5)
        self.translation_x_slider = ctk.CTkSlider(transform_frame, from_=-100, to=100, command=self.update_translation_x, width=150)
        self.translation_x_slider.pack(anchor="w", padx=5, pady=2)

        ctk.CTkLabel(transform_frame, text="Translation Y:").pack(anchor="w", padx=5)
        self.translation_y_slider = ctk.CTkSlider(transform_frame, from_=-100, to=100, command=self.update_translation_y, width=150)
        self.translation_y_slider.pack(anchor="w", padx=5, pady=2)

        # Affine and Perspective
        self.affine_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Apply Random Affine", variable=self.affine_var).pack(anchor="w", padx=5, pady=2)
        self.perspective_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Apply Random Perspective", variable=self.perspective_var).pack(anchor="w", padx=5, pady=2)

        # Cropping
        ctk.CTkLabel(transform_frame, text="Crop X1, Y1:").pack(anchor="w", padx=5)
        self.crop_x1_entry = ctk.CTkEntry(transform_frame, width=70)
        self.crop_x1_entry.insert(0, "0")
        self.crop_x1_entry.pack(side="left", padx=(5, 2), pady=2)
        self.crop_y1_entry = ctk.CTkEntry(transform_frame, width=70)
        self.crop_y1_entry.insert(0, "0")
        self.crop_y1_entry.pack(side="left", padx=(2, 5), pady=2)

        ctk.CTkLabel(transform_frame, text="Crop X2, Y2:").pack(anchor="w", padx=5)
        self.crop_x2_entry = ctk.CTkEntry(transform_frame, width=70)
        self.crop_x2_entry.insert(0, "640")
        self.crop_x2_entry.pack(side="left", padx=(5, 2), pady=2)
        self.crop_y2_entry = ctk.CTkEntry(transform_frame, width=70)
        self.crop_y2_entry.insert(0, "480")
        self.crop_y2_entry.pack(side="left", padx=(2, 5), pady=2)

        # Padding
        ctk.CTkLabel(transform_frame, text="Padding:").pack(anchor="w", padx=5)
        self.padding_slider = ctk.CTkSlider(transform_frame, from_=0, to=100, command=self.update_padding, width=150)
        self.padding_slider.pack(anchor="w", padx=5, pady=2)

        # Flipping
        self.flip_h_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Flip Horizontal", variable=self.flip_h_var).pack(anchor="w", padx=5, pady=2)
        self.flip_v_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Flip Vertical", variable=self.flip_v_var).pack(anchor="w", padx=5, pady=2)

        # Rotation
        self.rotate_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Random Rotate (±15°)", variable=self.rotate_var,
                        command=self.update_rotation_angle).pack(anchor="w", padx=5, pady=2)

        # Shearing
        ctk.CTkLabel(transform_frame, text="Shear X:").pack(anchor="w", padx=5)
        self.shear_x_slider = ctk.CTkSlider(transform_frame, from_=-0.5, to=0.5, command=self.update_shear_x, width=150)
        self.shear_x_slider.pack(anchor="w", padx=5, pady=2)

        # Warping
        self.warp_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Apply Random Warp", variable=self.warp_var).pack(anchor="w", padx=5, pady=2)

        # Face detection
        self.face_detection_var = ctk.BooleanVar()
        ctk.CTkCheckBox(transform_frame, text="Face Detection", variable=self.face_detection_var).pack(anchor="w", padx=5, pady=2)

        # Label
        ctk.CTkLabel(transform_frame, text="Image Label:").pack(anchor="w", padx=5)
        self.label_entry = ctk.CTkEntry(transform_frame, width=150)
        self.label_entry.pack(anchor="w", padx=5, pady=2)

        # Status and count
        self.status_label = ctk.CTkLabel(self.main_frame, text="Status: Idle")
        self.status_label.pack(pady=5)
        self.count_label = ctk.CTkLabel(self.main_frame, text=f"Images Captured: {self.image_count}/{self.max_images}")
        self.count_label.pack(pady=5)

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Start video feed
        self.update_video()

    def change_camera(self, value):
        """Switch to the selected camera."""
        self.camera_index = int(value)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_label.configure(text=f"Error: Could not open camera {self.camera_index}")
            self.camera_index = 0
            self.cap = cv2.VideoCapture(self.camera_index)
            self.camera_var.set(str(self.camera_index))

    def update_rotation_angle(self):
        """Update the rotation angle when the rotate checkbox is toggled."""
        if self.rotate_var.get():
            self.current_rotation_angle = random.uniform(-15, 15)
            print(f"Rotation angle set to: {self.current_rotation_angle:.2f} degrees")
        else:
            self.current_rotation_angle = 0

    def update_brightness(self, val):
        self.brightness = int(float(val))

    def update_contrast(self, val):
        self.contrast = float(val)

    def update_resolution(self, value):
        w, h = map(int, value.split('x'))
        self.resolution = (w, h)

    def update_scale(self, val):
        self.scale_factor = float(val)

    def update_interpolation(self, value):
        interpolation_map = {
            "Nearest": cv2.INTER_NEAREST,
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC
        }
        self.interpolation_mode = interpolation_map[value]

    def update_translation_x(self, val):
        self.translation_x = int(float(val))

    def update_translation_y(self, val):
        self.translation_y = int(float(val))

    def update_padding(self, val):
        self.padding = int(float(val))

    def update_shear_x(self, val):
        self.shear_x = float(val)

    def apply_transformations(self, frame, for_display=False):
        h, w = frame.shape[:2]

        # Brightness and Contrast
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

        # Color mode
        if self.color_mode_var.get() == "Grayscale":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Gaussian Blur
        if self.blur_var.get():
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Sharpen
        if self.sharpen_var.get():
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)

        # Noise
        if self.noise_var.get():
            noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)

        # Apply scale only when saving, not for display
        if not for_display and self.scale_factor != 1.0:
            new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
            if new_w > 0 and new_h > 0 and new_w <= 1920 and new_h <= 1080:  # Limit max size
                frame = cv2.resize(frame, (new_w, new_h), interpolation=self.interpolation_mode)
                h, w = frame.shape[:2]
            else:
                print("Invalid scale dimensions, skipping resize")

        # Translation
        if self.translation_x != 0 or self.translation_y != 0:
            M = np.float32([[1, 0, self.translation_x], [0, 1, self.translation_y]])
            frame = cv2.warpAffine(frame, M, (w, h))

        # Affine Transformation
        if self.affine_var.get():
            pts1 = np.float32([[0, 0], [w, 0], [0, h]])
            pts2 = np.float32([[random.uniform(-10, 10), random.uniform(-10, 10)],
                              [w + random.uniform(-10, 10), random.uniform(-10, 10)],
                              [random.uniform(-10, 10), h + random.uniform(-10, 10)]])
            M = cv2.getAffineTransform(pts1, pts2)
            frame = cv2.warpAffine(frame, M, (w, h))

        # Perspective Transformation
        if self.perspective_var.get():
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = np.float32([[random.uniform(-20, 20), random.uniform(-20, 20)],
                              [w + random.uniform(-20, 20), random.uniform(-20, 20)],
                              [w + random.uniform(-20, 20), h + random.uniform(-20, 20)],
                              [random.uniform(-20, 20), h + random.uniform(-20, 20)]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, M, (w, h))

        # Cropping
        try:
            x1 = int(self.crop_x1_entry.get())
            y1 = int(self.crop_y1_entry.get())
            x2 = int(self.crop_x2_entry.get())
            y2 = int(self.crop_y2_entry.get())
            if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and x1 < x2 and y1 < y2:
                frame = frame[y1:y2, x1:x2]
                h, w = frame.shape[:2]
            else:
                print("Invalid crop coordinates, skipping crop")
        except ValueError:
            print("Invalid crop coordinates, skipping crop")

        # Padding
        if self.padding > 0:
            frame = cv2.copyMakeBorder(frame, self.padding, self.padding, self.padding, self.padding,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h, w = frame.shape[:2]

        # Flipping
        if self.flip_h_var.get():
            frame = cv2.flip(frame, 1)
        if self.flip_v_var.get():
            frame = cv2.flip(frame, 0)

        # Rotation
        if self.rotate_var.get() and self.current_rotation_angle != 0:
            scale = 1.5
            new_h, new_w = int(h * scale), int(w * scale)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), self.current_rotation_angle, 1)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            frame = cv2.warpAffine(frame, M, (new_w, new_h))
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            frame = frame[start_y:start_y + h, start_x:start_x + w]

        # Shearing
        if self.shear_x != 0:
            M = np.float32([[1, self.shear_x, 0], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (w, h))

        # Warping
        if self.warp_var.get():
            rows, cols = frame.shape[:2]
            src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
            dst_points = np.float32([[random.uniform(0, cols * 0.1), random.uniform(0, rows * 0.1)],
                                    [cols - random.uniform(0, cols * 0.1), random.uniform(0, rows * 0.1)],
                                    [random.uniform(0, cols * 0.1), rows - random.uniform(0, rows * 0.1)],
                                    [cols - random.uniform(0, cols * 0.1), rows - random.uniform(0, rows * 0.1)]])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            frame = cv2.warpPerspective(frame, M, (cols, rows))

        return frame

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Apply transformations for display without scaling
            frame = self.apply_transformations(frame, for_display=True)
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def save_image(self, frame):
        with self.save_lock:
            if self.image_count >= int(self.max_images_entry.get()):
                self.status_label.configure(text="Status: Max images reached")
                self.stop_capture()
                return False
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
            self.sequence_number += 1
            filename = os.path.join(self.save_folder, f"photo_{timestamp}_{self.sequence_number:04d}.jpg")
            # Apply all transformations including scale for saving
            frame = self.apply_transformations(frame, for_display=False)
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)
            success = cv2.imwrite(filename, frame)
            if not success:
                print(f"Error: Failed to save image {filename}")
                self.status_label.configure(text=f"Error: Failed to save {filename}")
                return False
            self.image_count += 1
            self.count_label.configure(text=f"Images Captured: {self.image_count}/{self.max_images}")
            with open(os.path.join(self.save_folder, "metadata.csv"), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, timestamp, self.label_entry.get(), self.brightness, self.contrast,
                                self.resolution, self.color_mode_var.get(), self.blur_var.get(),
                                self.sharpen_var.get(), self.noise_var.get(), self.flip_h_var.get(),
                                self.flip_v_var.get(), self.rotate_var.get(), self.face_detection_var.get(),
                                self.scale_factor, self.interpolation_var.get(), self.translation_x,
                                self.translation_y, self.affine_var.get(), self.perspective_var.get(),
                                f"{self.crop_x1_entry.get()},{self.crop_y1_entry.get()},{self.crop_x2_entry.get()},{self.crop_y2_entry.get()}",
                                self.padding, self.shear_x, self.warp_var.get()])
            self.status_label.configure(text=f"Status: Captured {filename}")
            if self.rotate_var.get():
                self.current_rotation_angle = random.uniform(-15, 15)
                print(f"New rotation angle for capture: {self.current_rotation_angle:.2f} degrees")
            return True

    def start_capture(self):
        if not self.is_capturing:
            try:
                self.capture_interval = float(self.interval_entry.get())
                if self.capture_interval <= 0:
                    self.status_label.configure(text="Error: Interval must be positive")
                    return
                self.save_folder = self.folder_entry.get()
                self.max_images = int(self.max_images_entry.get())
                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                self.image_count = 0
                self.sequence_number = 0
                self.is_capturing = True
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal")
                self.manual_button.configure(state="normal")
                self.status_label.configure(text="Status: Capturing")
                with open(os.path.join(self.save_folder, "metadata.csv"), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Filename", "Timestamp", "Label", "Brightness", "Contrast", "Resolution",
                                    "Color_Mode", "Blur", "Sharpen", "Noise", "Flip_Horizontal", "Flip_Vertical",
                                    "Rotate", "Face_Detection", "Scale_Factor", "Interpolation", "Translation_X",
                                    "Translation_Y", "Affine", "Perspective", "Crop_Coordinates", "Padding",
                                    "Shear_X", "Warp"])
                threading.Thread(target=self.capture_photos, daemon=True).start()
            except ValueError:
                self.status_label.configure(text="Error: Invalid interval value")

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.manual_button.configure(state="normal")
        self.status_label.configure(text="Status: Idle")

    def manual_capture(self):
        ret, frame = self.cap.read()
        if ret and (not self.face_detection_var.get() or self.detect_face(frame)):
            frame = self.apply_transformations(frame, for_display=False)
            self.save_image(frame)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0 and self.face_detection_var.get():
            print("No faces detected, skipping save")
            return False
        return True

    def capture_photos(self):
        while self.is_capturing and self.image_count < self.max_images:
            ret, frame = self.cap.read()
            if ret and (not self.face_detection_var.get() or self.detect_face(frame)):
                frame = self.apply_transformations(frame, for_display=False)
                self.save_image(frame)
            time.sleep(self.capture_interval)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = ctk.CTk()
    app = PhotoCaptureApp(root)
    root.mainloop()