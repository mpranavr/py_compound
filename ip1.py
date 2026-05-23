import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from ipl1 import (
    nearest_neighbour,
    bilinear_interpolation,
    rotation
)

from ipl2 import (
    img_read,
    img_write,
    red_color_chl_manipulation,
    blue_color_chl_manipulation,
    green_color_chl_manipulation
)

from ipl3 import *
from ipl4 import *
from ipl5 import *
from ipl6 import *
from ipl7 import *
from ipl8 import *

ipapp = QApplication(sys.argv)

window = QWidget()

window.setWindowTitle("IP Window")

window.resize(1000, 700)

tabs = QTabWidget()


class Interpole(QWidget):

    def __init__(self):

        super().__init__()

        self.layout1 = QVBoxLayout()

        # ---------------- INPUT BUTTON ----------------

        self.button = QPushButton("Input Image")

        self.button.clicked.connect(
            self.open_image
        )

        self.layout1.addWidget(self.button)

        # ---------------- TOOL BUTTON ----------------

        self.interpole_button = QToolButton()

        self.interpole_button.setArrowType(
            Qt.DownArrow
        )

        self.interpole_button.setAutoRaise(True)

        self.layout1.addWidget(
            self.interpole_button,
            alignment=Qt.AlignCenter
        )

        # ---------------- COMBO BOX ----------------

        self.combo = QComboBox()

        self.combo.addItems([
            "Nearest Neighbour",
            "Bilinear",
            "Rotation"
        ])

        self.layout1.addWidget(self.combo)

        # ---------------- INPUT BOX ----------------

        self.scale_box = QLineEdit()

        self.scale_box.setPlaceholderText(
            "Enter Scale Factor / Rotation Angle"
        )

        self.layout1.addWidget(self.scale_box)

        # ---------------- PROCESS BUTTON ----------------

        self.process_button = QPushButton("Process")

        self.process_button.clicked.connect(
            self.process_image
        )

        self.layout1.addWidget(self.process_button)

        # ---------------- IMAGE DISPLAY AREA ----------------

        self.image_layout = QHBoxLayout()

        # INPUT LABEL

        self.input_label = QLabel(
            "Input Image"
        )

        self.input_label.setFixedSize(
            400,
            400
        )

        self.input_label.setAlignment(
            Qt.AlignCenter
        )

        self.input_label.setStyleSheet(
            "border: 2px solid black;"
        )

        # OUTPUT LABEL

        self.output_label = QLabel(
            "Output Image"
        )

        self.output_label.setFixedSize(
            400,
            400
        )

        self.output_label.setAlignment(
            Qt.AlignCenter
        )

        self.output_label.setStyleSheet(
            "border: 2px solid black;"
        )

        self.image_layout.addWidget(
            self.input_label
        )

        self.image_layout.addWidget(
            self.output_label
        )

        self.layout1.addLayout(
            self.image_layout
        )

        self.setLayout(self.layout1)

    # --------------------------------------------------

    def open_image(self):

        self.filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp)"
        )

        if self.filename:

            # COLOR IMAGE
            self.img = cv2.imread(
                self.filename
            )

            self.display_image(
                self.img,
                self.input_label
            )

    # --------------------------------------------------

    def process_image(self):

        if not hasattr(self, 'img'):

            print("Please Load Image First")

            return

        if self.scale_box.text() == "":

            print("Enter Value")

            return

        value = float(
            self.scale_box.text()
        )

        h, w, c = self.img.shape

        option = self.combo.currentText()

        print("Selected:", option)

        channels = []

        # ---------------- NEAREST ----------------

        if option == "Nearest Neighbour":

            for i in range(3):

                ch = nearest_neighbour(
                    self.img[:, :, i],
                    h,
                    w,
                    value
                )

                channels.append(ch)

        # ---------------- BILINEAR ----------------

        elif option == "Bilinear":

            for i in range(3):

                ch = bilinear_interpolation(
                    self.img[:, :, i],
                    h,
                    w,
                    value
                )

                channels.append(ch)

        # ---------------- ROTATION ----------------

        elif option == "Rotation":

            for i in range(3):

                ch = rotation(
                    self.img[:, :, i],
                    h,
                    w,
                    value
                )

                channels.append(ch)

        # ---------------- MERGE CHANNELS ----------------

        output = cv2.merge(channels)

        output = output.astype(
            np.uint8
        )

        self.display_image(
            output,
            self.output_label
        )

    # --------------------------------------------------

    def display_image(self, img, label):

        # ---------------- COLOR IMAGE ----------------

        if len(img.shape) == 3:

            img_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            h, w, ch = img_rgb.shape

            bytes_per_line = ch * w

            qimg = QImage(
                img_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

        # ---------------- GRAYSCALE IMAGE ----------------

        else:

            h, w = img.shape

            bytes_per_line = w

            qimg = QImage(
                img.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_Grayscale8
            )

        pixmap = QPixmap.fromImage(
            qimg
        )

        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            )
        )


# --------------------------------------------------

class Image_info(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "BMP Image Processing"
        )

        self.resize(1400, 750)

        # =====================================================
        # MAIN LAYOUT
        # =====================================================

        self.main_layout = QVBoxLayout()

        # =====================================================
        # TOP BUTTON LAYOUT
        # =====================================================

        self.top_btn_layout = QHBoxLayout()

        # -----------------------------------------------------

        self.input_btn = QPushButton(
            "Input Image"
        )

        self.input_btn.clicked.connect(
            self.open_image
        )

        self.top_btn_layout.addWidget(
            self.input_btn
        )

        # -----------------------------------------------------

        self.read_btn = QPushButton(
            "Read Image Info"
        )

        self.read_btn.clicked.connect(
            self.read_image_info
        )

        self.top_btn_layout.addWidget(
            self.read_btn
        )

        # =====================================================

        self.main_layout.addLayout(
            self.top_btn_layout
        )

        # =====================================================
        # IMAGE LAYOUT
        # =====================================================

        self.image_layout = QHBoxLayout()

        # =====================================================
        # INPUT IMAGE SPACE
        # =====================================================

        self.input_group = QGroupBox(
            "Input Image"
        )

        self.input_layout = QVBoxLayout()

        self.input_label = QLabel()

        self.input_label.setFixedSize(
            600,
            500
        )

        self.input_label.setAlignment(
            Qt.AlignCenter
        )

        self.input_label.setStyleSheet(
            """
            border: 2px solid black;
            background-color: white;
            """
        )

        self.input_layout.addWidget(
            self.input_label
        )

        self.input_group.setLayout(
            self.input_layout
        )

        # =====================================================
        # OUTPUT SPACE
        # =====================================================

        self.output_group = QGroupBox(
            "Output"
        )

        self.output_layout = QVBoxLayout()

        # -----------------------------------------------------
        # OUTPUT IMAGE
        # -----------------------------------------------------

        self.output_label = QLabel()

        self.output_label.setFixedSize(
            600,
            400
        )

        self.output_label.setAlignment(
            Qt.AlignCenter
        )

        self.output_label.setStyleSheet(
            """
            border: 2px solid black;
            background-color: white;
            """
        )

        self.output_layout.addWidget(
            self.output_label
        )

        # =====================================================
        # RGB SLIDER GROUP
        # =====================================================

        self.slider_group = QGroupBox(
            "RGB Channel Manipulation"
        )

        self.slider_layout = QVBoxLayout()

        # =====================================================
        # RED SLIDER
        # =====================================================

        self.red_label = QLabel(
            "Red Channel"
        )

        self.slider_layout.addWidget(
            self.red_label
        )

        self.red_slider = QSlider(
            Qt.Horizontal
        )

        self.red_slider.setMinimum(0)

        self.red_slider.setMaximum(255)

        self.red_slider.setValue(255)

        self.red_slider.valueChanged.connect(
            self.apply_rgb_manipulation
        )

        self.slider_layout.addWidget(
            self.red_slider
        )

        # =====================================================
        # GREEN SLIDER
        # =====================================================

        self.green_label = QLabel(
            "Green Channel"
        )

        self.slider_layout.addWidget(
            self.green_label
        )

        self.green_slider = QSlider(
            Qt.Horizontal
        )

        self.green_slider.setMinimum(0)

        self.green_slider.setMaximum(255)

        self.green_slider.setValue(255)

        self.green_slider.valueChanged.connect(
            self.apply_rgb_manipulation
        )

        self.slider_layout.addWidget(
            self.green_slider
        )

        # =====================================================
        # BLUE SLIDER
        # =====================================================

        self.blue_label = QLabel(
            "Blue Channel"
        )

        self.slider_layout.addWidget(
            self.blue_label
        )

        self.blue_slider = QSlider(
            Qt.Horizontal
        )

        self.blue_slider.setMinimum(0)

        self.blue_slider.setMaximum(255)

        self.blue_slider.setValue(255)

        self.blue_slider.valueChanged.connect(
            self.apply_rgb_manipulation
        )

        self.slider_layout.addWidget(
            self.blue_slider
        )

        # =====================================================
        # RESET BUTTON
        # =====================================================

        self.reset_btn = QPushButton(
            "Reset Channels"
        )

        self.reset_btn.clicked.connect(
            self.reset_channels
        )

        self.slider_layout.addWidget(
            self.reset_btn
        )

        # =====================================================

        self.slider_group.setLayout(
            self.slider_layout
        )

        self.output_layout.addWidget(
            self.slider_group
        )

        # =====================================================
        # OUTPUT TEXT
        # =====================================================

        self.output_text = QTextEdit()

        self.output_text.setReadOnly(
            True
        )

        self.output_layout.addWidget(
            self.output_text
        )

        # =====================================================

        self.output_group.setLayout(
            self.output_layout
        )

        # =====================================================

        self.image_layout.addWidget(
            self.input_group
        )

        self.image_layout.addWidget(
            self.output_group
        )

        # =====================================================

        self.main_layout.addLayout(
            self.image_layout
        )

        self.setLayout(
            self.main_layout
        )

    # =========================================================
    # OPEN IMAGE
    # =========================================================

    def open_image(self):

        self.filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if self.filename:

            # =================================================
            # CONVERT TO BMP IF NOT BMP
            # =================================================

            if not self.filename.lower().endswith(
                ".bmp"
            ):

                img = cv2.imread(
                    self.filename
                )

                base = os.path.splitext(
                    self.filename
                )[0]

                bmp_path = base + ".bmp"

                cv2.imwrite(
                    bmp_path,
                    img
                )

                self.filename = bmp_path

                print(
                    "Image Converted To BMP"
                )

            # =================================================

            self.original_img = cv2.imread(
                self.filename
            )

            self.display_image(
                self.original_img,
                self.input_label
            )

            self.display_image(
                self.original_img,
                self.output_label
            )

            self.output_text.clear()

    # =========================================================
    # DISPLAY IMAGE
    # =========================================================

    def display_image(
        self,
        img,
        label
    ):

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = img_rgb.shape

        bytes_per_line = ch * w

        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            qimg
        )

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        label.setPixmap(
            pixmap
        )

    # =========================================================
    # READ IMAGE INFO
    # =========================================================

    def read_image_info(self):

        if not hasattr(
            self,
            'filename'
        ):

            return

        img_info, img_pixel = img_read(
            self.filename
        )

        self.output_text.clear()

        for key, value in img_info.items():

            if key != "pixel_data":

                self.output_text.append(
                    f"{key} : {value}"
                )

    # =========================================================
    # APPLY RGB MANIPULATION
    # =========================================================

    def apply_rgb_manipulation(self):

        if not hasattr(
            self,
            'original_img'
        ):

            return

        # ---------------------------------------------
        # ALWAYS COPY ORIGINAL IMAGE
        # ---------------------------------------------

        img = self.original_img.copy().astype(
            np.float32
        )

        # ---------------------------------------------
        # GET SLIDER VALUES
        # ---------------------------------------------

        red_value = self.red_slider.value()

        green_value = self.green_slider.value()

        blue_value = self.blue_slider.value()

        # ---------------------------------------------
        # NORMALIZE
        # ---------------------------------------------

        r_scale = red_value / 255.0

        g_scale = green_value / 255.0

        b_scale = blue_value / 255.0

        # ---------------------------------------------
        # APPLY CHANNEL SCALING
        # OpenCV = BGR
        # ---------------------------------------------

        img[:, :, 2] *= r_scale

        img[:, :, 1] *= g_scale

        img[:, :, 0] *= b_scale

        # ---------------------------------------------
        # CONVERT BACK
        # ---------------------------------------------

        img = np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

        # ---------------------------------------------
        # DISPLAY IMAGE
        # ---------------------------------------------

        self.display_image(
            img,
            self.output_label
        )

        # ---------------------------------------------
        # DISPLAY VALUES
        # ---------------------------------------------

        self.output_text.clear()

        self.output_text.append(
            f"Red Channel   : {red_value}"
        )

        self.output_text.append(
            f"Green Channel : {green_value}"
        )

        self.output_text.append(
            f"Blue Channel  : {blue_value}"
        )

    # =========================================================
    # RESET CHANNELS
    # =========================================================

    def reset_channels(self):

        if not hasattr(
            self,
            'original_img'
        ):

            return

        self.red_slider.setValue(
            255
        )

        self.green_slider.setValue(
            255
        )

        self.blue_slider.setValue(
            255
        )

        self.display_image(
            self.original_img,
            self.output_label
        )

        self.output_text.clear()

        self.output_text.append(
            "Channels Reset"
        )

# --------------------------------------------------
# =========================================================
# GUI CLASS
# =========================================================

class MorphologyWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Binary Morphological Operations"
        )

        self.resize(1500, 800)

        # =====================================================
        # MAIN LAYOUT
        # =====================================================

        self.main_layout = QVBoxLayout()

        # =====================================================
        # TOP BUTTON LAYOUT
        # =====================================================

        self.top_layout = QHBoxLayout()

        # -----------------------------------------------------

        self.input_btn = QPushButton(
            "Input Image"
        )

        self.input_btn.clicked.connect(
            self.open_image
        )

        self.top_layout.addWidget(
            self.input_btn
        )

        # -----------------------------------------------------

        self.erode_btn = QPushButton(
            "Erosion"
        )

        self.erode_btn.clicked.connect(
            self.apply_erosion
        )

        self.top_layout.addWidget(
            self.erode_btn
        )

        # -----------------------------------------------------

        self.dilate_btn = QPushButton(
            "Dilation"
        )

        self.dilate_btn.clicked.connect(
            self.apply_dilation
        )

        self.top_layout.addWidget(
            self.dilate_btn
        )

        # -----------------------------------------------------

        self.open_btn = QPushButton(
            "Opening"
        )

        self.open_btn.clicked.connect(
            self.apply_opening
        )

        self.top_layout.addWidget(
            self.open_btn
        )

        # -----------------------------------------------------

        self.close_btn = QPushButton(
            "Closing"
        )

        self.close_btn.clicked.connect(
            self.apply_closing
        )

        self.top_layout.addWidget(
            self.close_btn
        )

        # =====================================================

        self.main_layout.addLayout(
            self.top_layout
        )

        # =====================================================
        # IMAGE DISPLAY LAYOUT
        # =====================================================

        self.image_layout = QHBoxLayout()

        # =====================================================
        # INPUT IMAGE
        # =====================================================

        self.input_group = QGroupBox(
            "Input Binary Image"
        )

        self.input_group_layout = QVBoxLayout()

        self.input_label = QLabel()

        self.input_label.setFixedSize(
            600,
            500
        )

        self.input_label.setAlignment(
            Qt.AlignCenter
        )

        self.input_label.setStyleSheet(
            """
            border: 2px solid black;
            background-color: white;
            """
        )

        self.input_group_layout.addWidget(
            self.input_label
        )

        self.input_group.setLayout(
            self.input_group_layout
        )

        # =====================================================
        # OUTPUT IMAGE
        # =====================================================

        self.output_group = QGroupBox(
            "Output Image"
        )

        self.output_group_layout = QVBoxLayout()

        self.output_label = QLabel()

        self.output_label.setFixedSize(
            600,
            500
        )

        self.output_label.setAlignment(
            Qt.AlignCenter
        )

        self.output_label.setStyleSheet(
            """
            border: 2px solid black;
            background-color: white;
            """
        )

        self.output_group_layout.addWidget(
            self.output_label
        )

        # =====================================================
        # STRUCTURING ELEMENT SIZE
        # =====================================================

        self.slider_label = QLabel(
            "Structuring Element Size"
        )

        self.output_group_layout.addWidget(
            self.slider_label
        )

        self.se_slider = QSlider(
            Qt.Horizontal
        )

        self.se_slider.setMinimum(3)

        self.se_slider.setMaximum(15)

        self.se_slider.setValue(3)

        self.se_slider.setSingleStep(2)

        self.output_group_layout.addWidget(
            self.se_slider
        )

        # =====================================================
        # TEXT OUTPUT
        # =====================================================

        self.output_text = QTextEdit()

        self.output_text.setReadOnly(
            True
        )

        self.output_group_layout.addWidget(
            self.output_text
        )

        # =====================================================

        self.output_group.setLayout(
            self.output_group_layout
        )

        # =====================================================

        self.image_layout.addWidget(
            self.input_group
        )

        self.image_layout.addWidget(
            self.output_group
        )

        # =====================================================

        self.main_layout.addLayout(
            self.image_layout
        )

        self.setLayout(
            self.main_layout
        )

    # =========================================================
    # OPEN IMAGE
    # =========================================================

    def open_image(self):

        self.filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if self.filename:

            img = cv2.imread(
                self.filename,
                cv2.IMREAD_GRAYSCALE
            )

            # -------------------------------------------------
            # BINARY CONVERSION
            # -------------------------------------------------

            _, self.binary_img = cv2.threshold(
                img,
                127,
                1,
                cv2.THRESH_BINARY
            )

            self.display_image(
                self.binary_img,
                self.input_label
            )

            self.display_image(
                self.binary_img,
                self.output_label
            )

            self.output_text.clear()

            self.output_text.append(
                "Binary Image Loaded"
            )

    # =========================================================
    # DISPLAY IMAGE
    # =========================================================

    def display_image(
        self,
        img,
        label
    ):

        display_img = (
            img * 255
        ).astype(np.uint8)

        h, w = display_img.shape

        bytes_per_line = w

        qimg = QImage(
            display_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(
            qimg
        )

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        label.setPixmap(
            pixmap
        )

    # =========================================================
    # STRUCTURING ELEMENT
    # =========================================================

    def get_structuring_element(self):

        size = self.se_slider.value()

        # FORCE ODD SIZE

        if size % 2 == 0:

            size += 1

        se = np.ones(
            (size, size),
            dtype=np.uint8
        )

        return se, size

    # =========================================================
    # EROSION
    # =========================================================

    def apply_erosion(self):

        if not hasattr(
            self,
            'binary_img'
        ):

            return

        se, size = self.get_structuring_element()

        result = ErodeBinary(
            self.binary_img,
            se
        )

        self.display_image(
            result,
            self.output_label
        )

        self.output_text.clear()

        self.output_text.append(
            f"Erosion Applied"
        )

        self.output_text.append(
            f"Structuring Element Size : {size}x{size}"
        )

    # =========================================================
    # DILATION
    # =========================================================

    def apply_dilation(self):

        if not hasattr(
            self,
            'binary_img'
        ):

            return

        se, size = self.get_structuring_element()

        result = DilateBinary(
            self.binary_img,
            se
        )

        self.display_image(
            result,
            self.output_label
        )

        self.output_text.clear()

        self.output_text.append(
            f"Dilation Applied"
        )

        self.output_text.append(
            f"Structuring Element Size : {size}x{size}"
        )

    # =========================================================
    # OPENING
    # =========================================================

    def apply_opening(self):

        if not hasattr(
            self,
            'binary_img'
        ):

            return

        se, size = self.get_structuring_element()

        result = OpenBinary(
            self.binary_img,
            se
        )

        self.display_image(
            result,
            self.output_label
        )

        self.output_text.clear()

        self.output_text.append(
            f"Opening Applied"
        )

        self.output_text.append(
            f"Structuring Element Size : {size}x{size}"
        )

    # =========================================================
    # CLOSING
    # =========================================================

    def apply_closing(self):

        if not hasattr(
            self,
            'binary_img'
        ):

            return

        se, size = self.get_structuring_element()

        result = CloseBinary(
            self.binary_img,
            se
        )

        self.display_image(
            result,
            self.output_label
        )

        self.output_text.clear()

        self.output_text.append(
            f"Closing Applied"
        )

        self.output_text.append(
            f"Structuring Element Size : {size}x{size}"
        )

# ------------------------------------------------------

class ImageFilter(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Spatial Filtering GUI"
        )

        self.resize(1300, 750)

        self.image = None

        self.filtered_image = None

        self.initUI()

    # =====================================================

    def initUI(self):

        tabs = QTabWidget()

        filter_tab = QWidget()

        tabs.addTab(
            filter_tab,
            "Spatial Filters"
        )

        self.setCentralWidget(tabs)

        # =================================================
        # IMAGE LABELS
        # =================================================

        self.original_label = QLabel(
            "Original Image"
        )

        self.filtered_label = QLabel(
            "Filtered Image"
        )

        self.original_label.setAlignment(
            Qt.AlignCenter
        )

        self.filtered_label.setAlignment(
            Qt.AlignCenter
        )

        self.original_label.setFixedSize(
            550,
            550
        )

        self.filtered_label.setFixedSize(
            550,
            550
        )

        self.original_label.setStyleSheet(
            """
            border:2px solid black;
            background:white;
            """
        )

        self.filtered_label.setStyleSheet(
            """
            border:2px solid black;
            background:white;
            """
        )

        # =================================================
        # BUTTONS
        # =================================================

        open_btn = QPushButton(
            "Open Image"
        )

        apply_btn = QPushButton(
            "Apply Filter"
        )

        open_btn.clicked.connect(
            self.load_image
        )

        apply_btn.clicked.connect(
            self.apply_filter
        )

        # =================================================
        # FILTER COMBO
        # =================================================

        self.filter_combo = QComboBox()

        self.filter_combo.addItems([
            "Mean Filter",
            "Median Filter",
            "Gaussian Blur",
            "Gaussian Unblur",
            "Laplacian"
        ])

        # =================================================
        # KERNEL SIZE
        # =================================================

        self.kernel_slider = QSlider(
            Qt.Horizontal
        )

        self.kernel_slider.setMinimum(3)

        self.kernel_slider.setMaximum(21)

        self.kernel_slider.setValue(5)

        self.kernel_slider.setTickInterval(2)

        self.kernel_slider.valueChanged.connect(
            self.update_kernel_label
        )

        self.kernel_label = QLabel(
            "Kernel Size: 5"
        )

        # =================================================
        # SIGMA
        # =================================================

        self.sigma_slider = QSlider(
            Qt.Horizontal
        )

        self.sigma_slider.setMinimum(1)

        self.sigma_slider.setMaximum(100)

        self.sigma_slider.setValue(15)

        self.sigma_slider.valueChanged.connect(
            self.update_sigma_label
        )

        self.sigma_label = QLabel(
            "Sigma: 1.5"
        )

        # =================================================
        # ITERATIONS
        # =================================================

        self.iter_slider = QSlider(
            Qt.Horizontal
        )

        self.iter_slider.setMinimum(1)

        self.iter_slider.setMaximum(50)

        self.iter_slider.setValue(10)

        self.iter_slider.valueChanged.connect(
            self.update_iter_label
        )

        self.iter_label = QLabel(
            "Iterations: 10"
        )

        # =================================================
        # CONTROL LAYOUT
        # =================================================

        control_layout = QGridLayout()

        control_layout.addWidget(
            open_btn,
            0,
            0
        )

        control_layout.addWidget(
            apply_btn,
            0,
            1
        )

        control_layout.addWidget(
            QLabel("Filter:"),
            1,
            0
        )

        control_layout.addWidget(
            self.filter_combo,
            1,
            1
        )

        control_layout.addWidget(
            self.kernel_label,
            2,
            0
        )

        control_layout.addWidget(
            self.kernel_slider,
            2,
            1
        )

        control_layout.addWidget(
            self.sigma_label,
            3,
            0
        )

        control_layout.addWidget(
            self.sigma_slider,
            3,
            1
        )

        control_layout.addWidget(
            self.iter_label,
            4,
            0
        )

        control_layout.addWidget(
            self.iter_slider,
            4,
            1
        )

        # =================================================
        # IMAGE LAYOUT
        # =================================================

        image_layout = QHBoxLayout()

        image_layout.addWidget(
            self.original_label
        )

        image_layout.addWidget(
            self.filtered_label
        )

        # =================================================
        # MAIN LAYOUT
        # =================================================

        main_layout = QVBoxLayout()

        main_layout.addLayout(
            control_layout
        )

        main_layout.addLayout(
            image_layout
        )

        filter_tab.setLayout(main_layout)

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp)"
        )

        if file_name:

            self.image = cv2.imread(
                file_name,
                cv2.IMREAD_GRAYSCALE
            )

            self.display_image(
                self.image,
                self.original_label
            )

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def display_image(self, img, label):

        h, w = img.shape

        bytes_per_line = w

        qimg = QImage(
            img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio
        )

        label.setPixmap(pixmap)

    # =====================================================
    # APPLY FILTER
    # =====================================================

    def apply_filter(self):

        if self.image is None:
            return

        filter_name = self.filter_combo.currentText()

        ksize = self.kernel_slider.value()

        if ksize % 2 == 0:
            ksize += 1

        sigma = self.sigma_slider.value() / 10

        iterations = self.iter_slider.value()

        # =============================================

        if filter_name == "Mean Filter":

            self.filtered_image = mean_filter(
                self.image,
                ksize
            )

        elif filter_name == "Median Filter":

            self.filtered_image = median_filter(
                self.image,
                ksize
            )

        elif filter_name == "Gaussian Blur":

            self.filtered_image = gaussian_blur(
                self.image,
                ksize,
                sigma
            )

        elif filter_name == "Gaussian Unblur":

            self.filtered_image = gaussian_unblur(
                self.image,
                ksize,
                sigma,
                max_iter=iterations,
                verbose=True
            )

        elif filter_name == "Laplacian":

            self.filtered_image = laplacian_filter(
                self.image
            )

        # =============================================

        self.display_image(
            self.filtered_image,
            self.filtered_label
        )

    # =====================================================
    # UPDATE LABELS
    # =====================================================

    def update_kernel_label(self):

        self.kernel_label.setText(
            f"Kernel Size: {self.kernel_slider.value()}"
        )

    def update_sigma_label(self):

        sigma = self.sigma_slider.value() / 10

        self.sigma_label.setText(
            f"Sigma: {sigma}"
        )

    def update_iter_label(self):

        self.iter_label.setText(
            f"Iterations: {self.iter_slider.value()}"
        )

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Spatial Filtering GUI")

        self.resize(1200, 700)

        self.image = None
        self.filtered_image = None

        self.initUI()

    def initUI(self):

        tabs = QTabWidget()

        filter_tab = QWidget()

        tabs.addTab(filter_tab, "Spatial Filters")

        self.setCentralWidget(tabs)

        # ---------- IMAGE LABELS ---------- #

        self.original_label = QLabel("Original Image")
        self.filtered_label = QLabel("Filtered Image")

        self.original_label.setAlignment(Qt.AlignCenter)
        self.filtered_label.setAlignment(Qt.AlignCenter)

        self.original_label.setFixedSize(500, 500)
        self.filtered_label.setFixedSize(500, 500)

        self.original_label.setStyleSheet("border:2px solid black;")
        self.filtered_label.setStyleSheet("border:2px solid black;")

        # ---------- BUTTONS ---------- #

        open_btn = QPushButton("Open Image")
        apply_btn = QPushButton("Apply Filter")

        open_btn.clicked.connect(self.load_image)
        apply_btn.clicked.connect(self.apply_filter)

        # ---------- FILTER OPTIONS ---------- #

        self.filter_combo = QComboBox()

        self.filter_combo.addItems([
            "Mean Filter",
            "Median Filter",
            "Gaussian Blur",
            "Laplacian"
        ])

        # ---------- KERNEL SLIDER ---------- #

        self.kernel_slider = QSlider(Qt.Horizontal)

        self.kernel_slider.setMinimum(3)
        self.kernel_slider.setMaximum(21)
        self.kernel_slider.setValue(5)

        self.kernel_slider.setTickInterval(2)

        self.kernel_slider.valueChanged.connect(self.update_kernel_label)

        self.kernel_label = QLabel("Kernel Size: 5")

        # ---------- SIGMA SLIDER ---------- #

        self.sigma_slider = QSlider(Qt.Horizontal)

        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(100)

        self.sigma_slider.setValue(15)

        self.sigma_slider.valueChanged.connect(self.update_sigma_label)

        self.sigma_label = QLabel("Sigma: 1.5")

        # ---------- CONTROL LAYOUT ---------- #

        control_layout = QGridLayout()

        control_layout.addWidget(open_btn, 0, 0)

        control_layout.addWidget(apply_btn, 0, 1)

        control_layout.addWidget(QLabel("Filter:"), 1, 0)
        control_layout.addWidget(self.filter_combo, 1, 1)

        control_layout.addWidget(self.kernel_label, 2, 0)
        control_layout.addWidget(self.kernel_slider, 2, 1)

        control_layout.addWidget(self.sigma_label, 3, 0)
        control_layout.addWidget(self.sigma_slider, 3, 1)

        # ---------- IMAGE LAYOUT ---------- #

        image_layout = QHBoxLayout()

        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.filtered_label)

        # ---------- MAIN LAYOUT ---------- #

        main_layout = QVBoxLayout()

        main_layout.addLayout(control_layout)
        main_layout.addLayout(image_layout)

        filter_tab.setLayout(main_layout)

    # ---------------- LOAD IMAGE ---------------- #

    def load_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp)"
        )

        if file_name:

            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

            self.display_image(self.image, self.original_label)

    # ---------------- DISPLAY IMAGE ---------------- #

    def display_image(self, img, label):

        h, w = img.shape

        bytes_per_line = w

        qimg = QImage(
            img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio
        )

        label.setPixmap(pixmap)

    # ---------------- APPLY FILTER ---------------- #

    def apply_filter(self):

        if self.image is None:
            return

        filter_name = self.filter_combo.currentText()

        ksize = self.kernel_slider.value()

        if ksize % 2 == 0:
            ksize += 1

        sigma = self.sigma_slider.value() / 10

        if filter_name == "Mean Filter":

            self.filtered_image = mean_filter(self.image, ksize)

        elif filter_name == "Median Filter":

            self.filtered_image = median_filter(self.image, ksize)

        elif filter_name == "Gaussian Blur":

            self.filtered_image = gaussian_blur(
                self.image,
                ksize,
                sigma
            )

        elif filter_name == "Laplacian":

            self.filtered_image = laplacian_filter(self.image)

        self.display_image(
            self.filtered_image,
            self.filtered_label
        )

    # ---------------- LABEL UPDATES ---------------- #

    def update_kernel_label(self):

        self.kernel_label.setText(
            f"Kernel Size: {self.kernel_slider.value()}"
        )

    def update_sigma_label(self):

        sigma = self.sigma_slider.value() / 10

        self.sigma_label.setText(f"Sigma: {sigma}")

# -----------------------------------------------------------

class FFTApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "FFT Image Processing GUI"
        )

        self.resize(1600, 900)

        self.image = None

        self.fft_result = None

        self.initUI()

    # =====================================================

    def initUI(self):

        tabs = QTabWidget()

        fft_tab = QWidget()

        tabs.addTab(
            fft_tab,
            "FFT Analysis"
        )

        self.setCentralWidget(tabs)

        # =================================================
        # FFT TAB
        # =================================================

        self.original_label = self.create_image_label(
            "Original Image"
        )

        self.magnitude_label = self.create_image_label(
            "Magnitude Spectrum"
        )

        self.phase_label = self.create_image_label(
            "Phase Spectrum"
        )

        self.reconstructed_label = self.create_image_label(
            "Reconstructed Image"
        )

        # =============================================
        # BUTTONS
        # =============================================

        open_btn = QPushButton(
            "Open Image"
        )

        run_btn = QPushButton(
            "Run FFT"
        )

        open_btn.clicked.connect(
            self.load_image
        )

        run_btn.clicked.connect(
            self.run_fft
        )

        # =============================================
        # TOP LAYOUT
        # =============================================

        top_layout = QHBoxLayout()

        top_layout.addWidget(open_btn)

        top_layout.addWidget(run_btn)

        # =============================================
        # IMAGE GRID
        # =============================================

        grid = QGridLayout()

        grid.addWidget(
            self.original_label,
            0,
            0
        )

        grid.addWidget(
            self.magnitude_label,
            0,
            1
        )

        grid.addWidget(
            self.phase_label,
            1,
            0
        )

        grid.addWidget(
            self.reconstructed_label,
            1,
            1
        )

        # =============================================
        # MAIN LAYOUT
        # =============================================

        fft_layout = QVBoxLayout()

        fft_layout.addLayout(top_layout)

        fft_layout.addLayout(grid)

        fft_tab.setLayout(fft_layout)

    # =====================================================
    # IMAGE LABEL CREATOR
    # =====================================================

    def create_image_label(self, text):

        label = QLabel(text)

        label.setAlignment(Qt.AlignCenter)

        label.setFixedSize(650, 350)

        label.setStyleSheet(
            """
            border:2px solid black;
            background:white;
            """
        )

        return label

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )

        if file_name:

            self.image = cv2.imread(
                file_name,
                cv2.IMREAD_GRAYSCALE
            )

            self.display_image(
                self.image,
                self.original_label
            )

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def display_image(self, img, label):

        img = np.ascontiguousarray(img)

        if img.dtype != np.uint8:

            img = np.clip(
                img,
                0,
                255
            ).astype(np.uint8)

        # =============================================
        # GRAYSCALE
        # =============================================

        if len(img.shape) == 2:

            h, w = img.shape

            bytes_per_line = w

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format_Grayscale8
            )

        # =============================================
        # COLOR
        # =============================================

        else:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            h, w, ch = img.shape

            bytes_per_line = ch * w

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

        # =============================================
        # DISPLAY
        # =============================================

        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio
        )

        label.setPixmap(pixmap)

    # =====================================================
    # RUN FFT
    # =====================================================

    def run_fft(self):

        if self.image is None:
            return

        img = np.float32(self.image)

        # =============================================
        # FFT
        # =============================================

        self.fft_result = fft2d(img)

        magnitude, phase = magnitude_phase_spectrum(
            self.fft_result
        )

        # =============================================
        # MAGNITUDE SPECTRUM
        # =============================================

        mag_spectrum = np.log(
            1 + magnitude
        )

        mag_spectrum = fftshift(
            mag_spectrum
        )

        mag_spectrum = cv2.normalize(
            mag_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # =============================================
        # PHASE SPECTRUM
        # =============================================

        phase_spectrum = fftshift(
            phase
        )

        phase_spectrum = cv2.normalize(
            phase_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # =============================================
        # RECONSTRUCTION
        # =============================================

        reconstructed = ifft2d(
            self.fft_result
        )

        reconstructed = reconstructed[
            :self.image.shape[0],
            :self.image.shape[1]
        ]

        reconstructed = np.clip(
            reconstructed,
            0,
            255
        ).astype(np.uint8)

        # =============================================
        # DISPLAY
        # =============================================

        self.display_image(
            mag_spectrum,
            self.magnitude_label
        )

        self.display_image(
            phase_spectrum,
            self.phase_label
        )

        self.display_image(
            reconstructed,
            self.reconstructed_label
        )

# -----------------------------------------------------------

class FrequencyFilteringApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Frequency Domain Filtering GUI"
        )

        self.resize(1800, 950)

        self.image_path = None

        self.hybrid_path1 = None

        self.hybrid_path2 = None

        self.initUI()

    # =====================================================
    # UI
    # =====================================================

    def initUI(self):

        tabs = QTabWidget()

        filter_tab = QWidget()

        hybrid_tab = QWidget()

        denoise_tab = QWidget()

        tabs.addTab(
            filter_tab,
            "Frequency Filters"
        )

        tabs.addTab(
            hybrid_tab,
            "Hybrid Image"
        )

        self.setCentralWidget(tabs)

        # =================================================
        # FILTER TAB
        # =================================================

        self.original_label = self.create_label(
            "Original Image"
        )

        self.spectrum_label = self.create_label(
            "Spectrum"
        )

        self.mask_label = self.create_label(
            "Filter Mask"
        )

        self.result_label = self.create_label(
            "Filtered Output"
        )

        # =============================================
        # BUTTONS
        # =============================================

        open_btn = QPushButton(
            "Open Image"
        )

        apply_btn = QPushButton(
            "Apply Filter"
        )

        open_btn.clicked.connect(
            self.load_image
        )

        apply_btn.clicked.connect(
            self.apply_frequency_filter
        )

        # =============================================
        # FILTER COMBO
        # =============================================

        self.filter_combo = QComboBox()

        self.filter_combo.addItems([
            "Ideal LPF",
            "Ideal HPF",
            "Gaussian LPF",
            "Gaussian HPF",
            "Butterworth LPF",
            "Butterworth HPF"
        ])

        # =============================================
        # CUTOFF SLIDER
        # =============================================

        self.cutoff_slider = QSlider(
            Qt.Horizontal
        )

        self.cutoff_slider.setMinimum(1)

        self.cutoff_slider.setMaximum(200)

        self.cutoff_slider.setValue(30)

        self.cutoff_slider.valueChanged.connect(
            self.update_cutoff_label
        )

        self.cutoff_label = QLabel(
            "Cutoff: 30"
        )

        # =============================================
        # ORDER SLIDER
        # =============================================

        self.order_slider = QSlider(
            Qt.Horizontal
        )

        self.order_slider.setMinimum(1)

        self.order_slider.setMaximum(10)

        self.order_slider.setValue(2)

        self.order_slider.valueChanged.connect(
            self.update_order_label
        )

        self.order_label = QLabel(
            "Order: 2"
        )

        # =============================================
        # CONTROLS
        # =============================================

        controls = QGridLayout()

        controls.addWidget(
            open_btn,
            0,
            0
        )

        controls.addWidget(
            apply_btn,
            0,
            1
        )

        controls.addWidget(
            QLabel("Filter Type"),
            1,
            0
        )

        controls.addWidget(
            self.filter_combo,
            1,
            1
        )

        controls.addWidget(
            self.cutoff_label,
            2,
            0
        )

        controls.addWidget(
            self.cutoff_slider,
            2,
            1
        )

        controls.addWidget(
            self.order_label,
            3,
            0
        )

        controls.addWidget(
            self.order_slider,
            3,
            1
        )

        # =============================================
        # IMAGE GRID
        # =============================================

        grid = QGridLayout()

        grid.addWidget(
            self.original_label,
            0,
            0
        )

        grid.addWidget(
            self.spectrum_label,
            0,
            1
        )

        grid.addWidget(
            self.mask_label,
            1,
            0
        )

        grid.addWidget(
            self.result_label,
            1,
            1
        )

        layout = QVBoxLayout()

        layout.addLayout(
            controls
        )

        layout.addLayout(
            grid
        )

        filter_tab.setLayout(layout)

        # =================================================
        # HYBRID TAB
        # =================================================

        self.hybrid_input1 = self.create_label(
            "Input Image 1"
        )

        self.hybrid_input2 = self.create_label(
            "Input Image 2"
        )

        self.hybrid_output = self.create_label(
            "Hybrid Output"
        )

        self.hybrid_type_combo = QComboBox()

        self.hybrid_type_combo.addItems([
            "Type 1 : Image1 HPF + Image2 LPF",
            "Type 2 : Image1 LPF + Image2 HPF"
        ])

        # =============================================
        # BUTTONS
        # =============================================

        load_img1_btn = QPushButton(
            "Load Image 1"
        )

        load_img2_btn = QPushButton(
            "Load Image 2"
        )

        hybrid_btn = QPushButton(
            "Create Hybrid Image"
        )

        load_img1_btn.clicked.connect(
            self.load_hybrid_image1
        )

        load_img2_btn.clicked.connect(
            self.load_hybrid_image2
        )

        hybrid_btn.clicked.connect(
            self.run_hybrid_image
        )

        # =============================================
        # CONTROLS
        # =============================================

        hybrid_controls = QHBoxLayout()

        hybrid_controls.addWidget(
            load_img1_btn
        )

        hybrid_controls.addWidget(
            load_img2_btn
        )

        hybrid_controls.addWidget(
            self.hybrid_type_combo
        )

        hybrid_controls.addWidget(
            hybrid_btn
        )

        # =============================================
        # IMAGE GRID
        # =============================================

        hybrid_grid = QGridLayout()

        hybrid_grid.addWidget(
            self.hybrid_input1,
            0,
            0
        )

        hybrid_grid.addWidget(
            self.hybrid_input2,
            0,
            1
        )

        hybrid_grid.addWidget(
            self.hybrid_output,
            1,
            0,
            1,
            2
        )

        # =============================================
        # MAIN LAYOUT
        # =============================================

        hybrid_layout = QVBoxLayout()

        hybrid_layout.addLayout(
            hybrid_controls
        )

        hybrid_layout.addLayout(
            hybrid_grid
        )

        hybrid_tab.setLayout(
            hybrid_layout
        )


    # =====================================================
    # LABEL CREATOR
    # =====================================================

    def create_label(self, text):

        label = QLabel(text)

        label.setAlignment(Qt.AlignCenter)

        label.setFixedSize(700, 350)

        label.setStyleSheet(
            """
            border:2px solid black;
            background:white;
            """
        )

        return label

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )

        if file_name:

            self.image_path = file_name

            img = cv2.imread(
                file_name,
                cv2.IMREAD_GRAYSCALE
            )

            self.display_image(
                img,
                self.original_label
            )

    # =====================================================
    # HYBRID IMAGE LOADERS
    # =====================================================

    def load_hybrid_image1(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select First Image",
            "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )

        if file_name:

            self.hybrid_path1 = file_name

            img = cv2.imread(
                file_name,
                cv2.IMREAD_GRAYSCALE
            )

            self.display_image(
                img,
                self.hybrid_input1
            )

    def load_hybrid_image2(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Second Image",
            "",
            "Images (*.png *.jpg *.bmp *.tif)"
        )

        if file_name:

            self.hybrid_path2 = file_name

            img = cv2.imread(
                file_name,
                cv2.IMREAD_GRAYSCALE
            )

            self.display_image(
                img,
                self.hybrid_input2
            )

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def display_image(self, img, label):

        img = np.ascontiguousarray(img)

        img = np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

        h, w = img.shape

        bytes_per_line = w

        qimg = QImage(
            img.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio
        )

        label.setPixmap(pixmap)

    # =====================================================
    # APPLY FILTER
    # =====================================================

    def apply_frequency_filter(self):

        if self.image_path is None:
            return

        cutoff = self.cutoff_slider.value()

        order = self.order_slider.value()

        filter_name = self.filter_combo.currentText()

        # =============================================
        # FILTER CALLS
        # =============================================

        if filter_name == "Ideal LPF":

            spectrum, mask, result = ideal_lpf(
                self.image_path,
                cutoff
            )

        elif filter_name == "Ideal HPF":

            spectrum, mask, result = ideal_hpf(
                self.image_path,
                cutoff
            )

        elif filter_name == "Gaussian LPF":

            spectrum, mask, result = gaussian_lpf(
                self.image_path,
                cutoff
            )

        elif filter_name == "Gaussian HPF":

            spectrum, mask, result = gaussian_hpf(
                self.image_path,
                cutoff
            )

        elif filter_name == "Butterworth LPF":

            spectrum, mask, result = butterworth_lpf(
                self.image_path,
                cutoff,
                order
            )

        elif filter_name == "Butterworth HPF":

            spectrum, mask, result = butterworth_hpf(
                self.image_path,
                cutoff,
                order
            )

        # =============================================
        # DISPLAY
        # =============================================

        self.display_image(
            spectrum,
            self.spectrum_label
        )

        self.display_image(
            mask,
            self.mask_label
        )

        self.display_image(
            result,
            self.result_label
        )

    # =====================================================
    # HYBRID IMAGE
    # =====================================================

    def run_hybrid_image(self):

        if self.hybrid_path1 is None:
            return

        if self.hybrid_path2 is None:
            return

        hybrid_type = self.hybrid_type_combo.currentText()

        mode = "type1"

        if "Type 2" in hybrid_type:

            mode = "type2"

        hybrid = hybrid_image(
            self.hybrid_path1,
            self.hybrid_path2,
            cutoff=self.cutoff_slider.value(),
            mode=mode
        )

        self.display_image(
            hybrid,
            self.hybrid_output
        )

    # =====================================================
    # DENOISING
    # =====================================================



    # =====================================================
    # LABEL UPDATES
    # =====================================================

    def update_cutoff_label(self):

        self.cutoff_label.setText(
            f"Cutoff: {self.cutoff_slider.value()}"
        )

    def update_order_label(self):

        self.order_label.setText(
            f"Order: {self.order_slider.value()}"
        )

# -----------------------------------------------------------

class ImageProcessingGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Histogram Processing")
        self.resize(1200, 700)

        self.tabs = QTabWidget()

        self.equalization_tab()
        self.matching_tab()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        self.setLayout(main_layout)

    # =====================================================
    # ---------------- HISTOGRAM EQUALIZATION -------------
    # =====================================================

    def equalization_tab(self):

        tab = QWidget()

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()

        self.eq_btn = QPushButton("Load Image")
        self.eq_btn.clicked.connect(self.load_equalization_image)

        btn_layout.addWidget(self.eq_btn)

        self.eq_input_label = QLabel("Input Image")
        self.eq_output_label = QLabel("Equalized Image")

        self.eq_input_label.setFixedSize(500, 400)
        self.eq_output_label.setFixedSize(500, 400)

        self.eq_input_label.setStyleSheet("border:2px solid black")
        self.eq_output_label.setStyleSheet("border:2px solid black")

        img_layout = QHBoxLayout()

        img_layout.addWidget(self.eq_input_label)
        img_layout.addWidget(self.eq_output_label)

        layout.addLayout(btn_layout)
        layout.addLayout(img_layout)

        tab.setLayout(layout)

        self.tabs.addTab(tab, "Histogram Equalization")

    # =====================================================
    # ---------------- HISTOGRAM MATCHING -----------------
    # =====================================================

    def matching_tab(self):

        tab = QWidget()

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()

        self.src_btn = QPushButton("Load Source")
        self.tgt_btn = QPushButton("Load Target")

        self.src_btn.clicked.connect(self.load_source)
        self.tgt_btn.clicked.connect(self.load_target)

        btn_layout.addWidget(self.src_btn)
        btn_layout.addWidget(self.tgt_btn)

        self.src_label = QLabel("Source")
        self.tgt_label = QLabel("Target")
        self.match_label = QLabel("Matched")

        labels = [self.src_label, self.tgt_label, self.match_label]

        for lbl in labels:
            lbl.setFixedSize(350, 350)
            lbl.setStyleSheet("border:2px solid black")

        img_layout = QHBoxLayout()

        img_layout.addWidget(self.src_label)
        img_layout.addWidget(self.tgt_label)
        img_layout.addWidget(self.match_label)

        layout.addLayout(btn_layout)
        layout.addLayout(img_layout)

        tab.setLayout(layout)

        self.tabs.addTab(tab, "Histogram Matching")

    # =====================================================
    # ---------------- IMAGE DISPLAY FUNCTION -------------
    # =====================================================

    def display_image(self, label, img):

        if len(img.shape) == 2:

            h, w = img.shape

            qimg = QImage(
                img.data,
                w,
                h,
                w,
                QImage.Format_Grayscale8
            )

        else:

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb.shape

            qimg = QImage(
                rgb.data,
                w,
                h,
                ch * w,
                QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimg)

        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            )
        )

    # =====================================================
    # ------------- LOAD IMAGE FOR EQUALIZATION -----------
    # =====================================================

    def load_equalization_image(self):

        path, _ = QFileDialog.getOpenFileName()

        if not path:
            return

        img = cv2.imread(path)

        if img is None:
            QMessageBox.warning(self, "Error", "Cannot Load Image")
            return

        result = equalize_color(img)

        self.display_image(self.eq_input_label, img)
        self.display_image(self.eq_output_label, result)

    # =====================================================
    # ---------------- LOAD SOURCE IMAGE ------------------
    # =====================================================

    def load_source(self):

        path, _ = QFileDialog.getOpenFileName()

        if not path:
            return

        self.source_img = cv2.imread(path)

        self.display_image(self.src_label, self.source_img)

        self.try_matching()

    # =====================================================
    # ---------------- LOAD TARGET IMAGE ------------------
    # =====================================================

    def load_target(self):

        path, _ = QFileDialog.getOpenFileName()

        if not path:
            return

        self.target_img = cv2.imread(path)

        self.display_image(self.tgt_label, self.target_img)

        self.try_matching()

    # =====================================================
    # ---------------- HISTOGRAM MATCHING -----------------
    # =====================================================

    def try_matching(self):

        if hasattr(self, 'source_img') and hasattr(self, 'target_img'):

            matched = match_histogram_color(
                self.source_img,
                self.target_img
            )

            self.display_image(self.match_label, matched)

# -----------------------------------------------------------

class JPEGCompressor(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("JPEG Compression Simulator")

        self.resize(1400, 800)

        self.image = None

        self.setup_ui()

    # =====================================================
    # UI
    # =====================================================

    def setup_ui(self):

        main_layout = QVBoxLayout()

        title = QLabel("JPEG Compression")

        title.setAlignment(Qt.AlignCenter)

        title.setFont(QFont("Arial", 20, QFont.Bold))

        title.setStyleSheet("""
            padding:15px;
            background:#222;
            color:white;
            border-radius:10px;
        """)

        main_layout.addWidget(title)

        # =================================================
        # BUTTONS
        # =================================================

        btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Image")

        self.compress_btn = QPushButton("Compress Image")

        self.save_btn = QPushButton("Save Compressed File")

        self.load_btn.clicked.connect(self.load_image)

        self.compress_btn.clicked.connect(self.compress_image)

        self.save_btn.clicked.connect(self.save_file)

        buttons = [
            self.load_btn,
            self.compress_btn,
            self.save_btn
        ]

        for btn in buttons:

            btn.setMinimumHeight(45)

            btn.setStyleSheet("""
                QPushButton{
                    background:#3498db;
                    color:white;
                    border:none;
                    border-radius:8px;
                    font-size:16px;
                }

                QPushButton:hover{
                    background:#2980b9;
                }
            """)

            btn_layout.addWidget(btn)

        main_layout.addLayout(btn_layout)

        # =================================================
        # IMAGE DISPLAY
        # =================================================

        image_layout = QHBoxLayout()

        self.input_label = QLabel("Input Image")

        self.output_label = QLabel("Compressed Preview")

        labels = [self.input_label, self.output_label]

        for lbl in labels:

            lbl.setFixedSize(600, 450)

            lbl.setAlignment(Qt.AlignCenter)

            lbl.setStyleSheet("""
                border:2px solid gray;
                background:#f0f0f0;
                font-size:18px;
            """)

            image_layout.addWidget(lbl)

        main_layout.addLayout(image_layout)

        # =================================================
        # STATISTICS PANEL
        # =================================================

        stats_frame = QFrame()

        stats_frame.setStyleSheet("""
            QFrame{
                background:white;
                border:2px solid #cccccc;
                border-radius:10px;
            }
        """)

        stats_layout = QVBoxLayout()

        stats_title = QLabel("Compression Statistics")

        stats_title.setFont(QFont("Arial", 14, QFont.Bold))

        self.stats_text = QTextEdit()

        self.stats_text.setReadOnly(True)

        stats_layout.addWidget(stats_title)

        stats_layout.addWidget(self.stats_text)

        stats_frame.setLayout(stats_layout)

        main_layout.addWidget(stats_frame)

        self.setLayout(main_layout)

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def display_image(self, label, img):

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        qimg = QImage(
            rgb.data,
            w,
            h,
            ch*w,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimg)

        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            )
        )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if not path:
            return

        self.image = cv2.imread(path)

        if self.image is None:

            QMessageBox.warning(
                self,
                "Error",
                "Cannot load image"
            )

            return

        self.display_image(self.input_label, self.image)

        self.stats_text.clear()

    # =====================================================
    # COMPRESS IMAGE
    # =====================================================

    def compress_image(self):

        if self.image is None:

            QMessageBox.warning(
                self,
                "Error",
                "Load an image first"
            )

            return

        img = self.image.copy()

        # RGB -> YCbCr

        Y, Cb, Cr = rgb2ycc(img)

        # Chroma Subsampling

        Cb = chroma_subsample(Cb)
        Cr = chroma_subsample(Cr)

        # Zero Centering

        Y = zero_center(Y)
        Cb = zero_center(Cb)
        Cr = zero_center(Cr)

        # Block Processing

        Y_blocks = process_channel(Y, LUM)
        Cb_blocks = process_channel(Cb, CHR)
        Cr_blocks = process_channel(Cr, CHR)

        # Run Length Encoding

        Y_rle = run_length_encode(Y_blocks)
        Cb_rle = run_length_encode(Cb_blocks)
        Cr_rle = run_length_encode(Cr_blocks)

        all_stream = Y_rle + Cb_rle + Cr_rle

        # Huffman Coding

        tree = build_huffman_tree(all_stream)

        codes = generate_codes(tree)

        bitstream = ""

        for symbol in all_stream:

            bitstream += codes[symbol]

        # Convert to bytes

        padding = (8 - len(bitstream)%8)%8

        bitstream += "0"*padding

        byte_array = bytearray()

        for i in range(0, len(bitstream), 8):

            byte_array.append(
                int(bitstream[i:i+8], 2)
            )

        self.compressed_data = byte_array

        # Compression Statistics

        compressed_size = len(byte_array)

        original_size = (
            img.shape[0]
            * img.shape[1]
            * 3
        )

        ratio = original_size / compressed_size

        # Preview

        preview = cv2.resize(
            img,
            (img.shape[1]//2, img.shape[0]//2)
        )

        self.display_image(self.output_label, preview)

        # Statistics Text

        text = f"""
ORIGINAL SIZE      : {original_size} bytes

COMPRESSED SIZE    : {compressed_size} bytes

COMPRESSION RATIO  : {round(ratio,2)} : 1

TOTAL BITS         : {len(bitstream)}

TOTAL SYMBOLS      : {len(all_stream)}

IMAGE DIMENSIONS   : {img.shape[1]} x {img.shape[0]}
"""

        self.stats_text.setText(text)

    # =====================================================
    # SAVE FILE
    # =====================================================

    def save_file(self):

        if not hasattr(self, "compressed_data"):

            QMessageBox.warning(
                self,
                "Error",
                "Compress image first"
            )

            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "Binary Files (*.bin)"
        )

        if not path:
            return

        with open(path, "wb") as f:

            f.write(self.compressed_data)

        QMessageBox.information(
            self,
            "Success",
            "Compressed file saved successfully"
        )

# ---------------------------------------------------------

tab1 = Interpole()
tab2 = Image_info()
tab3 = MorphologyWindow()
tab4 = ImageFilter()
tab5 = FFTApp()
tab6 = FrequencyFilteringApp()
tab7 = ImageProcessingGUI()
tab8 = JPEGCompressor()

# Add each tab individually: .addTab(widget, "Title")
tabs.addTab(tab1, "Interpolate")
tabs.addTab(tab2, "Image Info")
tabs.addTab(tab3, "Morphology")
tabs.addTab(tab4, "Image Filtering")
tabs.addTab(tab5, "FFT Analysis")
tabs.addTab(tab6, "Frequency Filtering")
tabs.addTab(tab7, "Image Processing")
tabs.addTab(tab8, "JPEG Compression")


mainlayout = QVBoxLayout()

mainlayout.addWidget(tabs)

window.setLayout(mainlayout)

window.show()

sys.exit(ipapp.exec())