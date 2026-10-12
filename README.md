# 🖼️ py_compound — Image Processing from Scratch in Python

> A hands-on collection of image processing algorithms implemented in pure Python and NumPy — no hiding behind OpenCV's magic functions.

---

## What is this?

This repo is a learning-focused implementation of classic **Image Processing and Computer Vision** concepts. Instead of just calling `cv2.resize()` or `cv2.filter2D()` and moving on, every algorithm here is built from the ground up — pixel by pixel, formula by formula.

Think of it as your IPL (Image Processing Lab) coursework brought to life in Python.

---

## 📁 What's Inside

| File | What it does |
|------|-------------|
| `ip1.py` | Core image processing utilities (the big one — ~80KB!) |
| `ipl1.py` | Image scaling & rotation |
| `ipl2.py` | BMP file reader from scratch |
| `ipl3.py` | Morphological operations |
| `ipl4.py` | Spatial filtering & edge detection |
| `ipl5.py` | FFT / Frequency domain (version 1) |
| `ipl6.py` | FFT / Frequency domain (version 2, optimized) |
| `ipl7.py` | Histogram equalization & color space conversion |
| `ipl8.py` | JPEG-style compression (DCT + quantization) |
| `test.py` | PyQt5 GUI test/demo |

---

## 🔬 Module Breakdown

### `ipl1.py` — Scaling & Rotation
Two classic interpolation methods implemented without any library shortcuts:
- **Nearest Neighbour Interpolation** — Fast and simple: just pick the closest pixel when scaling.
- **Bilinear Interpolation** — Smoother results by blending the four surrounding pixels.
- **Image Rotation** — Rotate an image by an arbitrary angle using forward/inverse mapping.

```python
# Scale an image up by 2x using nearest neighbour
scaled = nearest_neighbour(img, height, width, scale_factor=2.0)

# Or get smoother results with bilinear interpolation
scaled = bilinear_interpolation(img, height, width, scale_factor=2.0)
```

---

### `ipl2.py` — BMP File Reader
Ever wondered what's actually inside a `.bmp` file? This module reads BMP images by parsing the **raw binary header** manually — extracting width, height, bit depth, pixel data — everything, byte by byte.

No `cv2.imread()` used here. Just `open()`, binary reads, and math.

---

### `ipl3.py` — Morphological Operations
Classic binary image morphology:
- **Erosion** — Shrinks white regions; great for noise removal.
- **Dilation** — Expands white regions; fills small gaps.

Both operations use custom structuring elements (kernels) and are built with nested loops — no shortcuts.

---

### `ipl4.py` — Spatial Filtering & Edge Detection
A full convolution engine plus several useful filters:
- **`convolve(img, kernel)`** — Generic 2D convolution with reflect-padding.
- **Mean Filter** — Smoothing by averaging neighborhood pixels.
- **Median Filter** — Better at removing salt-and-pepper noise than mean.
- **Prewitt Filter** — Horizontal and vertical edge detection.

```python
# Blur an image with a 5x5 mean filter
blurred = mean_filter(img, ksize=5)

# Detect horizontal edges with Prewitt
edges = prewitt_filter(img, direction='x')
```

---

### `ipl5.py` & `ipl6.py` — Fast Fourier Transform (FFT)
A from-scratch **Cooley-Tukey FFT** implementation — the same algorithm that powers `numpy.fft`. Both files tackle the same problem with slightly different approaches:

- **`fft1d(signal)`** — 1D FFT using the divide-and-conquer butterfly algorithm.
- **`ifft1d(signal)`** — Inverse FFT to reconstruct the original signal.
- **`fft2d(img)`** — 2D FFT by applying `fft1d` row-wise then column-wise.
- **Zero-padding** — Images are padded to the next power-of-2 size for FFT efficiency.

This is genuinely impressive stuff — the math is non-trivial and seeing it written in plain Python makes it very readable.

---

### `ipl7.py` — Histogram Equalization & Color Space Conversion
- **`histogram(img)`** — Compute pixel intensity histogram manually (no `np.histogram`).
- **`equalize_gray(img)`** — Histogram equalization to improve contrast in grayscale images using the CDF.
- **`rgb_to_hsv_manual(img)`** — Full RGB → HSV color space conversion using the mathematical formula.

```python
# Boost contrast in a washed-out image
enhanced = equalize_gray(gray_img)

# Convert to HSV for color-based processing
hsv = rgb_to_hsv_manual(color_img)
```

---

### `ipl8.py` — JPEG-Style Compression (DCT + Quantization)
This one's the most advanced module. It implements the core of how JPEG compression actually works:

1. **`rgb2ycc(img)`** — Convert RGB → YCbCr (separate luminance from color).
2. **`chroma_subsample(channel)`** — Downsample chroma channels (4:2:0 subsampling).
3. **`zero_center(channel)`** — Shift pixel values to center around 0.
4. **`dct_2d(block)`** — 2D Discrete Cosine Transform on 8×8 blocks — the heart of JPEG.
5. **Quantization matrices** — Standard JPEG luminance (`LUM`) and chrominance (`CHR`) tables are hardcoded.

```python
Y, Cb, Cr = rgb2ycc(img)          # Step 1: Color space
Cb_sub = chroma_subsample(Cb)      # Step 2: Subsample
Y_centered = zero_center(Y)        # Step 3: Center
# ... then apply DCT on 8x8 blocks
```

---

## 🛠️ Setup

**Prerequisites:** Python 3.8+

```bash
# Clone the repo
git clone https://github.com/mpranavr/py_compound.git
cd py_compound

# Install dependencies
pip install -r requirements.txt
```

The main dependencies are:
- `numpy` — Array math and matrix operations
- `opencv-python` (cv2) — Image I/O and display
- `matplotlib` — Plotting histograms and results
- `PyQt5` — GUI components (for `test.py`)

---

## 💡 Who is this for?

- Students taking an **Image Processing or Computer Vision** course
- Anyone curious about **what's happening under the hood** of libraries like OpenCV
- People who want to **understand algorithms by implementing them**, not just using them

---

## 🧠 Philosophy

> *"I hear and I forget. I see and I remember. I do and I understand."*

Every function in this repo was written to understand — not just to work. You'll find things like manually parsing BMP headers, hand-rolling FFT butterfly computations, and implementing the exact JPEG quantization tables from the spec. It's educational by design.

---

## 📄 License

MIT — use it, learn from it, build on it.
