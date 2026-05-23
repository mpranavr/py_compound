import cv2
import numpy as np

import cv2
import numpy as np


# =========================================================
# ZERO PADDING
# =========================================================

def zeropadding(img):

    rows, cols = img.shape

    new_rows = 1 << (rows - 1).bit_length()

    new_cols = 1 << (cols - 1).bit_length()

    padded = np.zeros(
        (new_rows, new_cols),
        dtype=np.float32
    )

    padded[:rows, :cols] = img

    return padded, rows, cols


# =========================================================
# FFT 1D
# =========================================================

def fft1d(signal):

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    n = len(signal)

    if n == 1:
        return signal

    even = fft1d(signal[0::2])

    odd = fft1d(signal[1::2])

    factor = np.exp(
        -2j * np.pi *
        np.arange(n // 2) / n
    ) * odd

    return np.concatenate([
        even + factor,
        even - factor
    ])


# =========================================================
# IFFT 1D
# =========================================================

def ifft1d(signal):

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    n = len(signal)

    if n == 1:
        return signal

    even = ifft1d(signal[0::2])

    odd = ifft1d(signal[1::2])

    factor = np.exp(
        2j * np.pi *
        np.arange(n // 2) / n
    ) * odd

    return np.concatenate([
        even + factor,
        even - factor
    ])


# =========================================================
# FFT 2D
# =========================================================

def fft2d(img):

    padded, orig_rows, orig_cols = zeropadding(img)

    rows, cols = padded.shape

    row_fft = np.array([
        fft1d(row)
        for row in padded
    ])

    col_fft = np.array([
        fft1d(row_fft[:, col])
        for col in range(cols)
    ]).T

    return col_fft, orig_rows, orig_cols


# =========================================================
# IFFT 2D
# =========================================================

def ifft2d(freq,
           orig_rows=None,
           orig_cols=None):

    rows, cols = freq.shape

    row_ifft = np.array([
        ifft1d(row)
        for row in freq
    ])

    col_ifft = np.array([
        ifft1d(row_ifft[:, col])
        for col in range(cols)
    ]).T

    result = np.real(
        col_ifft
    ) / (rows * cols)

    if orig_rows is not None:

        result = result[
            :orig_rows,
            :orig_cols
        ]

    return result


# =========================================================
# FFT SHIFT
# =========================================================

def fftshift(img):

    rows, cols = img.shape

    r_h = rows // 2

    c_h = cols // 2

    shifted = np.zeros_like(img)

    shifted[:r_h, :c_h] = img[r_h:, c_h:]

    shifted[:r_h, c_h:] = img[r_h:, :c_h]

    shifted[r_h:, :c_h] = img[:r_h, c_h:]

    shifted[r_h:, c_h:] = img[:r_h, :c_h]

    return shifted


# =========================================================
# SPECTRUM
# =========================================================

def create_spectrum(fshift):

    return cv2.normalize(
        np.log(1 + np.abs(fshift)),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


# =========================================================
# IDEAL LPF
# =========================================================

def ideal_lpf(image_filename, cutoff):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            if d <= cutoff:

                mask[i, j] = 1

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# IDEAL HPF
# =========================================================

def ideal_hpf(image_filename, cutoff):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.ones(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            if d <= cutoff:

                mask[i, j] = 0

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# GAUSSIAN LPF
# =========================================================

def gaussian_lpf(image_filename, cutoff):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            mask[i, j] = np.exp(
                -(d**2) /
                (2 * cutoff**2)
            )

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# GAUSSIAN HPF
# =========================================================

def gaussian_hpf(image_filename, cutoff):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            mask[i, j] = 1 - np.exp(
                -(d**2) /
                (2 * cutoff**2)
            )

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# BUTTERWORTH LPF
# =========================================================

def butterworth_lpf(image_filename,
                     cutoff,
                     order=2):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            mask[i, j] = 1 / (
                1 +
                (d / (cutoff + 1e-6))**(2 * order)
            )

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# BUTTERWORTH HPF
# =========================================================

def butterworth_hpf(image_filename,
                     cutoff,
                     order=2):

    img = cv2.imread(
        image_filename,
        cv2.IMREAD_GRAYSCALE
    ).astype(np.float32)

    freq, orig_rows, orig_cols = fft2d(img)

    rows, cols = freq.shape

    crow, ccol = rows // 2, cols // 2

    fshift = fftshift(freq)

    mask = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):

        for j in range(cols):

            d = np.sqrt(
                (i - crow)**2 +
                (j - ccol)**2
            )

            mask[i, j] = 1 / (
                1 +
                ((cutoff + 1e-6) /
                 (d + 1e-6))**(2 * order)
            )

    filtered = fshift * mask

    result = ifft2d(
        fftshift(filtered),
        orig_rows,
        orig_cols
    )

    result = cv2.normalize(
        np.abs(result),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return (
        create_spectrum(fshift),
        (mask * 255).astype(np.uint8),
        result.astype(np.uint8)
    )


# =========================================================
# HYBRID IMAGE
# =========================================================

def hybrid_image(image1_path,
                 image2_path,
                 cutoff=30,
                 mode="type1"):

    # =========================================
    # LOAD IMAGES
    # =========================================

    image1 = cv2.imread(
        image1_path,
        cv2.IMREAD_GRAYSCALE
    )

    image2 = cv2.imread(
        image2_path,
        cv2.IMREAD_GRAYSCALE
    )

    h = min(
        image1.shape[0],
        image2.shape[0]
    )

    w = min(
        image1.shape[1],
        image2.shape[1]
    )

    image1 = cv2.resize(
        image1,
        (w, h)
    )

    image2 = cv2.resize(
        image2,
        (w, h)
    )

    # =========================================
    # TEMP SAVE RESIZED
    # =========================================

    cv2.imwrite("temp1.png", image1)

    cv2.imwrite("temp2.png", image2)

    # =========================================
    # TYPE 1
    # IMAGE1 HPF + IMAGE2 LPF
    # =========================================

    if mode == "type1":

        _, _, high_img = gaussian_hpf(
            "temp1.png",
            cutoff
        )

        _, _, low_img = gaussian_lpf(
            "temp2.png",
            cutoff
        )

    else:

        _, _, low_img = gaussian_lpf(
            "temp1.png",
            cutoff
        )

        _, _, high_img = gaussian_hpf(
            "temp2.png",
            cutoff
        )

    # =========================================
    # TYPE 2
    # IMAGE1 LPF + IMAGE2 HPF
    # =========================================


    # =========================================
    # HYBRID
    # =========================================

    hybrid = cv2.add(
        low_img,
        high_img
    )

    hybrid = cv2.normalize(
        hybrid,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return hybrid.astype(np.uint8)

def denoising_cameraman(image_filename, centers, radius=5):

    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    freq = fft2d(img)
    fshift = fftshift(freq)
    mask = np.ones((rows, cols), np.float32)
    for dx, dy in centers:
        x, y = crow + dx, ccol + dy
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - x)**2 + (j - y)**2) <= radius:
                    d = np.sqrt((i - x)**2 + (j - y)**2)
                    mask[i, j] = 1 / (1 + (radius / (d+1e-6))**(2*2))
        x_c, y_c = crow - dx, ccol - dy
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - x_c)**2 + (j - y_c)**2) <= radius:
                    d = np.sqrt((i - x_c)**2 + (j - y_c)**2)
                    mask[i, j] = 1 / (1 + (radius / (d+1e-6))**(2*2))
    fshift_filtered = fshift * mask
    f_ishift = fftshift(fshift_filtered)
    filtered_img = ifft2d(f_ishift)
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)

