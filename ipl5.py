import cv2
import numpy as np


def fft1d(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = fft1d(signal[0::2])
    odd = fft1d(signal[1::2])
    factor = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + factor[k] for k in range(n // 2)] + \
           [even[k] - factor[k] for k in range(n // 2)]


def ifft1d(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = ifft1d(signal[0::2])
    odd = ifft1d(signal[1::2])
    factor = [np.exp(2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + factor[k] for k in range(n // 2)] + \
           [even[k] - factor[k] for k in range(n // 2)]

def next_power_of_2(n):

    return 1 if n == 0 else 2**(n - 1).bit_length()

def fft2d(img):

    rows, cols = img.shape

    # ==========================================
    # PAD TO POWER OF 2
    # ==========================================

    new_rows = next_power_of_2(rows)

    new_cols = next_power_of_2(cols)

    padded = np.zeros(
        (new_rows, new_cols),
        dtype=np.float32
    )

    padded[:rows, :cols] = img

    # ==========================================
    # ROW FFT
    # ==========================================

    row_fft = np.array([
        fft1d(row)
        for row in padded
    ])

    # ==========================================
    # COLUMN FFT
    # ==========================================

    col_fft = np.array([
        fft1d(row_fft[:, col])
        for col in range(new_cols)
    ]).T

    return col_fft


def ifft2d(freq):

    rows, cols = freq.shape

    row_ifft = np.array([
        ifft1d(row)
        for row in freq
    ])

    col_ifft = np.array([
        ifft1d(row_ifft[:, col])
        for col in range(cols)
    ]).T

    result = np.real(col_ifft) / (rows * cols)

    # =====================================
    # CONVERT HERE
    # =====================================

    result = np.clip(result, 0, 255)

    result = result.astype(np.uint8)

    return result
  
def fftshift(img):
    rows, cols = img.shape
    r_half, c_half = rows // 2, cols // 2
    
    shifted = np.zeros_like(img)
    shifted[:r_half, :c_half] = img[r_half:, c_half:]    
    shifted[:r_half, c_half:] = img[r_half:, :c_half]     
    shifted[r_half:, :c_half] = img[:r_half, c_half:]     
    shifted[r_half:, c_half:] = img[:r_half, :c_half]     
    return shifted


def magnitude_phase_spectrum(fft_result):
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return magnitude, phase


def run_fft_example(img_path):
 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: image not found.")
        return

    img = np.float32(img)

    fft_result = fft2d(img)

    
    magnitude, phase = magnitude_phase_spectrum(fft_result)

    
    mag_spectrum = np.log(1 + magnitude)
    mag_spectrum = fftshift(mag_spectrum)
    mag_spectrum = cv2.normalize(mag_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    phase_spectrum = fftshift(phase)
    phase_spectrum = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    print("Original Image:")

    print("Magnitude Spectrum:")

    print("Phase Spectrum:")

    reconstructed = ifft2d(fft_result)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    print("Reconstructed Image:")

    return img, fft_result, reconstructed


def experiment_q2(img_path):
   
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: image not found.")
        return
    
    img = np.float32(img)
    rows, cols = img.shape

    mask = np.fromfunction(lambda x, y: (-1)**(x+y), (rows, cols))
    shifted_img = img * mask

    fft_result = fft2d(shifted_img)

    conj_fft = np.conjugate(fft_result)

    ifft_result = ifft2d(conj_fft)

    final_result = np.real(ifft_result) * mask

    final_result = np.clip(final_result, 0, 255).astype(np.uint8)

    print("Original Image:")

    print("Resultant Image After Steps:")
    

    return final_result

