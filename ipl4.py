import cv2
import numpy as np
import os


def convolve(img, kernel):
    img_f = img.astype(np.float32)
    kernel_f = kernel.astype(np.float32)
    k = kernel_f.shape[0] // 2
    padded = np.pad(img_f, k, mode='reflect')
    output = np.zeros_like(img_f, dtype=np.float32)
    kh, kw = kernel_f.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel_f)
    return output

def mean_filter(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=float) / (ksize * ksize)
    float_result = convolve(img, kernel)
    return np.clip(float_result, 0, 255).astype(np.uint8)

def median_filter(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    k = ksize // 2
    padded = np.pad(img, k, mode='reflect')
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+ksize, j:j+ksize].flatten()
            output[i, j] = np.median(region)
    return output

def prewitt_filter(img, direction='x'):
    kernel_x = np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=float)
    kernel_y = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=float)
    if direction == 'x':
        gx = convolve(img, kernel_x)
        return np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    elif direction == 'y':
        gy = convolve(img, kernel_y)
        return np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    else:
        gx = convolve(img, kernel_x)
        gy = convolve(img, kernel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

def laplacian_filter(img):
    kernel = np.array([[0, 1, 0], [1,-4, 1], [0, 1, 0]], dtype=float)
    float_result = convolve(img, kernel)
    return np.clip(np.abs(float_result), 0, 255).astype(np.uint8)

def sobel_filter(img, direction='x'):
    kernel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=float)
    kernel_y = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=float)
    if direction == 'x':
        gx = convolve(img, kernel_x)
        return np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    elif direction == 'y':
        gy = convolve(img, kernel_y)
        return np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    else:
        gx = convolve(img, kernel_x)
        gy = convolve(img, kernel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

def gaussian_kernel(ksize=5, sigma=1.0):
    if ksize % 2 == 0:
        ksize += 1
    ax = np.linspace(-(ksize-1)//2, (ksize-1)//2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma**2)))
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel / kernel_sum
    return kernel

def gaussian_blur(img, ksize=5, sigma=1.0):
    kernel = gaussian_kernel(ksize, sigma)
    float_result = convolve(img, kernel)
    return np.clip(float_result, 0, 255).astype(np.uint8)

def laplacian_of_gaussian(img, ksize=5, sigma=1.0):
    blurred = gaussian_blur(img, ksize, sigma)
    return laplacian_filter(blurred)

def run_spatial_filter(filter_name, image_path, ksize, sigma, direction):
    if not image_path:
        print("No image selected or found.")
        return
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    if ksize % 2 == 0:
        ksize += 1
    print(f"Applying {filter_name} to {os.path.basename(image_path)}")
    if filter_name == 'Mean Filter':
        filtered_img = mean_filter(img, ksize=ksize)
    elif filter_name == 'Median Filter':
        filtered_img = median_filter(img, ksize=ksize)
    elif filter_name == 'Gaussian Blur':
        filtered_img = gaussian_blur(img, ksize=ksize, sigma=sigma)
    elif filter_name == 'Laplacian':
        filtered_img = laplacian_filter(img)
    elif filter_name == 'Laplacian of Gaussian':
        filtered_img = laplacian_of_gaussian(img, ksize=ksize, sigma=sigma)
    elif filter_name == 'Prewitt':
        filtered_img = prewitt_filter(img, direction=direction)
    elif filter_name == 'Sobel':
        filtered_img = sobel_filter(img, direction=direction)
    else:
        filtered_img = img
    combined_display = np.hstack([img, filtered_img])
    print("Original Image (Left) vs. Filtered Image (Right)")
    
def gaussian_unblur(I_blur, ksize=5, sigma=1.0, max_iter=10, tol=1e-4, verbose=False):
    kernel = gaussian_kernel(ksize, sigma).astype(np.float32)
    I_k = I_blur.astype(np.float32).copy()
    prev = I_k.copy()
    if verbose:
        print("Running deblurring...")
    for it in range(max_iter):
        A_k = convolve(I_k, kernel).astype(np.float32) + 1e-6
        B_k = I_blur.astype(np.float32) / A_k
        C_k = convolve(B_k, kernel).astype(np.float32)
        I_k = I_k * C_k
        numer = np.linalg.norm(I_k - prev)
        denom = np.linalg.norm(prev) + 1e-6
        diff = numer / denom
        if verbose:
            print(f"   Iter {it+1}: change = {diff:.6e}")
        if diff < tol:
            if verbose:
                print(f"Converged at iteration {it+1}.")
            break
        prev = I_k.copy()
    I_k = np.clip(I_k, 0, 255)
    if verbose:
        print("Deblurring finished.")
    return I_k.astype(np.uint8)
