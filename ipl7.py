import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def cdf_from_hist(hist):
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized

def equalize_gray(img):
    hist = histogram(img)
    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    eq_img = cdf_normalized[img]
    return eq_img

def rgb_to_hsv_manual(img):
    img = img.astype('float32') / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Cmax = np.maximum(R, np.maximum(G, B))
    Cmin = np.minimum(R, np.minimum(G, B))
    diff = Cmax - Cmin
    H = np.zeros_like(Cmax)
    mask = diff != 0
    r_mask = (Cmax == R) & mask
    g_mask = (Cmax == G) & mask
    b_mask = (Cmax == B) & mask
    H[r_mask] = (60 * ((G[r_mask] - B[r_mask]) / diff[r_mask]) + 360) % 360
    H[g_mask] = (60 * ((B[g_mask] - R[g_mask]) / diff[g_mask]) + 120) % 360
    H[b_mask] = (60 * ((R[b_mask] - G[b_mask]) / diff[b_mask]) + 240) % 360
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = (diff[Cmax != 0] / Cmax[Cmax != 0])
    V = Cmax
    hsv = np.stack((H, S, V), axis=2)
    return hsv

def hsv_to_rgb_manual(hsv):
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    C = V * S
    X = C * (1 - np.abs((H / 60) % 2 - 1))
    m = V - C
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    mask0 = (H >= 0) & (H < 60)
    mask1 = (H >= 60) & (H < 120)
    mask2 = (H >= 120) & (H < 180)
    mask3 = (H >= 180) & (H < 240)
    mask4 = (H >= 240) & (H < 300)
    mask5 = (H >= 300) & (H < 360)
    R[mask0], G[mask0], B[mask0] = C[mask0], X[mask0], 0
    R[mask1], G[mask1], B[mask1] = X[mask1], C[mask1], 0
    R[mask2], G[mask2], B[mask2] = 0, C[mask2], X[mask2]
    R[mask3], G[mask3], B[mask3] = 0, X[mask3], C[mask3]
    R[mask4], G[mask4], B[mask4] = X[mask4], 0, C[mask4]
    R[mask5], G[mask5], B[mask5] = C[mask5], 0, X[mask5]
    R = (R + m) * 255
    G = (G + m) * 255
    B = (B + m) * 255
    rgb = np.stack((B, G, R), axis=2).astype(np.uint8)
    return rgb

def equalize_color(img):
    hsv = rgb_to_hsv_manual(img)
    v = (hsv[:, :, 2] * 255).astype(np.uint8)
    v_eq = equalize_gray(v)
    hsv[:, :, 2] = v_eq / 255.0
    eq_img = hsv_to_rgb_manual(hsv)
    return eq_img

def match_histogram(source, target):
    hist_src = histogram(source)
    hist_tgt = histogram(target)
    cdf_src = cdf_from_hist(hist_src)
    cdf_tgt = cdf_from_hist(hist_tgt)
    mapping = np.zeros(256)
    for i in range(256):
        diff = np.abs(cdf_src[i] - cdf_tgt)
        mapping[i] = np.argmin(diff)
    matched = mapping[source]
    return matched.astype(np.uint8)
  
def match_histogram_color(src, tgt):
    hsv_src = rgb_to_hsv_manual(src)
    hsv_tgt = rgb_to_hsv_manual(tgt)
    v_src = (hsv_src[:, :, 2] * 255).astype(np.uint8)
    v_tgt = (hsv_tgt[:, :, 2] * 255).astype(np.uint8)
    v_matched = match_histogram(v_src, v_tgt)
    hsv_src[:, :, 2] = v_matched / 255.0
    matched = hsv_to_rgb_manual(hsv_src)
    return matched

def plot_eq_hist(img1, img2, title1, title2):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.hist(img1.ravel(), 256, [0,256])
    plt.subplot(1,2,2)
    plt.title(title2)
    plt.hist(img2.ravel(), 256, [0,256])
    plt.show()

def plot_mat_hist(img1, img2, img3, title1, title2, title3):
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.title(title1)
    plt.hist(img1.ravel(), 256, [0,256])
    plt.subplot(1,3,2)
    plt.title(title2)
    plt.hist(img2.ravel(), 256, [0,256])
    plt.subplot(1,3,3)
    plt.title(title3)
    plt.hist(img3.ravel(), 256, [0,256])
    plt.show()
