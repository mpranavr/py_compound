import cv2
import numpy as np

def pad_image(img, se):
    m, n = se.shape
    pad_h, pad_w = m // 2, n // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def ErodeBinary(image, struct_element):
    padded = pad_image(image, struct_element)
    m, n = struct_element.shape
    eroded = np.zeros_like(image)
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            region = padded[i:i+m, j:j+n]
            if np.all(region[struct_element == 1] == 1):
                eroded[i, j] = 1
    return eroded

def DilateBinary(image, struct_element):
    padded = pad_image(image, struct_element)
    m, n = struct_element.shape
    dilated = np.zeros_like(image)
    for i in range(dilated.shape[0]):
        for j in range(dilated.shape[1]):
            region = padded[i:i+m, j:j+n]
            if np.any(region[struct_element == 1] == 1):
                dilated[i, j] = 1
    return dilated

def OpenBinary(image, struct_element):
    return DilateBinary(ErodeBinary(image, struct_element), struct_element)

def CloseBinary(image, struct_element):
    return ErodeBinary(DilateBinary(image, struct_element), struct_element)

