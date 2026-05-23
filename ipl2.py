import cv2
import numpy as np


def img_read(img):
    with open(img, "rb") as f:
        header = f.read(54)
        file_type=header[0:2]
        if file_type != b'BM':
            print("file should be in .bmp format")
            return
        file_size = int.from_bytes(header[2:6], byteorder='little')
        offset = int.from_bytes(header[10:14], byteorder='little')
        header_size = int.from_bytes(header[14:18], byteorder='little')
        img_width = int.from_bytes(header[18:22], byteorder='little')
        img_height = int.from_bytes(header[22:26], byteorder='little')
        planes = int.from_bytes(header[26:28], byteorder='little')
        bit_count = int.from_bytes(header[28:30], byteorder='little')
        compression = int.from_bytes(header[30:34], byteorder='little')
        img_size = int.from_bytes(header[34:38], byteorder='little')
        xpix_m = int.from_bytes(header[38:42], byteorder='little')
        ypix_m = int.from_bytes(header[42:46], byteorder='little')
        colors = int.from_bytes(header[46:50], byteorder='little')
        imp_colors = int.from_bytes(header[50:54], byteorder='little')
        pixels = f.read()
        l = 0

        if bit_count == 24:
            img_pixel = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            for i in range(img_height - 1, -1, -1):
                for j in range(img_width):
                    for k in range(3):
                        img_pixel[i][j][k] = int(pixels[l])
                        l += 1

            img_info = {
                "file_size": file_size, "offset": offset,
                "header_type": file_type, "header_size": header_size,
                "width": img_width, "height": img_height,
                "planes": planes, "bit_count": bit_count,
                "compression": compression, "img_size": img_size,
                "xpix_m": xpix_m, "ypix_m": ypix_m,
                "colors": colors, "imp_colors": imp_colors,
                "pixel_data": img_pixel
            }

            items_to_print = list(img_info.items())[:-1]
            for key, value in items_to_print:
              print(f"{key}: {value}")

            return img_info, img_pixel

        if bit_count <= 8:
            palette = []

            for i in range(256):
                blue = int(pixels[l]); l += 1
                green = int(pixels[l]); l += 1
                red = int(pixels[l]); l += 1
                l += 1
                palette.append([blue, green, red])

            img_pixel = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            indices = np.zeros((img_height, img_width), dtype=np.uint8)

            for i in range(img_height - 1, -1, -1):
                for j in range(img_width):
                    pixel_value = int(pixels[l])
                    img_pixel[i, j] = palette[pixel_value]
                    indices[i, j] = pixel_value
                    l += 1

            img_info = {
                "file_size": file_size, "offset": offset,
                "header_type": file_type, "header_size": header_size,
                "width": img_width, "height": img_height,
                "planes": planes, "bit_count": bit_count,
                "compression": compression, "img_size": img_size,
                "xpix_m": xpix_m, "ypix_m": ypix_m,
                "colors": colors, "imp_colors": imp_colors,
                "paletters": palette, "indices": indices
            }
            print("image is read")

            items_to_print = list(img_info.items())[:-1]
            for key, value in items_to_print:
              print(f"{key}: {value}")


            return img_info, img_pixel


def img_write(img, img_info, img_pixel):

    header = bytearray(54)
    header[0:2] = b'BM'
    header[2:6] = img_info["file_size"].to_bytes(4, byteorder='little')
    header[6:10] = (0).to_bytes(4, byteorder='little')
    header[10:14] = img_info["offset"].to_bytes(4, byteorder='little')
    header[14:18] = img_info["header_size"].to_bytes(4, byteorder='little')
    header[18:22] = img_info["width"].to_bytes(4, byteorder='little')
    header[22:26] = img_info["height"].to_bytes(4, byteorder='little')
    header[26:28] = img_info["planes"].to_bytes(2, byteorder='little')
    header[28:30] = img_info["bit_count"].to_bytes(2, byteorder='little')
    header[30:34] = img_info["compression"].to_bytes(4, byteorder='little')
    header[34:38] = img_info["img_size"].to_bytes(4, byteorder='little')
    header[38:42] = img_info["xpix_m"].to_bytes(4, byteorder='little')
    header[42:46] = img_info["ypix_m"].to_bytes(4, byteorder='little')
    header[46:50] = img_info["colors"].to_bytes(4, byteorder='little')
    header[50:54] = img_info["imp_colors"].to_bytes(4, byteorder='little')

    pixels = bytearray()

    if img_info["bit_count"] == 24:
        for i in range(img_info["height"] - 1, -1, -1):
            for j in range(img_info["width"]):
                for k in range(3):
                    pixels.append(int(img_pixel[i][j][k]))

    elif img_info["bit_count"] == 8:

        palette_bytes = bytearray()
        for (b, g, r) in img_info["paletters"]:
            palette_bytes.extend([b, g, r, 0])


        for i in range(img_info["height"] - 1, -1, -1):
            for j in range(img_info["width"]):
                pixels.append(int(img_info["indices"][i][j]))

    with open(img, "wb") as f:
        f.write(header)
        if img_info["bit_count"] <= 8:
            f.write(palette_bytes)
        f.write(pixels)

    print(f"Image written successfully to {img} \n")


def red_color_chl_manipulation(img):
    (img_info, img_pixel) = img_read(img)
    bit_count = img_info["bit_count"]
    if bit_count == 24:
        for i in range(img_info["height"] - 1, -1, -1):
            for j in range(img_info["width"]):
                img_pixel[i][j][2] = 0
    elif bit_count <= 8:
        for i in range(256):
            img_info["paletters"][i][2] = 0
    img_write("red_channel_manipulated_image.bmp", img_info, img_pixel)


def blue_color_chl_manipulation(img):
    (img_info, img_pixel) = img_read(img)
    bit_count = img_info["bit_count"]
    if bit_count == 24:
        for i in range(img_info["height"] - 1, -1, -1):
            for j in range(img_info["width"]):
                img_pixel[i][j][0] = 0
    elif bit_count <= 8:
        for i in range(256):
            img_info["paletters"][i][0] = 0
    img_write("blue_channel_manipulated_image.bmp", img_info, img_pixel)


def green_color_chl_manipulation(img):
    (img_info, img_pixel) = img_read(img)
    bit_count = img_info["bit_count"]
    if bit_count == 24:
        for i in range(img_info["height"] - 1, -1, -1):
            for j in range(img_info["width"]):
                img_pixel[i][j][1] = 0
    elif bit_count <= 8:
        for i in range(256):
            img_info["paletters"][i][1] = 0
    img_write("green_channel_manipulated_image.bmp", img_info, img_pixel)
