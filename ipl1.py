import cv2
import numpy as np
from math import cos, sin, radians



def nearest_neighbour(input_img,
                      inp_img_height,
                      inp_img_width,
                      scale_factor):

    scl_img_height = int(inp_img_height * scale_factor)
    scl_img_width = int(inp_img_width * scale_factor)

    scl_img = np.zeros(
        (scl_img_height, scl_img_width)
    )

    for i in range(scl_img_height):

        for j in range(scl_img_width):

            val_x = int(i / scale_factor)
            val_y = int(j / scale_factor)

            # boundary protection
            val_x = min(val_x, inp_img_height - 1)
            val_y = min(val_y, inp_img_width - 1)

            scl_img[i][j] = input_img[val_x][val_y]

    return scl_img



def bilinear_interpolation(input_img,inp_img_height,inp_img_width,scale_factor):
  scl_img_height = int(inp_img_height*scale_factor)
  scl_img_width = int(inp_img_width*scale_factor)
  interpol_img=np.zeros((scl_img_height,scl_img_width))

  for i in range(scl_img_height):
    for j in range(scl_img_width):
      val_x=i/scale_factor
      val_y=j/scale_factor

      i_low = int(val_x)
      i_high = min(int(val_x+1),inp_img_height-1)
      j_low = int(val_y)
      j_high = min(int(val_y+1),inp_img_width-1)


      iv_11 = input_img[i_low][j_low]
      iv_12 = input_img[i_low][j_high]
      iv_21 = input_img[i_high][j_low]
      iv_22 = input_img[i_high][j_high]

      dx = val_x - i_low
      dy = val_y - j_low

      q_x = iv_11*(1-(dx)) + iv_21*(dx)
      r_x = iv_12*(1-(dx)) + iv_22*(dx)

      q_y = q_x*(1-(dy)) + r_x*(dy)
      interpol_img[i][j]=q_y
  print('The scale image after Bilinear Interpolation is')
  return interpol_img

def rotation(input_img,
             inp_img_height,
             inp_img_width,
             theta):

    theta = radians(theta)

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    new_dim_height = int(
        np.ceil(
            abs(inp_img_height * cos_theta)
            +
            abs(inp_img_width * sin_theta)
        )
    )

    new_dim_width = int(
        np.ceil(
            abs(inp_img_width * cos_theta)
            +
            abs(inp_img_height * sin_theta)
        )
    )

    rot_img = np.zeros(
        (new_dim_height, new_dim_width)
    )

    # floating centers
    new_ct_x = new_dim_height / 2
    new_ct_y = new_dim_width / 2

    old_ct_x = inp_img_height / 2
    old_ct_y = inp_img_width / 2

    for i in range(new_dim_height):

        for j in range(new_dim_width):

            ct_x = i - new_ct_x
            ct_y = j - new_ct_y

            rot_x = (
                cos_theta * ct_x
                +
                sin_theta * ct_y
            )

            rot_y = (
                -sin_theta * ct_x
                +
                cos_theta * ct_y
            )

            src_x = round(rot_x + old_ct_x)
            src_y = round(rot_y + old_ct_y)

            if (
                0 <= src_x < inp_img_height
                and
                0 <= src_y < inp_img_width
            ):

                rot_img[i, j] = input_img[src_x, src_y]

    return rot_img
