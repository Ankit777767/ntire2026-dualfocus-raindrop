import torch

def rgb_to_y(img):
    """
    img: tensor in [0,1], shape (3,H,W)
    """
    r, g, b = img[0], img[1], img[2]
    y = 0.299*r + 0.587*g + 0.114*b
    return y.unsqueeze(0)
