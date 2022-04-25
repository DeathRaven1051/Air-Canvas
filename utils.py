


import ctypes

import cv2


user32 = ctypes.windll.user32
SCREEN_WIDTH, SCREEN_HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


"""
Function to resize the screen 
to full size.
Accepts an image
Returns an image of full screen size
"""
def resize_to_full_screen(img):
    frame = cv2.flip(img, 1)
    frame_height, frame_width, _ = frame.shape

    scale_width = float(SCREEN_WIDTH) / float(frame_width)
    scale_height = float(SCREEN_HEIGHT) / float(frame_height)

    new_x, new_y = frame.shape[1] * max(scale_height, scale_width), frame.shape[0] * max(scale_width, scale_height)

    frame = cv2.resize(frame, (int(new_x), int(new_y)))

    return frame
