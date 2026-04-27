import cv2

import cv2
import numpy as np

def get_dark_channel(img, kernel_size=15):
    """
    Computes the dark channel of an image.
    In haze-free images, the dark channel is mostly zero.
    In hazy images, the dark channel has high values due to atmospheric light.
    """
    min_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def classify_restoration_module(cv_img):
    try:
        # 1. Darkness Check (Lowest priority for detail, highest for trigger)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 65:
            return 'darkir'

        # 2. Haze Detection via DCP and Saturation
        # Haze is bright in dark channel, has low saturation, and is blurry.
        dark_channel = get_dark_channel(cv_img)
        dark_score = float(dark_channel.mean())
        
        # Saturation (Hazy images are washed out)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:,:,1].mean())
        
        # Sharpness (Haze blurs edges)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Classification Logic for Haze
        # If the dark channel is "heavy" and contrast/saturation are low
        contrast = float(np.std(gray))
        
        # Relaxed thresholds based on debug data
        if dark_score > 40 and saturation < 60 and contrast < 50:
            return 'dehaze'
            
        # Fallback for heavy blur/low contrast regions
        if laplacian_var < 700 and contrast < 40 and brightness > 65:
            return 'dehaze'

        # 3. Otherwise normal
        return 'normal'

    except Exception:
        return 'normal'