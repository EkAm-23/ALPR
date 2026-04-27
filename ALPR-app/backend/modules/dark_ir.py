import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARKIR_ROOT = os.path.join(PROJECT_ROOT, "DarkIR")
DARKIR_CONFIG_PATH = os.path.join(DARKIR_ROOT, "options", "inference", "LOLBlur.yml")
DARKIR_WEIGHTS_PATH = os.path.join(DARKIR_ROOT, "models", "DarkIR_1k_cr_mt.pt")


def _add_darkir_to_path():
    if DARKIR_ROOT not in sys.path:
        sys.path.insert(0, DARKIR_ROOT)


@lru_cache(maxsize=1)
def _load_darkir_model():
    """
    Lazily load the DarkIR network and checkpoint once for repeated inference.
    """
    _add_darkir_to_path()

    import torch
    from archs.DarkIR import DarkIR
    from options.options import parse

    opt = parse(DARKIR_CONFIG_PATH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoints = torch.load(DARKIR_WEIGHTS_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoints, dict) and "model_state_dict" in checkpoints:
        state_dict = checkpoints["model_state_dict"]
    elif isinstance(checkpoints, dict) and "params" in checkpoints:
        state_dict = checkpoints["params"]
    else:
        state_dict = checkpoints

    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    detected_width = state_dict["intro.weight"].shape[0]

    model = DarkIR(
        img_channel=opt["network"]["img_channels"],
        width=detected_width,
        middle_blk_num_enc=opt["network"]["middle_blk_num_enc"],
        middle_blk_num_dec=opt["network"]["middle_blk_num_dec"],
        enc_blk_nums=opt["network"]["enc_blk_nums"],
        dec_blk_nums=opt["network"]["dec_blk_nums"],
        dilations=opt["network"]["dilations"],
        extra_depth_wise=opt["network"]["extra_depth_wise"],
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, device


def process_with_darkir(image):
    """
    Restore a single image with the DarkIR model and return the result as a PIL image.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image).copy()

    img_np = img_np.astype(np.float32) / 255.0

    model, device = _load_darkir_model()

    import torch

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_tensor = model(tensor)

    restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
    restored_img = (restored_img * 255.0).round().astype(np.uint8)
    return Image.fromarray(restored_img)


def managing_contrast_and_brightness_mathematically(image, gamma=0.5, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Enhances a low-light image with a grayscale OCR-focused preprocessing chain.
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image.
        gamma (float): Gamma value for correction (< 1.0 brightens the image).
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple): Grid size for histogram equalization.
        
    Returns:
        PIL.Image: The enhanced image.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    # Ensure single-channel grayscale processing regardless of input format.
    if len(img_np.shape) == 2:
        gray = img_np
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
        gray = img_np[:, :, 0]
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Gamma Correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    contrast_enhanced = clahe.apply(gamma_corrected)

    # 3. Bilateral filter
    denoised = cv2.bilateralFilter(contrast_enhanced, 11, 75, 75)

    # 4. Black-hat morphology for dark text/glare separation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

    # 5. Adaptive threshold for OCR-ready character emphasis
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Final cleanup: combine black-hat and threshold responses
    final_output = cv2.add(blackhat, thresh)

    # Keep downstream compatibility: detector/UI expect 3-channel images.
    final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_output_rgb)
