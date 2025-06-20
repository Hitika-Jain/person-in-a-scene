# [same imports as before]
import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import io
import cv2
from rembg import remove
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import requests

MODEL_PATH = "models/yolov8n-seg.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

download_model()

from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import SegmentationModel
add_safe_globals([SegmentationModel])

from ultralytics import YOLO
model = YOLO(MODEL_PATH)

blend_factor = 0.3

st.set_page_config(page_title="Place a Person in a scene smoothly", layout="centered")
st.title("🧍‍♂️ Person Placement in a scene")

bg_file = st.file_uploader("📤 Upload Background Image", type=["png", "jpg", "jpeg"])
person_file = st.file_uploader("🧍 Upload Person Image", type=["png", "jpg", "jpeg"])

def auto_crop_person(image):
    img_np = np.array(image.convert("RGB"))
    results = model.predict(img_np, verbose=False)[0]
    for i, cls in enumerate(results.boxes.cls):
        if model.names[int(cls)] == "person":
            mask = results.masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x1, y1, x2, y2 = min(x_indices), min(y_indices), max(x_indices), max(y_indices)
            padding = 20  # pixels
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)

            person_crop = image.crop((x1, y1, x2, y2))
            return person_crop
    return image

def remove_background(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    result = remove(buffer.getvalue())
    return Image.open(io.BytesIO(result)).convert("RGBA")

def trim_transparent_border(img):
    """Crop out fully transparent edges after background removal."""
    np_img = np.array(img)
    if np_img.shape[2] == 4:  # Has alpha channel
        alpha = np_img[:, :, 3]
        y_nonzero, x_nonzero = np.where(alpha > 10)
        if len(y_nonzero) == 0 or len(x_nonzero) == 0:
            return img  # completely transparent
        x1, y1, x2, y2 = min(x_nonzero), min(y_nonzero), max(x_nonzero), max(y_nonzero)
        return img.crop((x1, y1, x2, y2))
    else:
        return img

def color_transfer_rgba(source_img, target_img):
    source_rgba = source_img.convert("RGBA")
    target_rgb = target_img.convert("RGB")
    source_np = np.array(source_rgba)
    alpha = source_np[..., 3]
    source_rgb = source_np[..., :3]
    target_np = np.array(target_rgb.resize(source_img.size))
    source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
    src_mean, src_std = cv2.meanStdDev(source_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
    transfer = (source_lab - src_mean.T) * (tgt_std.T / (src_std.T + 1e-6)) + tgt_mean.T
    transfer = np.clip(transfer, 0, 255).astype(np.uint8)
    transfer_rgb = cv2.cvtColor(transfer, cv2.COLOR_LAB2RGB)
    blended_rgb = ((1 - blend_factor) * source_rgb + blend_factor * transfer_rgb).astype(np.uint8)
    blended_rgba = np.dstack((blended_rgb, alpha))
    return Image.fromarray(blended_rgba, "RGBA")

def generate_shadow_mask(shadow_layer):
    shadow_np = np.array(shadow_layer)
    alpha_channel = shadow_np[..., 3]
    mask = (alpha_channel > 10).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

def add_foot_shadow_only(image_size, position, person_size, shadow_opacity=100, shadow_color=(0, 0, 0), blur_radius=8):
    shadow_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)

    foot_center_x = position[0] + person_size[0] // 2
    foot_base_y = position[1] + person_size[1] - 5
    shadow_width = int(person_size[0] * 0.6)
    shadow_height = int(person_size[1] * 0.12)

    bbox = [foot_center_x - shadow_width // 2, foot_base_y,
            foot_center_x + shadow_width // 2, foot_base_y + shadow_height]

    draw.ellipse(bbox, fill=shadow_color + (shadow_opacity,))
    blurred_shadow = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    return blurred_shadow

def classify_shadow_type(shadow_mask_img, blur_threshold=0.05):
    """
    Classifies shadow as hard or soft based on edge sharpness.
    Args:
        shadow_mask_img (PIL.Image): Binary shadow mask in 'L' mode (0 or 255)
        blur_threshold (float): Threshold to distinguish hard/soft based on edge density
    
    Returns:
        str: 'Hard Shadow' or 'Soft Shadow'
    """
    mask_np = np.array(shadow_mask)

    # Use Canny edge detection to find sharp transitions
    edges = cv2.Canny(mask_np, threshold1=30, threshold2=100)

    # Measure edge density (amount of sharp edges)
    edge_density = np.sum(edges > 0) / (mask_np.shape[0] * mask_np.shape[1])

    # Calculate edge sharpness using Laplacian variance
    laplacian = cv2.Laplacian(mask_np, cv2.CV_64F)
    sharpness = laplacian.var()

    # Threshold-based classification
    if sharpness > 200 and edge_density > 0.01:
        return "Hard Shadow"
    else:
        return "Soft Shadow"

def paste_person_on_bg(bg, person_rgba, pos):
    bg = bg.convert("RGBA")
    temp = Image.new("RGBA", bg.size)
    temp.paste(person_rgba, pos, mask=person_rgba)
    return Image.alpha_composite(bg, temp)

def get_dynamic_scale(y, img_height, min_scale=0.25, max_scale=0.65):
    relative_pos = y / img_height
    scale = min_scale + (1 - relative_pos) * (max_scale - min_scale)
    return scale

def estimate_average_person_height_in_background(bg_img):
    img_np = np.array(bg_img.convert("RGB"))
    results = model.predict(img_np, verbose=False)[0]
    heights = []
    for i, cls in enumerate(results.boxes.cls):
        if model.names[int(cls)] == "person":
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
            heights.append(y2 - y1)
    return np.mean(heights) if heights else None

if bg_file and person_file:
    bg_img_orig = Image.open(bg_file).convert("RGB")
    person_img = Image.open(person_file).convert("RGB")

    with st.spinner(" Cropping Person Using YOLO..."):
        cropped = auto_crop_person(person_img)

    with st.spinner(" Removing Background..."):
        transparent_person = remove_background(cropped)
        transparent_person = trim_transparent_border(transparent_person)


    st.markdown("### Click where you want to place the foot of the person")
    display_width = 800
    display_bg = bg_img_orig.copy()
    resize_factor = 800 / bg_img_orig.width if bg_img_orig.width > 800 else 1.0
    display_bg = display_bg.resize((int(bg_img_orig.width * resize_factor), int(bg_img_orig.height * resize_factor)))

    coords = streamlit_image_coordinates(display_bg, key="place-person")

    if coords:
        x_disp, y_disp = int(coords["x"]), int(coords["y"])
        scale_factor = bg_img_orig.width / display_bg.width
        x = int(x_disp * scale_factor)
        y = int(y_disp * scale_factor)

        avg_height = estimate_average_person_height_in_background(bg_img_orig)
        if avg_height:
            scale = avg_height / transparent_person.height
        else:
            scale = get_dynamic_scale(y, bg_img_orig.height)

        person_resized = transparent_person.resize((
            int(transparent_person.width * scale),
            int(transparent_person.height * scale)
        ))
        st.image(person_resized, caption=" resized Person")
        st.success(f" Placing person at: ({x}, {y}) in original resolution")

        patch_size = 30
        patch = bg_img_orig.crop((
            max(x - patch_size, 0),
            max(y - patch_size, 0),
            min(x + patch_size, bg_img_orig.width),
            min(y + patch_size, bg_img_orig.height)
        ))

        harmonized = color_transfer_rgba(person_resized, patch)

        adjusted_x = x - person_resized.width // 2
        adjusted_y = y - person_resized.height

        shadow_layer = Image.new("RGBA", bg_img_orig.size, (0, 0, 0, 0))
        shadow_only = add_foot_shadow_only(bg_img_orig.size, (adjusted_x, adjusted_y), person_resized.size)
        # shadow_only = add_foot_shadow(shadow_layer, harmonized, (adjusted_x, adjusted_y))
        shadow_mask = generate_shadow_mask(shadow_only)
        unique_vals = np.unique(np.array(shadow_mask))
        st.write("Unique values in shadow mask:", unique_vals)
        st.image(shadow_mask, caption="Binary Shadow Mask")
        shadow_type = classify_shadow_type(shadow_mask)
        st.image(shadow_mask, caption=f"Shadow Mask - {shadow_type}")
        # final = add_foot_shadow(bg_img_orig.convert("RGBA"), harmonized, (adjusted_x, adjusted_y))
        bg_with_shadow = bg_img_orig.convert("RGBA").copy()
        bg_with_shadow.paste(shadow_only, (0, 0), shadow_only)
        final = paste_person_on_bg(bg_with_shadow, harmonized, (adjusted_x, adjusted_y))

        st.image(harmonized, caption=" Harmonized Person")
        st.image(final, caption=" Final Composite Image")

        buf = io.BytesIO()
        final.save(buf, format="PNG")
        st.download_button(" Download Final Image", buf.getvalue(), "composite.png", "image/png")
else:
    st.info("👆 Please upload both a background and person image to begin.")