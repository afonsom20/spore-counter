import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from hashlib import md5
from streamlit_cropper import st_cropper  # New mobile-friendly cropper


# --------------------------
# Helper Function: Format Scientific Notation
# --------------------------
def format_scientific(num, precision=2):
    formatted = f"{num:.{precision}e}"
    mantissa, exponent = formatted.split("e")
    if exponent[0] == '+':
        exponent = exponent[1:]
    exponent = exponent.lstrip("0")
    superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸',
                       '9': '⁹', '-': '⁻'}
    exponent_sup = "".join(superscript_map.get(ch, ch) for ch in exponent)
    return f"{mantissa} × 10{exponent_sup}"

# --------------------------
# Helper Function: process_image
# --------------------------
def process_image(image_bytes=None, pil_img=None, pre_cropped=False,
                  initial_thresh=50, adaptive_thresh_val=10, zoom_margin=30,
                  spore_minArea=20, spore_minThreshold=0.1, advanced=False, contrast=1.0,
                  crop_mode="manual", manual_crop_coords=None):
    """
    Process the image to detect spores.

    If pil_img is provided (and pre_cropped=True), it will bypass the internal cropping
    and use the supplied cropped image.

    Parameters:
      - image_bytes: Bytes from the uploaded image (if no PIL image is provided).
      - pil_img: A PIL.Image instance (if provided, used instead of image_bytes).
      - pre_cropped: If True, skips the cropping steps and uses the input image directly.
      - initial_thresh: Threshold for initial contour detection.
      - adaptive_thresh_val: 'C' value for adaptive thresholding.
      - zoom_margin: Margin (in pixels) for zoom cropping.
      - spore_minArea: Minimum area for blob detection.
      - spore_minThreshold: Minimum threshold for blob detection.
      - advanced: If True, displays intermediate debugging images.
      - contrast: Contrast factor (1.0 = no change).
      - crop_mode: "automatic" or "manual".
      - manual_crop_coords: Tuple (x, y, w, h) for manual cropping (ignored if pre_cropped is True).

    Returns:
      - output_img: Processed image (with spore keypoints drawn).
      - spore_keypoints: List of detected spore keypoints.
    """
    # Convert input to an OpenCV image.
    if pil_img is not None:
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif image_bytes is not None:
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        st.error("No image provided!")
        return None, []

    orig_image = image.copy()  # backup

    # --- Contrast Enhancement ---
    if contrast != 1.0:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # --- Cropping Step ---
    if pre_cropped:
        # Use the already cropped image (from st_cropper)
        final_crop = image
    else:
        if crop_mode == "manual" and manual_crop_coords is not None:
            x, y, w, h = manual_crop_coords
            final_crop = image[y:y + h, x:x + w]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh_auto = cv2.threshold(gray, initial_thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_auto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_image = image[y:y + h, x:x + w]
            else:
                cropped_image = orig_image
            h_img, w_img = cropped_image.shape[:2]
            min_side = min(h_img, w_img)
            start_x = (w_img - min_side) // 2
            start_y = (h_img - min_side) // 2
            final_crop = cropped_image[start_y:start_y + min_side, start_x:start_x + min_side]

    if advanced:
        st.image(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB),
                 caption="Cropped Image", use_container_width=True)

    # --- Preprocessing for Boundary Detection ---
    gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    eroded = cv2.erode(closed, erode_kernel, iterations=2)
    if advanced:
        st.image(cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR),
                 caption="After Erosion", use_container_width=True)

    # --- Automatic Cropping (Experimental) ---
    if not pre_cropped and crop_mode == "automatic":
        h_img, w_img = final_crop.shape[:2]
        minLineLength = int(0.8 * w_img)
        maxLineGap = 20
        linesP = cv2.HoughLinesP(eroded, 1, np.pi / 180, threshold=10,
                                 minLineLength=minLineLength, maxLineGap=maxLineGap)
        horizontals = []
        verticals = []
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 10:
                    horizontals.append((x1, y1, x2, y2))
                elif abs(abs(angle) - 90) <= 10:
                    verticals.append((x1, y1, x2, y2))
        if horizontals:
            horizontals = sorted(horizontals, key=lambda l: (l[1] + l[3]) / 2)
            top_line = horizontals[0]
            bottom_line = horizontals[-1]
        else:
            top_line = bottom_line = None
        if verticals:
            verticals = sorted(verticals, key=lambda l: (l[0] + l[2]) / 2)
            left_line = verticals[0]
            right_line = verticals[-1]
        else:
            left_line = right_line = None

        def intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if denom == 0:
                return None
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))

        top_left_pt = intersection(top_line, left_line) if top_line and left_line else None
        top_right_pt = intersection(top_line, right_line) if top_line and right_line else None
        bottom_left_pt = intersection(bottom_line, left_line) if bottom_line and left_line else None
        bottom_right_pt = intersection(bottom_line, right_line) if bottom_line and right_line else None

        if advanced:
            debug_img = final_crop.copy()
            if top_left_pt:
                cv2.circle(debug_img, (int(round(top_left_pt[0])), int(round(top_left_pt[1]))), 30, (0, 0, 255), -1)
            if top_right_pt:
                cv2.circle(debug_img, (int(round(top_right_pt[0])), int(round(top_right_pt[1]))), 30, (0, 0, 255), -1)
            if bottom_left_pt:
                cv2.circle(debug_img, (int(round(bottom_left_pt[0])), int(round(bottom_left_pt[1]))), 30, (0, 0, 255),
                           -1)
            if bottom_right_pt:
                cv2.circle(debug_img, (int(round(bottom_right_pt[0])), int(round(bottom_right_pt[1]))), 30, (0, 0, 255),
                           -1)
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                     caption="Detected Corners (Auto Mode)", use_container_width=True)

        if all([top_left_pt, top_right_pt, bottom_left_pt, bottom_right_pt]):
            xs = [top_left_pt[0], top_right_pt[0], bottom_left_pt[0], bottom_right_pt[0]]
            ys = [top_left_pt[1], top_right_pt[1], bottom_left_pt[1], bottom_right_pt[1]]
            crop_x = max(0, int(round(min(xs))) + zoom_margin)
            crop_y = max(0, int(round(min(ys))) + zoom_margin)
            crop_x2 = min(w_img, int(round(max(xs))) - zoom_margin)
            crop_y2 = min(h_img, int(round(max(ys))) - zoom_margin)
            if crop_x2 > crop_x and crop_y2 > crop_y:
                cropped_square = final_crop[crop_y:crop_y2, crop_x:crop_x2]
                final_crop = cropped_square
                if advanced:
                    st.image(cv2.cvtColor(cropped_square, cv2.COLOR_BGR2RGB),
                             caption="Auto-Cropped", use_container_width=True)
            else:
                st.write("Zoom margin too high; using unzoomed region.")
        else:
            st.write("Could not compute all corners; using full cropped image.")

    # --- Further Processing for Quadrant Detection ---
    gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
    _, thresh3 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    adaptive_thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, 21, adaptive_thresh_val)
    lines = cv2.HoughLinesP(adaptive_thresh_img, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=200)
    if lines is not None:
        x_coords, y_coords = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        min_x_auto = min(x_coords)
        max_x_auto = max(x_coords)
        min_y_auto = min(y_coords)
        max_y_auto = max(y_coords)
        cropped_square = adaptive_thresh_img[min_y_auto:max_y_auto, min_x_auto:max_x_auto]
    else:
        cropped_square = adaptive_thresh_img
    if advanced:
        st.image(cropped_square, caption="Quadrant Detection - Grayscale", use_container_width=True)

    # --- Blob (Spore) Detection ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = spore_minArea
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = spore_minThreshold
    detector = cv2.SimpleBlobDetector_create(params)
    spore_keypoints = detector.detect(cropped_square)
    spores_detected = cv2.drawKeypoints(final_crop, spore_keypoints, np.array([]),
                                        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_img = cv2.cvtColor(spores_detected, cv2.COLOR_BGR2RGB)
    return output_img, spore_keypoints







# --------------------------
# Streamlit App Interface
# --------------------------
st.title("Spore Counter")
st.write("Upload an image, adjust the crop using the cropper below, and the app will process it.")

# --- File Uploader (Single Image) ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image as a PIL image
    image_pil = Image.open(uploaded_file).convert("RGB")

    # --- Cropping using Streamlit Cropper ---
    st.write("Use the cropper below to define the crop region:")
    cropped_img = st_cropper(image_pil, realtime_update=True, aspect_ratio=None)

    # --- Advanced Options ---
    advanced_mode = st.toggle("Advanced mode")
    if advanced_mode:
        contrast_factor = st.slider("Contrast factor", value=1.0, min_value=0.5, max_value=3.0)
        adaptive_thresh_val = st.number_input("Adaptive threshold value", value=10.0, step=0.1)
        zoom_margin = st.number_input("Zoom margin", value=30, step=1)
        spore_minArea = st.number_input("Minimum spore area", value=20, step=1)
        spore_minThreshold = st.number_input("Minimum spore threshold", value=0.1, step=0.01)
        initial_thresh = st.number_input("Initial threshold sensitivity", value=50, step=1)
    else:
        contrast_factor = 1.0
        adaptive_thresh_val = 10.0
        spore_minThreshold = 0.1
        zoom_margin = 30
        spore_minArea = 20
        initial_thresh = 50

    # --- Dilution Factor ---
    dilution_factor = st.number_input("Dilution factor", value=10, step=1)

    # --- Process Image Button ---
    if st.button("🔬 Process and Count Spores"):
        # Process the image using the cropped image (pre_cropped=True)
        result_image, spore_keypoints = process_image(
            pil_img=cropped_img,
            pre_cropped=True,
            initial_thresh=initial_thresh,
            adaptive_thresh_val=adaptive_thresh_val,
            zoom_margin=zoom_margin,
            spore_minArea=spore_minArea,
            spore_minThreshold=spore_minThreshold,
            contrast=contrast_factor,
            advanced=advanced_mode
        )
        st.image(result_image, caption=f"Processed Image - {len(spore_keypoints)} spores detected",
                 use_container_width=True)
        spore_count = len(spore_keypoints)
        st.header(f"Spores Detected: {spore_count}")
        concentration = 250000 / 16 * dilution_factor * spore_count  # 250000 for chamber, 16 for quadrants
        formatted_concentration = format_scientific(concentration, precision=2)
        st.header(f"Estimated Concentration = {formatted_concentration} spores/mL")
else:
    st.info("Please upload an image to begin.")

