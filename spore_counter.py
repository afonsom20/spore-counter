import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import streamlit.elements.image as st_image
import base64
from io import BytesIO
from hashlib import md5


# --- Patch for st_canvas ---
# Some versions expect st.image.image_to_url. We define it here.
def pil_to_data_url(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def image_to_url(image, width, preserveAspectRatio, color_format, output_format, key):
    return pil_to_data_url(image)


st_image.image_to_url = image_to_url


# --------------------------
# Helper Function: process_image
# --------------------------
def process_image(image_bytes, initial_thresh=50, adaptive_thresh_val=10, zoom_margin=30,
                  spore_minArea=20, spore_minThreshold=0.1, advanced=False, contrast=1.0,
                  crop_mode="manual", manual_crop_coords=None):
    """
    Process the uploaded image to detect spores.

    Parameters:
      - image_bytes: Bytes from the uploaded image.
      - initial_thresh: Threshold for initial contour detection.
      - adaptive_thresh_val: 'C' value for adaptive thresholding.
      - zoom_margin: Margin (in pixels) for zoom cropping.
      - spore_minArea: Minimum area for blob detection.
      - spore_minThreshold: Minimum threshold for blob detection.
      - advanced: If True, displays intermediate debugging images.
      - contrast: Contrast factor (1.0 = no change).
      - crop_mode: "automatic" or "manual".
      - manual_crop_coords: Tuple (x, y, w, h) from manual cropping.

    Returns:
      - output_img: Processed image (with spore keypoints drawn).
      - spore_keypoints: List of detected spore keypoints.
    """
    # Convert uploaded bytes to an OpenCV image.
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_image = image.copy()  # backup

    # --- Contrast Enhancement ---
    if contrast != 1.0:
        # Increase contrast using CLAHE on the V channel in HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # --- Cropping Step ---
    if crop_mode == "manual" and manual_crop_coords is not None:
        # Use manually drawn rectangle coordinates.
        x, y, w, h = manual_crop_coords
        final_crop = image[y:y + h, x:x + w]
    else:
        # Automatic cropping: crop to the main circular area using largest contour.
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
    if crop_mode == "automatic":
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

        # Helper: compute intersection point of two lines.
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
# Streamlit App Interface
# --------------------------
st.title("Spore Counter")
st.write("Upload an image, and the app will process it and display the results.")

# --- Cropping Mode Selection ---
crop_mode = st.radio("Cropping Mode", options=["Manual","Automatic (Experimental)"], index=0)
crop_mode_key = "manual" if crop_mode == "Manual" else "automatic"

# --- Advanced Options ---
advanced_mode = st.toggle("Advanced mode")
if advanced_mode:
    contrast_factor = st.slider("Contrast factor", value=1.0, min_value=0.5, max_value=3.0)
    adaptive_thresh_val = st.number_input("Adaptive threshold value", value=10.0, step=0.1)
    zoom_margin = st.number_input("Zoom margin", value=30, step=1)
    spore_minArea = st.number_input("Minimum spore area", value=20, step=1)
    spore_minThreshold = st.number_input("Minimum spore threshold", value=0.1, step=0.01)
    canvas_scale = st.number_input("Contrast factor", value=10, step=1)
    initial_thresh = st.number_input("Initial threshold sensitivity", value=50, step=1)
else:
    contrast_factor=1.0
    adaptive_thresh_val = 10.0
    spore_minThreshold = 0.1
    zoom_margin = 30
    spore_minArea = 20
    canvas_scale = 10
    initial_thresh = 50


# --- Dilution Factor ---
dilution_factor = st.number_input("Dilution factor", value=10, step=1)

# --- File Uploader (Single Image) ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
original_file = uploaded_file

# --- Manual Cropping Mode: Draw Rectangle via Canvas ---
manual_coords = None
if uploaded_file is not None and crop_mode_key == "manual":
    st.write("Draw a rectangle on the image to define the crop region.")
    pil_image = Image.open(uploaded_file).convert("RGB")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        width=pil_image.width / canvas_scale,
        height=pil_image.height / canvas_scale,
        drawing_mode="rect",
        key="canvas_rect"
    )
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            rect = next((obj for obj in objects if obj["type"] == "rect"), None)
            if rect:
                x = int(rect["left"]) * canvas_scale
                y = int(rect["top"]) * canvas_scale
                w = int(rect["width"]) * canvas_scale
                h = int(rect["height"]) * canvas_scale
                manual_coords = (x, y, w, h)
                st.success(f"Crop area: (x={x}, y={y}, w={w}, h={h})")
            else:
                st.info("Please draw a rectangle on the image.")
        else:
            st.info("Please draw a rectangle on the image.")

# --- Process Image Button ---
if original_file is not None:
    # Reset the file pointer to the beginning
    original_file.seek(0)
    if st.button("🔬 Process and Count Spores"):
        result_image, spore_keypoints = process_image(
            image_bytes=original_file,
            crop_mode=crop_mode_key,
            manual_crop_coords=manual_coords,
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
        st.header(f"Concentration = {formatted_concentration} spores/mL")

else:
    st.info("Please upload an image to begin.")
