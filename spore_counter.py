import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from hashlib import md5
from streamlit_cropper import st_cropper  # Mobile-friendly cropping component
from pillow_heif import register_heif_opener

# Register HEIC opener
register_heif_opener()


# --------------------------
# Helper Function: Format Scientific Notation
# --------------------------
def format_scientific(num, precision=2):
    formatted = f"{num:.{precision}e}"
    mantissa, exponent = formatted.split("e")
    if exponent[0] == '+':
        exponent = exponent[1:]
    exponent = exponent.lstrip("0")
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
    }
    exponent_sup = "".join(superscript_map.get(ch, ch) for ch in exponent)
    return f"{mantissa} × 10{exponent_sup}"


def find_grid_lines(image, debug=False):
    """
    Identify grid lines (assumed to be very bright) and remove them by inpainting.
    
    Parameters:
      image: a BGR OpenCV image.
      debug: if True, show intermediate results via st.image().
    
    Returns:
      inpainted: the image with grid lines removed.
      grid_mask: the mask of detected grid lines.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold: assume grid lines are nearly white.
    _, binary = cv2.threshold(gray, 255 - grid_removal_threshold, 255, cv2.THRESH_BINARY)

    # Use morphological operations to detect vertical lines:
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Detect horizontal lines:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Combine the vertical and horizontal lines to get the grid mask
    grid_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
    grid_mask = cv2.bitwise_not(grid_mask)

    if debug:
        st.image(grid_mask, caption="Grid Mask", use_container_width=True)
    
    return grid_mask

# --------------------------
# Helper Function: process_image
# --------------------------
def process_image(image_bytes=None, pil_img=None, threshold=10, 
                  spore_minArea=20, debug=False):
    """
    Process the image to detect spores.
    If pil_img is provided with pre_cropped=True, the cropping step is skipped
    and the provided image is used directly.
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
    
    # --- Cropping Step ---
    unprocessed_cropped_image = image
    temporary_image = cv2.cvtColor(unprocessed_cropped_image, cv2.COLOR_BGR2GRAY)
    if debug:
        st.image(temporary_image, caption="Gray Cropped Image", use_container_width=True)
    
    grid_mask = find_grid_lines(temporary_image, debug=debug)
    temporary_image[grid_mask == 0] = np.median(temporary_image)  # Remove grid lines from the thresholded image
    if debug:
        st.image(temporary_image, caption="Remove grid", use_container_width=True)
    
    # --- Thresholding ---
    temporary_image = cv2.adaptiveThreshold(
        temporary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, threshold
    )
    if debug:
        st.image(temporary_image, caption="Threshold", use_container_width=True)

    processed_cropped_image = temporary_image

    # --- Blob (Spore) Detection ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = spore_minArea
    params.maxArea = 500
    params.filterByCircularity = False
    params.minCircularity = 0.3
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    spore_keypoints = detector.detect(processed_cropped_image)
    spore_count = len(spore_keypoints)

    output_image = unprocessed_cropped_image
    output_image[grid_mask == 0] = np.median(output_image)  # Remove grid lines from the original image
    spores_detected = cv2.drawKeypoints(
        output_image, spore_keypoints, np.array([]),
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    output_img = cv2.cvtColor(spores_detected, cv2.COLOR_BGR2RGB)
    return output_img, spore_count


# --------------------------
# Manual Adjustment Callback
# --------------------------
def adjust_count(image_index, delta):
    key_manual = f"manual_spore_{image_index}"
    st.session_state[key_manual] = max(0, st.session_state.get(key_manual, 0) + delta)


# --------------------------
# Main App Interface
# --------------------------
st.title("Spore Counter")
st.write("""
Upload one or more images and crop them using the cropper below.
After cropping, tweak the parameters to adjust the spore detection.
The final processed image, spore count, and estimated concentration update live.
You can manually adjust the final spore count using the ➖/➕ buttons, 
which will update the concentration calculations.
""")

# File Uploader and Global Dilution Factor
uploaded_files = st.file_uploader(
    "Choose one or more images...", type=["png", "jpg", "jpeg", "heic", "heif"],
    accept_multiple_files=True
)

# Set dilution factor once
if uploaded_files:
    dilution_factor = st.number_input("Dilution factor", value=10, step=1)

if uploaded_files:
    cropped_images = []
    st.write("### Crop Your Images")
    for idx, file in enumerate(uploaded_files):
        image_pil = Image.open(file).convert("RGB")
        st.subheader(f"Image {idx+1}")
        st.write("Define the crop region below:")
        cropped = st_cropper(image_pil, realtime_update=True, key=f"crop_{idx}", aspect_ratio=(1,1))
        cropped_images.append(cropped)
    
    spore_counts = []
    concentrations = []

    for idx, img in enumerate(cropped_images):
        st.write("---")
        st.header(f"Image {idx+1}")
        # Adjustable parameters for each image
        grid_removal_threshold = st.slider("Grid removal", min_value=0, value=0, max_value=255, key=f"grid_{idx}")       
        threshold = st.slider("Threshold", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"thresh_{idx}")
        spore_minArea = st.number_input("Minimum spore area", value=50, step=1, key=f"min_area_{idx}")
        debug_mode = st.checkbox("Debug", value=False, key=f"debug_{idx}")

        st.write("## Results")
        
        # Process image (cached to avoid heavy reprocessing on every click)
        result_img, auto_count = process_image(
            pil_img=img,
            threshold=threshold,
            spore_minArea=spore_minArea,
            debug=debug_mode
        )
        
        # Use a settings hash to determine whether to reset manual count
        current_settings = (threshold, spore_minArea, dilution_factor)
        settings_hash = hash(current_settings)
        key_hash = f"settings_hash_{idx}"
        key_manual = f"manual_spore_{idx}"
        if key_hash not in st.session_state or st.session_state[key_hash] != settings_hash:
            st.session_state[key_manual] = auto_count
            st.session_state[key_hash] = settings_hash
        
        manual_count = st.session_state[key_manual]
        
        # Display processed image
        st.image(result_img, caption=f"Image {idx+1}: {manual_count} spores detected", use_container_width=True)
        
        # Create a row with the adjustment buttons and spore count display.
        col_minus, col_count, col_plus = st.columns([0.5, 1, 0.5])
        col_minus.button("➖", key=f"minus_{idx}", on_click=adjust_count, args=(idx, -1))
        col_plus.button("➕", key=f"plus_{idx}", on_click=adjust_count, args=(idx, 1))
        col_count.subheader(f"**Spore Count:** {st.session_state[key_manual]}")
        
        # Calculate concentration using the global dilution factor.
        concentration = 250000 * float(dilution_factor) * float(st.session_state[key_manual])
        spore_counts.append(st.session_state[key_manual])
        concentrations.append(concentration)
        st.subheader(f"**Concentration:** {format_scientific(concentration, precision=2)} spores/mL")
    
    # Show average results if multiple images.
    if len(uploaded_files) > 1:
        avg_spore = sum(spore_counts) / len(spore_counts)
        avg_conc = sum(concentrations) / len(concentrations)
        st.write("---")
        st.header("Average Results")
        st.subheader(f"**Average Spore Count:** {avg_spore:.2f}")
        st.subheader(f"**Average Concentration:** {format_scientific(avg_conc, precision=2)} spores/mL")
else:
    st.info("Please upload at least one image.")
