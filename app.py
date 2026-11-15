import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import io


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PixelLab",
    page_icon="🖼️",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🖼️ PixelLab")
st.subheader("Interactive Digital Image Processing Application")

st.write(
    "Upload an image and apply different Digital Image "
    "Processing operations."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Image Processing")

operation = st.sidebar.selectbox(
    "Select Operation",
    [
        "Original Image",
        "Grayscale",
        "Negative",
        "Brightness",
        "Contrast",
        "Mean Filter",
        "Median Filter",
        "Gaussian Filter",
        "Min Filter",
        "Max Filter",
        "Sobel Edge Detection",
        "Thresholding",
        "Gamma Correction",
        "Histogram",
        "Histogram Equalization",
        "Image Statistics"
    ]
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)


# =========================================================
# PROCESSING
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Convert image to RGB if required
    if image.mode not in ["RGB", "L"]:
        image = image.convert("RGB")

    processed_image = image

    # =====================================================
    # ORIGINAL
    # =====================================================

    if operation == "Original Image":

        processed_image = image


    # =====================================================
    # GRAYSCALE
    # =====================================================

    elif operation == "Grayscale":

        processed_image = ImageOps.grayscale(image)


    # =====================================================
    # NEGATIVE
    # =====================================================

    elif operation == "Negative":

        gray_image = ImageOps.grayscale(image)

        processed_image = ImageOps.invert(gray_image)


    # =====================================================
    # BRIGHTNESS
    # =====================================================

    elif operation == "Brightness":

        brightness = st.sidebar.slider(
            "Brightness Level",
            0.0,
            3.0,
            1.0,
            0.1
        )

        enhancer = ImageEnhance.Brightness(image)

        processed_image = enhancer.enhance(brightness)


    # =====================================================
    # CONTRAST
    # =====================================================

    elif operation == "Contrast":

        contrast = st.sidebar.slider(
            "Contrast Level",
            0.0,
            3.0,
            1.0,
            0.1
        )

        enhancer = ImageEnhance.Contrast(image)

        processed_image = enhancer.enhance(contrast)


    # =====================================================
    # CONVERT TO GRAYSCALE ARRAY
    # =====================================================

    img_array = np.array(image)

    if len(img_array.shape) == 3:

        gray_array = np.mean(
            img_array[:, :, :3],
            axis=2
        ).astype(np.uint8)

    else:

        gray_array = img_array


    # =====================================================
    # MEAN FILTER
    # =====================================================

    if operation == "Mean Filter":

        kernel_size = st.sidebar.selectbox(
            "Kernel Size",
            [3, 5, 7]
        )

        height, width = gray_array.shape

        pad = kernel_size // 2

        padded = np.pad(
            gray_array,
            pad,
            mode="edge"
        )

        filtered = np.zeros_like(gray_array)

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size
                ]

                filtered[i, j] = np.mean(window)

        processed_image = Image.fromarray(filtered)


    # =====================================================
    # MEDIAN FILTER
    # =====================================================

    elif operation == "Median Filter":

        kernel_size = st.sidebar.selectbox(
            "Kernel Size",
            [3, 5, 7]
        )

        height, width = gray_array.shape

        pad = kernel_size // 2

        padded = np.pad(
            gray_array,
            pad,
            mode="edge"
        )

        filtered = np.zeros_like(gray_array)

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size
                ]

                filtered[i, j] = np.median(window)

        processed_image = Image.fromarray(filtered)


    # =====================================================
    # GAUSSIAN FILTER
    # =====================================================

    elif operation == "Gaussian Filter":

        kernel_size = st.sidebar.selectbox(
            "Kernel Size",
            [3, 5, 7]
        )

        sigma = st.sidebar.slider(
            "Sigma",
            0.1,
            5.0,
            1.0,
            0.1
        )

        center = kernel_size // 2

        x = np.arange(
            -center,
            center + 1
        )

        y = np.arange(
            -center,
            center + 1
        )

        xx, yy = np.meshgrid(x, y)

        gaussian_kernel = np.exp(
            -(xx ** 2 + yy ** 2)
            / (2 * sigma ** 2)
        )

        gaussian_kernel = (
            gaussian_kernel /
            np.sum(gaussian_kernel)
        )

        height, width = gray_array.shape

        pad = kernel_size // 2

        padded = np.pad(
            gray_array,
            pad,
            mode="edge"
        )

        filtered = np.zeros_like(
            gray_array,
            dtype=np.float64
        )

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size
                ]

                filtered[i, j] = np.sum(
                    window * gaussian_kernel
                )

        filtered = np.clip(
            filtered,
            0,
            255
        ).astype(np.uint8)

        processed_image = Image.fromarray(filtered)


    # =====================================================
    # MIN FILTER
    # =====================================================

    elif operation == "Min Filter":

        kernel_size = st.sidebar.selectbox(
            "Kernel Size",
            [3, 5, 7]
        )

        height, width = gray_array.shape

        pad = kernel_size // 2

        padded = np.pad(
            gray_array,
            pad,
            mode="edge"
        )

        filtered = np.zeros_like(gray_array)

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size
                ]

                filtered[i, j] = np.min(window)

        processed_image = Image.fromarray(filtered)


    # =====================================================
    # MAX FILTER
    # =====================================================

    elif operation == "Max Filter":

        kernel_size = st.sidebar.selectbox(
            "Kernel Size",
            [3, 5, 7]
        )

        height, width = gray_array.shape

        pad = kernel_size // 2

        padded = np.pad(
            gray_array,
            pad,
            mode="edge"
        )

        filtered = np.zeros_like(gray_array)

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size
                ]

                filtered[i, j] = np.max(window)

        processed_image = Image.fromarray(filtered)


    # =====================================================
    # SOBEL EDGE DETECTION
    # =====================================================

    elif operation == "Sobel Edge Detection":

        gx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        gy = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        padded = np.pad(
            gray_array,
            1,
            mode="edge"
        )

        height, width = gray_array.shape

        magnitude = np.zeros(
            (height, width),
            dtype=np.float64
        )

        for i in range(height):

            for j in range(width):

                window = padded[
                    i:i + 3,
                    j:j + 3
                ]

                x_value = np.sum(
                    window * gx
                )

                y_value = np.sum(
                    window * gy
                )

                magnitude[i, j] = np.sqrt(
                    x_value ** 2 +
                    y_value ** 2
                )

        magnitude = np.clip(
            magnitude,
            0,
            255
        ).astype(np.uint8)

        processed_image = Image.fromarray(
            magnitude
        )


    # =====================================================
    # THRESHOLDING
    # =====================================================

    elif operation == "Thresholding":

        threshold = st.sidebar.slider(
            "Threshold Value",
            0,
            255,
            128
        )

        thresholded = np.where(
            gray_array >= threshold,
            255,
            0
        ).astype(np.uint8)

        processed_image = Image.fromarray(
            thresholded
        )


    # =====================================================
    # GAMMA CORRECTION
    # =====================================================

    elif operation == "Gamma Correction":

        gamma = st.sidebar.slider(
            "Gamma",
            0.1,
            5.0,
            1.0,
            0.1
        )

        normalized = gray_array / 255.0

        corrected = np.power(
            normalized,
            gamma
        )

        corrected = (
            corrected * 255
        ).astype(np.uint8)

        processed_image = Image.fromarray(
            corrected
        )


    # =====================================================
    # HISTOGRAM EQUALIZATION
    # =====================================================

    elif operation == "Histogram Equalization":

        histogram, bins = np.histogram(
            gray_array.flatten(),
            bins=256,
            range=[0, 256]
        )

        cdf = histogram.cumsum()

        cdf_masked = np.ma.masked_equal(
            cdf,
            0
        )

        cdf_min = cdf_masked.min()

        cdf_max = cdf_masked.max()

        cdf_scaled = (
            (cdf_masked - cdf_min)
            * 255
            / (cdf_max - cdf_min)
        )

        cdf_final = np.ma.filled(
            cdf_scaled,
            0
        ).astype(np.uint8)

        equalized = cdf_final[
            gray_array
        ]

        processed_image = Image.fromarray(
            equalized
        )


    # =====================================================
    # HISTOGRAM
    # =====================================================

    elif operation == "Histogram":

        processed_image = Image.fromarray(
            gray_array
        )


    # =====================================================
    # IMAGE STATISTICS
    # =====================================================

    elif operation == "Image Statistics":

        processed_image = Image.fromarray(
            gray_array
        )


    # =====================================================
    # DISPLAY ORIGINAL + PROCESSED
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )


    with col2:

        st.subheader("✨ Processed Image")

        st.image(
            processed_image,
            use_container_width=True
        )


    # =====================================================
    # HISTOGRAM DISPLAY
    # =====================================================

    if operation == "Histogram":

        st.divider()

        st.subheader("📈 Image Histogram")

        fig = plt.figure()

        plt.hist(
            gray_array.flatten(),
            bins=256,
            range=(0, 256)
        )

        plt.xlabel("Pixel Intensity")

        plt.ylabel("Frequency")

        plt.title("Grayscale Histogram")

        st.pyplot(fig)

        plt.close(fig)


    # =====================================================
    # IMAGE STATISTICS
    # =====================================================

    if operation == "Image Statistics":

        st.divider()

        st.subheader("📊 Image Statistics")

        mean_value = np.mean(gray_array)

        median_value = np.median(gray_array)

        variance_value = np.var(gray_array)

        std_value = np.std(gray_array)

        minimum_value = np.min(gray_array)

        maximum_value = np.max(gray_array)

        stat1, stat2, stat3 = st.columns(3)

        with stat1:

            st.metric(
                "Mean",
                f"{mean_value:.2f}"
            )

            st.metric(
                "Minimum",
                f"{minimum_value}"
            )


        with stat2:

            st.metric(
                "Median",
                f"{median_value:.2f}"
            )

            st.metric(
                "Maximum",
                f"{maximum_value}"
            )


        with stat3:

            st.metric(
                "Variance",
                f"{variance_value:.2f}"
            )

            st.metric(
                "Standard Deviation",
                f"{std_value:.2f}"
            )


    # =====================================================
    # IMAGE INFORMATION
    # =====================================================

    st.divider()

    st.subheader("📋 Image Information")

    width, height = image.size

    info1, info2, info3, info4 = st.columns(4)

    with info1:
        st.metric(
            "Width",
            f"{width}px"
        )

    with info2:
        st.metric(
            "Height",
            f"{height}px"
        )

    with info3:
        st.metric(
            "Format",
            image.format
        )

    with info4:
        st.metric(
            "Mode",
            image.mode
        )


    # =====================================================
    # DOWNLOAD PROCESSED IMAGE
    # =====================================================

    st.divider()

    st.subheader("💾 Download Result")

    buffer = io.BytesIO()

    processed_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Processed Image",
        data=buffer.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )


else:

    st.info(
        "👆 Please upload an image to get started."
    )