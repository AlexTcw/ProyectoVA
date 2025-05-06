import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import matplotlib.pyplot as plt
import os

DEFAULT_IMAGE_PATH = "resources/img/"

def load_image(image_name):
    image_path = os.path.join(DEFAULT_IMAGE_PATH, image_name)
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen '{image_name}'. Verifica la ruta.")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    _, sobel_threshold = cv2.threshold(sobel_magnitude, 100, 255, cv2.THRESH_BINARY)
    return sobel_threshold

def dilate_image(binary_image, kernel_size=(4, 4)):
    if binary_image.dtype != np.uint8:
         binary_image = binary_image.astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image

def erode_image(binary_image, kernel_size=(5, 5)):
    if binary_image.dtype != np.uint8:
         binary_image = binary_image.astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    return eroded_image

def morphological_opening(binary_image, kernel_size=(5, 5)):
    if binary_image.dtype != np.uint8:
         binary_image = binary_image.astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    opened_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return opened_image

# --- Nueva función para aislar caracteres usando Componentes Conectados ---
def isolate_characters(binary_image, min_area_ratio=0.001, max_area_ratio=0.08,
                      min_width_ratio=0.01, max_width_ratio=0.15,
                      min_height_ratio=0.05, max_height_ratio=0.4,
                      min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    if binary_image.dtype != np.uint8:
         binary_image = binary_image.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
    output_img = np.zeros_like(binary_image)

    img_height, img_width = binary_image.shape[:2]
    total_area = img_height * img_width
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio
    min_width = img_width * min_width_ratio
    max_width = img_width * max_width_ratio
    min_height = img_height * min_height_ratio
    max_height = img_height * max_height_ratio
    chars_kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = height / width if width > 0 else 0
        is_potential_character = False
        if area >= min_area and area <= max_area:
             if width >= min_width and width <= max_width and \
                height >= min_height and height <= max_height:
                 if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:

                     is_potential_character = True

        if is_potential_character:
            chars_kept += 1

    return output_img

def quitar_puntos_blancos(imagen_binaria, tamaño_kernel=3):
    kernel = np.ones((tamaño_kernel, tamaño_kernel), np.uint8)
    imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)
    return imagen_limpia

def main():
    try:
        img = load_image("Car-plate.jpeg")
        img = preprocess_image(img)
        img = dilate_image(img, kernel_size=(10, 10))
        img = morphological_opening(img,20)
        img = dilate_image(img, kernel_size=(3, 3))


        isolated_chars_img = isolate_characters(
            img,
            min_area_ratio=0.0004,
            max_area_ratio=0.08,
            min_width_ratio=0.005,
            max_width_ratio=0.1,
            min_height_ratio=0.02,
            max_height_ratio=0.2,
            min_aspect_ratio=0.1,
            max_aspect_ratio=10.0
        )

        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagen Binaria Procesada')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(isolated_chars_img, cmap='gray')
        plt.title('Caracteres Aislados')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocurrió un error: {e}")



if __name__ == '__main__':
    main()