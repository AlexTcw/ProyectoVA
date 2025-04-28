import cv2
import os

import numpy as np

DEFAULT_IMAGE_PATH = "resources/img/"
N_REGIONS = 4

def load_image(image_name):
    image_path = os.path.join(DEFAULT_IMAGE_PATH, image_name)
    print("Loading image:", image_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")

    return image

def image_to_matrix(image):
    print("Converting image to matrix")
    if image is None:
        print("Error: No hay imagen para mostrar.")
        return None
    else:
        return np.array(image)

def show_images(images):
    if not images:
        print("Error: No hay imagenes para mostrar.")
        return
    for image, window_name, width, height in images:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def matrix_to_image(matrix, output_path=None):
    if matrix is None:
        print("Error: La matriz no es válida.")
        return None

    # Normalizar la matriz si es de tipo float (convertir a valores entre 0 y 255)
    if matrix.dtype == np.float32 or matrix.dtype == np.float64:
        matrix = np.clip(matrix, 0, 255).astype(np.uint8)

    # Guardar la imagen si se especifica una ruta
    if output_path:
        cv2.imwrite(output_path, matrix)

    return matrix

def convert_to_bw(matrix):
    print("Converting matrix to black and white")

    if matrix is None:
        print("Error: No hay imagen para procesar.")
        return None

    height, width, _ = matrix.shape
    bw_matrix = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r, g, b = matrix[i][j]
            gray = (r * 0.3 + g * 0.59 + b * 0.11)  # Conversión a escala de grises
            bw_matrix[i][j] = 255 if gray > 128 else 0  # Binarización manual

    return bw_matrix

def compute_histogram(matrix):
    """Calcula el histograma de la imagen manualmente."""
    height, width = matrix.shape
    histogram = np.zeros(256, dtype=int)

    for i in range(height):
        for j in range(width):
            intensity = matrix[i, j]
            histogram[intensity] += 1

    return histogram

def find_thresholds(histogram, n_regions):
    """Encuentra N-1 umbrales dividiendo el histograma en regiones de igual área."""
    total_pixels = sum(histogram)
    region_size = total_pixels // n_regions
    thresholds = []

    accumulated = 0
    for i in range(256):
        accumulated += histogram[i]
        if accumulated >= (len(thresholds) + 1) * region_size:
            thresholds.append(i)

    return thresholds

def apply_segmentation(matrix, thresholds):
    """Segmenta la imagen asignando niveles de gris según los umbrales."""
    segmented = np.zeros_like(matrix, dtype=np.uint8)
    regions = len(thresholds) + 1

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pixel = matrix[i, j]
            for t in range(len(thresholds)):
                if pixel <= thresholds[t]:
                    segmented[i, j] = (255 // (regions - 1)) * t
                    break
            else:
                segmented[i, j] = 255  # Última región

    return segmented


def main():
    src_img = load_image("Car-plate.jpeg")
    src_matrix = image_to_matrix(src_img)
    bw_matrix = convert_to_bw(src_matrix)
    histogram = compute_histogram(bw_matrix)
    thresholds = find_thresholds(histogram, N_REGIONS)
    segmented_img = apply_segmentation(bw_matrix, thresholds)
    dst_img = matrix_to_image(segmented_img)
    show_images([(src_img,"src",1000,600),(dst_img,"dst",1000,600)])

if __name__ == '__main__':
    main()