import os
import cv2
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from skimage.feature import local_binary_pattern

# Definir las rutas
FILES_PATH = os.getcwd()
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
DB_PATH = os.path.join(FILES_PATH, 'database')

# Crear la carpeta database si no existe
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Obtener las rutas de las imágenes y las etiquetas
def get_image_paths(directory):
    image_paths = []
    image_names = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Ruta completa de la imagen
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
                
                # Nombre relativo de la imagen (relativo a IMAGES_PATH)
                rel_dir = os.path.relpath(root, directory)
                rel_file = os.path.join(rel_dir, filename)
                image_names.append(rel_file)
                
                # Extraer la etiqueta (nombre de la subcarpeta)
                label = os.path.basename(root)
                labels.append(label)
    return image_paths, image_names, labels

image_paths, image_names, labels = get_image_paths(IMAGES_PATH)

# Crear el archivo db.csv con los nombres de las imágenes
df_db = pd.DataFrame({'image': image_names})
df_db.to_csv(os.path.join(DB_PATH, 'db.csv'), index=False)

# Crear el archivo labels.csv con los nombres de las imágenes y sus etiquetas
df_labels = pd.DataFrame({'image': image_names, 'label': labels})
df_labels.to_csv(os.path.join(DB_PATH, 'labels.csv'), index=False)

# ----------------------------------------
# Definir las funciones de extracción de características
# ----------------------------------------

# Extractor 1: Histogramas de Color
def extract_color_histogram(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    # Redimensionar la imagen
    image = cv2.resize(image, (224, 224))
    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calcular el histograma de color
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    # Normalizar el histograma
    cv2.normalize(hist, hist)
    return hist.flatten()

# Extractor 2: CNN Pre-entrenada (VGG16)
# Cargar el modelo pre-entrenado
model_cnn = VGG16(weights='imagenet', include_top=False, pooling='max')

def extract_cnn_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_cnn.predict(x)
    features = features.flatten()
    return features

# Extractor 3: Histogramas de Textura (LBP)
def extract_texture_histogram(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar la imagen
    image = cv2.resize(image, (224, 224))

    # Parámetros de LBP
    radius = 3
    n_points = 8 * radius
    method = 'uniform'

    # Calcular el patrón LBP
    lbp = local_binary_pattern(image, n_points, radius, method)

    # Calcular el histograma
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalizar el histograma
    hist = hist.astype('float32')
    hist /= (hist.sum() + 1e-6)

    return hist

# Extractor 4 : HOG (Histogram of Oriented Gradients)   
def extract_hog_features(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar la imagen a un tamaño uniforme
    image = cv2.resize(image, (128, 128))
    
    # Definir el descriptor HOG
    hog = cv2.HOGDescriptor()
    
    # Calcular el descriptor HOG
    h = hog.compute(image)
    h = h.flatten()
    
    return h

# Extractor 5 : ORB
def extract_orb_features(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar la imagen
    image = cv2.resize(image, (224, 224))
    
    # Crear el detector ORB
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detectar los puntos clave y calcular los descriptores
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # Si no se detectan descriptores, crear un vector de ceros
    if descriptors is None:
        descriptors = np.zeros((1, 32), dtype=np.uint8)
    
    # Aplanar los descriptores en un solo vector
    features = descriptors.flatten()
    
    # Si el vector es más corto que un tamaño fijo, rellenar con ceros
    max_length = 500 * 32  # 500 características, cada una de 32 bytes
    if features.size < max_length:
        features = np.pad(features, (0, max_length - features.size), 'constant')
    else:
        features = features[:max_length]
    
    return features.astype('float32')

# ----------------------------------------
# Extracción de características y creación de índices
# ----------------------------------------

# Lista de extractores y sus funciones
extractors = [
    ('feat_extract_1', extract_color_histogram),
    ('feat_extract_2', extract_cnn_features),
    ('feat_extract_3', extract_texture_histogram),
    ('feat_extract_4', extract_hog_features),
    ('feat_extract_5', extract_orb_features),  
]

for extractor_name, extractor_function in extractors:
    # Extraer características
    features_list = []
    for img_path in image_paths:
        features = extractor_function(img_path)
        features_list.append(features)
    features_array = np.array(features_list).astype('float32')

    # Normalizar las características
    features_array = normalize(features_array, norm='l2')

    # Crear el índice FAISS
    d = features_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features_array)

    # Guardar el índice
    index_filename = f'{extractor_name}.index'
    faiss.write_index(index, os.path.join(DB_PATH, index_filename))