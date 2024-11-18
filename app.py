import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Paths
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
DB_PATH = os.path.join(FILES_PATH, 'database')

DB_FILE = 'db.csv'       # Nombre del archivo de la base de datos
LABELS_FILE = 'labels.csv'  # Nombre del archivo de etiquetas

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)
    return image_list

def get_labels_df():
    labels_df = pd.read_csv(os.path.join(DB_PATH, LABELS_FILE))
    return labels_df

def calculate_precision_at_n(query_label, retrieved_indices, n, labels_df):
    retrieved_labels = labels_df['label'].iloc[retrieved_indices[:n]].values
    relevant_count = np.sum(retrieved_labels == query_label)
    precision_at_n = relevant_count / n
    return precision_at_n

def calculate_recall_at_n(query_label, retrieved_indices, n, labels_df):
    total_relevant = labels_df[labels_df['label'] == query_label].shape[0]
    if total_relevant == 0:
        return 0.0
    retrieved_labels = labels_df['label'].iloc[retrieved_indices[:n]].values
    relevant_retrieved = np.sum(retrieved_labels == query_label)
    recall_at_n = relevant_retrieved / total_relevant
    return recall_at_n

def calculate_average_precision(query_label, retrieved_indices, labels_df):
    retrieved_labels = labels_df['label'].iloc[retrieved_indices].values
    relevant = (retrieved_labels == query_label).astype(int)
    num_relevant = relevant.sum()
    if num_relevant == 0:
        return 0.0  # Evitar división por cero
    cumulative_precision = 0.0
    relevant_count = 0
    for i, rel in enumerate(relevant):
        if rel:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            cumulative_precision += precision_at_i
    average_precision = cumulative_precision / num_relevant
    return average_precision

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def retrieve_image(img_query, feature_extractor, n_imgs=100):
    import numpy as np
    if feature_extractor == 'Extractor 1 - Histograma de Color':
        # Función para extraer características
        def extract_color_histogram(image_pil):
            import cv2
            # Convertimos PIL Image a OpenCV
            image = np.array(image_pil.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Redimensionamos la imagen
            image = cv2.resize(image, (224, 224))
            # Convertimos a espacio de color HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Calculamos el histograma de color
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                                [0, 180, 0, 256, 0, 256])
            # Normalizamos el histograma
            cv2.normalize(hist, hist)
            return hist.flatten()

        model_feature_extractor = extract_color_histogram
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_1.index'))

    elif feature_extractor == 'Extractor 2 - CNN Pre-entrenada (VGG16)':
        # Función para extraer características
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        from tensorflow.keras.preprocessing import image

        model_cnn = VGG16(weights='imagenet', include_top=False, pooling='max')

        def extract_cnn_features(image_pil):
            img = image_pil.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model_cnn.predict(x)
            features = features.flatten()
            return features

        model_feature_extractor = extract_cnn_features
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_2.index'))

    elif feature_extractor == 'Extractor 3 - Histogramas de Textura (LBP)':
        # Función para extraer características
        from skimage.feature import local_binary_pattern
        import cv2

        def extract_texture_histogram(image_pil):
            # Convertimos PIL Image a escala de grises
            image = np.array(image_pil.convert('L'))

            # Parámetros de LBP
            radius = 3
            n_points = 8 * radius
            method = 'uniform'

            # Calculamos el patrón LBP
            lbp = local_binary_pattern(image, n_points, radius, method)

            # Calculamos el histograma
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

            # Normalizamos el histograma
            hist = hist.astype('float32')
            hist /= (hist.sum() + 1e-6)

            return hist

        model_feature_extractor = extract_texture_histogram
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_3.index'))

    elif feature_extractor == 'Extractor 4 - Histogramas de Gradientes Orientados (HOG)':
        # Función para extraer características HOG
        def extract_hog_features(image_pil):
            import cv2
            # Convertimos PIL Image a escala de grises
            image = np.array(image_pil.convert('L'))
            # Redimensionamos la imagen
            image = cv2.resize(image, (128, 128))
            
            # Definimos el descriptor HOG
            hog = cv2.HOGDescriptor()
            
            # Calculamos el descriptor HOG
            h = hog.compute(image)
            h = h.flatten()
            
            return h
        
        model_feature_extractor = extract_hog_features
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_4.index'))
    
    elif feature_extractor == 'Extractor 5 - ORB (Oriented FAST and Rotated BRIEF)':
        # Función para extraer características ORB
        def extract_orb_features(image_pil):
            import cv2
            # Convertimos PIL Image a escala de grises
            image = np.array(image_pil.convert('L'))
            # Redimensionamos la imagen
            image = cv2.resize(image, (224, 224))
            
            # Creamos el detector ORB
            orb = cv2.ORB_create(nfeatures=500)
            
            # Detectamos los puntos clave y calcular los descriptores
            keypoints, descriptors = orb.detectAndCompute(image, None)
            
            # Si no se detectan descriptores, creamos un vector de ceros
            if descriptors is None:
                descriptors = np.zeros((1, 32), dtype=np.uint8)
            
            # Aplanamos los descriptores en un solo vector
            features = descriptors.flatten()
            
            # Si el vector es más corto que un tamaño fijo, rellenamos con ceros
            max_length = 500 * 32  # 500 características, cada una de 32 bytes
            if features.size < max_length:
                features = np.pad(features, (0, max_length - features.size), 'constant')
            else:
                features = features[:max_length]
            
            return features.astype('float32')
        
        model_feature_extractor = extract_orb_features
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_5.index'))
    else:
        st.error('Extractor de características desconocido.')
        return []

    # Procesamos la imagen de consulta y extraer características
    embeddings = model_feature_extractor(img_query)
    vector = np.array([embeddings]).astype('float32')
    faiss.normalize_L2(vector)

    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]


def main():
    st.title('CBIR IMAGE SEARCH')
    col1, col2 = st.columns(2)
    with col1:
        st.header('CONSULTA')
        st.subheader('Elija el extractor de características')
        # Opciones de extractores
        extractor_options = [
            'Extractor 1 - Histograma de Color',
            'Extractor 2 - CNN Pre-entrenada (VGG16)',
            'Extractor 3 - Histogramas de Textura (LBP)', 
            'Extractor 4 - Histogramas de Gradientes Orientados (HOG)',
            'Extractor 5 - ORB (Oriented FAST and Rotated BRIEF)'
        ]
        option = st.selectbox('Seleccione el extractor de características:', extractor_options)
        st.subheader('Suba una imagen')
        img_file = st.file_uploader(label='.', type=['png', 'jpg', 'jpeg'])

        if img_file:
            img = Image.open(img_file)
            # Obtenemos una imagen recortada desde el frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            # Mostramos vista previa de la imagen recortada
            st.write("Previsualización")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

            labels_df = get_labels_df()
            available_labels = labels_df['label'].unique()
            st.write(f"Etiquetas disponibles: {', '.join(available_labels)}")

            st.subheader('Ingrese la etiqueta de la imagen de consulta')
            query_label = st.text_input('Etiqueta:', '')

    with col2:
        st.header('RESULTADOS')
        if img_file:
            if query_label.strip() == '':
                st.warning("Por favor, ingrese la etiqueta de la imagen de consulta para calcular las métricas.")
            else:
                st.markdown('**Buscando .......**')
                start = time.time()

                retriev = retrieve_image(cropped_img, option, n_imgs=100)
                image_list = get_image_list()
                labels_df = get_labels_df()

                end = time.time()
                st.markdown('**Finalizado en ' + str(round(end - start, 2)) + ' segundos**')

                # Calculamos las métricas
                n = 10 # Número de imágenes a recuperar

                precision_at_n = calculate_precision_at_n(query_label, retriev, n, labels_df)
                recall_at_n = calculate_recall_at_n(query_label, retriev, n, labels_df)
                average_precision = calculate_average_precision(query_label, retriev, labels_df)
                f1_score = calculate_f1_score(precision_at_n, recall_at_n)

                st.write(f"**Precisión@{n}:** {precision_at_n:.2f}")
                st.write(f"**Recall@{n}:** {recall_at_n:.2f}")
                st.write(f"**Average Precision:** {average_precision:.2f}")
                st.write(f"**F1 Score:** {f1_score:.2f}")

                # Mostramos las imágenes recuperadas
                cols = st.columns(5)
                for i, idx in enumerate(retriev[:n]):
                    img_name = image_list[idx]
                    img_path = os.path.join(IMAGES_PATH, img_name)
                    image = Image.open(img_path)
                    cols[i % 5].image(image, caption=img_name, use_column_width=True)

if __name__ == '__main__':
    main()
