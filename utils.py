import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configurações globais do projeto"""
    # Caminhos dos datasets
    HAM10000_PATH = "datasets/HAM10000"
    ISIC2019_PATH = "datasets/ISIC"
    
    # Parâmetros de processamento
    IMG_SIZE = (224, 224)  # Mesmo tamanho usado no ViT
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Classes HAM10000 (7 classes como no paper)
    HAM10000_CLASSES = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma', 
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }
    
    # Classes ISIC2019 (8 classes como no paper)
    ISIC2019_CLASSES = {
        'MEL': 'Melanoma',
        'NV': 'Melanocytic nevus',
        'BCC': 'Basal cell carcinoma',
        'AK': 'Actinic keratosis',
        'BKL': 'Benign keratosis',
        'DF': 'Dermatofibroma',
        'VASC': 'Vascular lesion',
        'SCC': 'Squamous cell carcinoma'
    }

def load_image(image_path, target_size=(224, 224)):
    """Carrega e pré-processa uma imagem"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Erro ao carregar imagem {image_path}: {e}")
        return None

def extract_color_features(image):
    """Extrai características de cor da imagem"""
    features = []
    
    # Estatísticas básicas por canal RGB
    for channel in range(3):
        channel_data = image[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.median(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75)
        ])
    
    # Histograma de cores (bins reduzidos para eficiência)
    hist_r = cv2.calcHist([image], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [16], [0, 256])
    
    features.extend(hist_r.flatten())
    features.extend(hist_g.flatten())
    features.extend(hist_b.flatten())
    
    return np.array(features)

def extract_texture_features(image):
    """Extrai características de textura usando filtros simples"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    
    # Gradientes
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    features.extend([
        np.mean(np.abs(grad_x)),
        np.std(grad_x),
        np.mean(np.abs(grad_y)),
        np.std(grad_y)
    ])
    
    # Laplaciano (detecção de bordas)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([
        np.mean(np.abs(laplacian)),
        np.std(laplacian)
    ])
    
    # Estatísticas básicas da imagem em escala de cinza
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 75)
    ])
    
    return np.array(features)

def extract_shape_features(image):
    """Extrai características de forma simples"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Binarização simples
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    if contours:
        # Maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        features.extend([
            area,
            perimeter,
            area / (perimeter + 1e-7),  # Compacidade
            len(largest_contour)  # Número de pontos do contorno
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    return np.array(features)