import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
from utils import Config, load_image, extract_color_features, extract_texture_features, extract_shape_features

class DatasetLoader:
    def __init__(self):
        self.config = Config()
    
    def load_ham10000(self):
        """Carrega o dataset HAM10000"""
        print("Carregando dataset HAM10000...")
        
        # Carrega metadados
        metadata_path = os.path.join(self.config.HAM10000_PATH, "HAM10000_metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Arquivo de metadados não encontrado: {metadata_path}")
            return None, None
        
        metadata = pd.read_csv(metadata_path)
        
        # Diretórios de imagens
        img_dirs = [
            os.path.join(self.config.HAM10000_PATH, "HAM10000_images_part_1"),
            os.path.join(self.config.HAM10000_PATH, "HAM10000_images_part_2")
        ]
        
        features_list = []
        labels_list = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processando HAM10000"):
            image_id = row['image_id']
            label = row['dx']
            
            # Procura a imagem nos diretórios
            image_path = None
            for img_dir in img_dirs:
                potential_path = os.path.join(img_dir, f"{image_id}.jpg")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                continue
            
            # Carrega e processa a imagem
            image = load_image(image_path, self.config.IMG_SIZE)
            if image is None:
                continue
            
            # Extrai características
            color_feat = extract_color_features(image)
            texture_feat = extract_texture_features(image)
            shape_feat = extract_shape_features(image)
            
            # Combina todas as características
            features = np.concatenate([color_feat, texture_feat, shape_feat])
            
            features_list.append(features)
            labels_list.append(label)
        
        print(f"HAM10000: {len(features_list)} imagens processadas")
        return np.array(features_list), np.array(labels_list)
    
    def load_isic2019(self):
        """Carrega o dataset ISIC2019"""
        print("Carregando dataset ISIC2019...")
        
        # Carrega metadados
        metadata_path = os.path.join(self.config.ISIC2019_PATH, "ISIC_2019_Training_GroundTruth.csv")
        if not os.path.exists(metadata_path):
            print(f"Arquivo de metadados não encontrado: {metadata_path}")
            return None, None
        
        metadata = pd.read_csv(metadata_path)
        
        # Diretório de imagens
        img_dir = os.path.join(self.config.ISIC2019_PATH, "ISIC_2019_Training_Input")
        
        features_list = []
        labels_list = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processando ISIC2019"):
            image_id = row['image']
            
            # Encontra a classe (coluna com valor 1)
            class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
            label = None
            for col in class_cols:
                if row[col] == 1:
                    label = col
                    break
            
            if label is None:
                continue
            
            # Carrega imagem
            image_path = os.path.join(img_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue
            
            image = load_image(image_path, self.config.IMG_SIZE)
            if image is None:
                continue
            
            # Extrai características
            color_feat = extract_color_features(image)
            texture_feat = extract_texture_features(image)
            shape_feat = extract_shape_features(image)
            
            # Combina todas as características
            features = np.concatenate([color_feat, texture_feat, shape_feat])
            
            features_list.append(features)
            labels_list.append(label)
        
        print(f"ISIC2019: {len(features_list)} imagens processadas")
        return np.array(features_list), np.array(labels_list)

def prepare_data(features, labels):
    """Prepara os dados para treinamento"""
    # Codifica labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Normaliza características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels_encoded, label_encoder, scaler