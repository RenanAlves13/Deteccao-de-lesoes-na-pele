import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

from utils import Config
from data_loader import DatasetLoader, prepare_data
from models import ShallowModels
from evaluation import ModelEvaluator

def main():
    # Configuração
    config = Config()
    
    # Resultados do paper para comparação
    paper_results_ham10000 = {
        'Global Test Accuracy': '90%',
        'Sensitivity': '88.2%',
        'Specificity': '91.4%',
        'Best AUC': '0.96'
    }
    
    paper_results_isic2019 = {
        'Global Test Accuracy': '87.6%',
        'Best AUC': '0.96'
    }
    
    # Carregador de dados
    loader = DatasetLoader()
    
    # Processa HAM10000
    print("="*80)
    print("PROCESSANDO DATASET HAM10000")
    print("="*80)
    
    X_ham, y_ham = loader.load_ham10000()
    
    if X_ham is not None and y_ham is not None:
        # Prepara dados
        X_ham_scaled, y_ham_encoded, label_encoder_ham, scaler_ham = prepare_data(X_ham, y_ham)
        
        # Split treino/teste
        X_train_ham, X_test_ham, y_train_ham, y_test_ham = train_test_split(
            X_ham_scaled, y_ham_encoded, 
            test_size=0.3, #config.TEST_SIZE
            random_state=42, #config.RANDOM_STATE,
            stratify=y_ham_encoded
        )
        
        print(f"HAM10000 - Treino: {X_train_ham.shape[0]}, Teste: {X_test_ham.shape[0]}")
        print(f"Número de características extraídas: {X_ham_scaled.shape[1]}")
        print(f"Classes: {list(label_encoder_ham.classes_)}")
        
        # Treina modelos
        models_ham = ShallowModels(random_state=config.RANDOM_STATE)
        models_ham.initialize_models()
        models_ham.train_with_grid_search(X_train_ham, y_train_ham, cv=config.CV_FOLDS)
        
        # Avalia modelos
        evaluator_ham = ModelEvaluator()
        evaluator_ham.evaluate_all_models(
            models_ham.get_best_models(), 
            X_test_ham, 
            y_test_ham,
            class_names=label_encoder_ham.classes_
        )
        
        # Resultados
        evaluator_ham.print_results_summary()
        evaluator_ham.plot_results("HAM10000", "results_ham10000_comparison.png")
        evaluator_ham.plot_confusion_matrices(
            label_encoder_ham.classes_, 
            "HAM10000", 
            "confusion_matrices_ham10000.png"
        )
        evaluator_ham.compare_with_paper_results(paper_results_ham10000)
        evaluator_ham.save_results("results_ham10000.pkl")
        
        # Salva modelos e preprocessors
        joblib.dump(models_ham.get_best_models(), "models_ham10000.pkl")
        joblib.dump(label_encoder_ham, "label_encoder_ham10000.pkl")
        joblib.dump(scaler_ham, "scaler_ham10000.pkl")
    
    # Processa ISIC2019
    print("\n" + "="*80)
    print("PROCESSANDO DATASET ISIC2019")
    print("="*80)
    
    X_isic, y_isic = loader.load_isic2019()
    
    if X_isic is not None and y_isic is not None:
        # Prepara dados
        X_isic_scaled, y_isic_encoded, label_encoder_isic, scaler_isic = prepare_data(X_isic, y_isic)
        
        # Split treino/teste
        X_train_isic, X_test_isic, y_train_isic, y_test_isic = train_test_split(
            X_isic_scaled, y_isic_encoded,
            test_size=0.3, #config.TEST_SIZE,
            random_state=42, #config.RANDOM_STATE,
            stratify=y_isic_encoded
        )
        
        print(f"ISIC2019 - Treino: {X_train_isic.shape[0]}, Teste: {X_test_isic.shape[0]}")
        print(f"Número de características extraídas: {X_isic_scaled.shape[1]}")
        print(f"Classes: {list(label_encoder_isic.classes_)}")
        
        # Treina modelos
        models_isic = ShallowModels(random_state=config.RANDOM_STATE)
        models_isic.initialize_models()
        models_isic.train_with_grid_search(X_train_isic, y_train_isic, cv=config.CV_FOLDS)
        
        # Avalia modelos
        evaluator_isic = ModelEvaluator()
        evaluator_isic.evaluate_all_models(
            models_isic.get_best_models(),
            X_test_isic,
            y_test_isic,
            class_names=label_encoder_isic.classes_
        )
        
        # Resultados
        evaluator_isic.print_results_summary()
        evaluator_isic.plot_results("ISIC2019", "results_isic2019_comparison.png")
        evaluator_isic.plot_confusion_matrices(
            label_encoder_isic.classes_,
            "ISIC2019",
            "confusion_matrices_isic2019.png"
        )
        evaluator_isic.compare_with_paper_results(paper_results_isic2019)
        evaluator_isic.save_results("results_isic2019.pkl")
        
        # Salva modelos e preprocessors
        joblib.dump(models_isic.get_best_models(), "models_isic2019.pkl")
        joblib.dump(label_encoder_isic, "label_encoder_isic2019.pkl")
        joblib.dump(scaler_isic, "scaler_isic2019.pkl")

if __name__ == "__main__":
    main()