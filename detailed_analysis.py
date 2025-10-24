import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def analyze_results():
    """Análise detalhada dos resultados"""
    
    # Carrega resultados
    try:
        results_ham = joblib.load("results_ham10000.pkl")
        results_isic = joblib.load("results_isic2019.pkl")
        
        print("="*80)
        print("ANÁLISE DETALHADA DOS RESULTADOS")
        print("="*80)
        
        # Análise HAM10000
        print("\nHAM10000 - Análise por Classe:")
        print("-" * 40)
        
        # Carrega label encoder para nomes das classes
        label_encoder_ham = joblib.load("label_encoder_ham10000.pkl")
        
        for model_name, metrics in results_ham.items():
            print(f"\n{model_name}:")
            # Aqui você pode adicionar análises mais detalhadas por classe
            print(f"  Acurácia Global: {metrics['accuracy']:.4f}")
            print(f"  F1-Score Macro: {metrics['f1_score']:.4f}")
            if metrics['auc']:
                print(f"  AUC: {metrics['auc']:.4f}")
        
        # Análise ISIC2019
        print("\n\nISIC2019 - Análise por Classe:")
        print("-" * 40)
        
        label_encoder_isic = joblib.load("label_encoder_isic2019.pkl")
        
        for model_name, metrics in results_isic.items():
            print(f"\n{model_name}:")
            print(f"  Acurácia Global: {metrics['accuracy']:.4f}")
            print(f"  F1-Score Macro: {metrics['f1_score']:.4f}")
            if metrics['auc']:
                print(f"  AUC: {metrics['auc']:.4f}")
        
        # Comparação final
        print("\n" + "="*80)
        print("COMPARAÇÃO FINAL: SHALLOW vs DEEP LEARNING")
        print("="*80)
        
        print("\nHAM10000:")
        print("  Paper (ViT Federado): 90% acurácia, 88.2% sensibilidade, 91.4% especificidade")
        best_ham = max(results_ham.keys(), key=lambda x: results_ham[x]['accuracy'])
        print(f"  Melhor Shallow ({best_ham}): {results_ham[best_ham]['accuracy']:.1%} acurácia")
        
        print("\nISIC2019:")
        print("  Paper (ViT Federado): 87.6% acurácia")
        best_isic = max(results_isic.keys(), key=lambda x: results_isic[x]['accuracy'])
        print(f"  Melhor Shallow ({best_isic}): {results_isic[best_isic]['accuracy']:.1%} acurácia")
        
    except FileNotFoundError as e:
        print(f"Arquivo não encontrado: {e}")
        print("Execute o script main.py primeiro para gerar os resultados.")

if __name__ == "__main__":
    analyze_results()