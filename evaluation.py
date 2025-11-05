import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
import joblib


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, model_name, X_test, y_test, class_names=None):
        """Avalia um modelo individual"""

        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = None

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # AUC (para multiclasse)
        auc_score = None
        if y_pred_proba is not None:
            try:
                # Binariza as labels para cálculo do AUC
                n_classes = len(np.unique(y_test))
                y_test_bin = label_binarize(y_test, classes=range(n_classes))

                if n_classes == 2:
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_test_bin, y_pred_proba,
                                              average='macro', multi_class='ovr')
            except:
                auc_score = None

        # Armazena resultados
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return self.results[model_name]

    def evaluate_all_models(self, models_dict, X_test, y_test, class_names=None):
        """Avalia todos os modelos"""
        print("Avaliando modelos...")

        for name, model_info in models_dict.items():
            model = model_info['model']
            print(f"Avaliando {name}...")
            self.evaluate_model(model, name, X_test, y_test, class_names)

    def print_results_summary(self):
        """Imprime resumo dos resultados"""
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS")
        print("="*80)

        # Cria DataFrame com resultados
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Modelo': model_name,
                'Acurácia': f"{metrics['accuracy']:.4f}",
                'Precisão': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'AUC': f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
            })

        df_results = pd.DataFrame(results_data)
        print(df_results.to_string(index=False))

        # Melhor modelo por métrica
        print(
            f"\nMelhor Acurácia: {max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])}")
        print(
            f"Melhor F1-Score: {max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])}")

        if any(self.results[x]['auc'] for x in self.results if self.results[x]['auc']):
            best_auc = max([x for x in self.results.keys() if self.results[x]['auc']],
                           key=lambda x: self.results[x]['auc'])
            print(f"Melhor AUC: {best_auc}")

    def plot_results(self, dataset_name="Dataset", save_path=None):
        """Plota gráficos comparativos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f'Comparação de Modelos - {dataset_name}', fontsize=16, fontweight='bold')

        # Prepara dados para plotagem
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx % 2]
            values = [self.results[model][metric] for model in models]

            bars = ax.bar(models, values, alpha=0.7, color=plt.cm.Set3(
                np.linspace(0, 1, len(models))))
            ax.set_title(metric_name, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # Adiciona valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self, class_names=None, dataset_name="Dataset", save_path=None):
        """Plota matrizes de confusão para todos os modelos"""
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle(
            f'Matrizes de Confusão - {dataset_name}', fontsize=16, fontweight='bold')

        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (model_name, results) in enumerate(self.results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f'{model_name}\nAcc: {results["accuracy"]:.3f}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')

        # Remove subplots vazios
        for idx in range(n_models, rows * cols):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.remove()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def compare_with_paper_results(self, paper_results):
        """Compara resultados com os do paper"""
        print("\n" + "="*80)
        print("COMPARAÇÃO COM RESULTADOS DO PAPER")
        print("="*80)

        print(f"Resultados do Paper (ViT Federado):")
        for metric, value in paper_results.items():
            print(f"  {metric}: {value}")

        print(f"\nMelhores Resultados dos Modelos Shallow:")
        best_acc = max(self.results.keys(),
                       key=lambda x: self.results[x]['accuracy'])
        best_f1 = max(self.results.keys(),
                      key=lambda x: self.results[x]['f1_score'])

        print(
            f"  Melhor Acurácia: {best_acc} ({self.results[best_acc]['accuracy']:.4f})")
        print(
            f"  Melhor F1-Score: {best_f1} ({self.results[best_f1]['f1_score']:.4f})")

        if 'auc' in paper_results:
            best_auc_models = [
                x for x in self.results.keys() if self.results[x]['auc']]
            if best_auc_models:
                best_auc = max(best_auc_models,
                               key=lambda x: self.results[x]['auc'])
                print(
                    f"  Melhor AUC: {best_auc} ({self.results[best_auc]['auc']:.4f})")

    def save_results(self, filepath):
        """Salva resultados em arquivo"""
        joblib.dump(self.results, filepath)
        print(f"Resultados salvos em: {filepath}")

    def load_results(self, filepath):
        """Carrega resultados de arquivo"""
        self.results = joblib.load(filepath)
        print(f"Resultados carregados de: {filepath}")
