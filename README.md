# Replicação: Federated ViT vs Shallow Learning para Classificação de Câncer de Pele

Este projeto replica e compara os resultados do artigo "Federated ViT: A Distributed Deep Learning Framework for Skin Cancer Classification" usando métodos de **shallow learning** ao invés de deep learning.

## Objetivo

Comparar o desempenho de métodos tradicionais de machine learning (shallow learning) com os resultados de Vision Transformers federados apresentados no paper original.

## Datasets

- **HAM10000**: 10.015 imagens, 7 classes de lesões de pele
- **ISIC2019**: 25.331 imagens, 8 classes de lesões de pele

## Metodologia

### Extração de Características
- **Características de Cor**: Estatísticas RGB, histogramas
- **Características de Textura**: Gradientes, Laplaciano, estatísticas
- **Características de Forma**: Área, perímetro, compacidade

### Modelos Testados
1. Random Forest
2. Gradient Boosting
3. Support Vector Machine (SVM)
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Decision Tree
8. AdaBoost

## Execução

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar experimento principal
python main.py

# Análise detalhada dos resultados
python detailed_analysis.py
```

## Estrutura dos Arquivos

```
├── utils.py              # Utilitários e configurações
├── data_loader.py        # Carregamento e processamento dos dados
├── models.py             # Modelos de shallow learning
├── evaluation.py         # Avaliação e métricas
├── main.py               # Script principal
├── detailed_analysis.py  # Análise detalhada
└── requirements.txt      # Dependências
```

## Resultados Esperados

O projeto gerará:
- Comparação de performance entre modelos shallow
- Gráficos comparativos de métricas
- Matrizes de confusão
- Comparação com resultados do paper original
- Análise detalhada por classe

## Comparação com o Paper Original

**Paper (ViT Federado)**:
- HAM10000: 90% acurácia global
- ISIC2019: 87.6% acurácia global

**Este estudo (Shallow Learning)**:
- Resultados serão comparados após execução
```

## Como Executar

1. **Prepare os dados**: Certifique-se de que os datasets estão na estrutura correta
2. **Instale as dependências**: `pip install -r requirements.txt`
3. **Execute o experimento**: `python main.py`
4. **Analise os resultados**: `python detailed_analysis.py`