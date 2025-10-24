from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class ShallowModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        
    def initialize_models(self):
        """Inicializa os modelos com hiperparâmetros para busca"""
        
        # Random Forest
        self.models['Random Forest'] = {
            'model': RandomForestClassifier(random_state=self.random_state),
            'params': {
                # UTILIZANDO MENOS PARÂMETROS
                'n_estimators': [100, 200], #[100, 200, 300]
                'max_depth': [10, None], #[10, 20, None]
                'min_samples_split': [2, 5], #[2, 5, 10]
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = {
            'model': GradientBoostingClassifier(random_state=self.random_state),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        # Support Vector Machine
        self.models['SVM'] = {
            'model': SVC(random_state=self.random_state, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Logistic Regression
        self.models['Logistic Regression'] = {
            'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        # K-Nearest Neighbors
        self.models['KNN'] = {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        # Naive Bayes
        self.models['Naive Bayes'] = {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
        
        # Decision Tree
        self.models['Decision Tree'] = {
            'model': DecisionTreeClassifier(random_state=self.random_state),
            'params': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # AdaBoost
        self.models['AdaBoost'] = {
            'model': AdaBoostClassifier(random_state=self.random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        }
    
    def train_with_grid_search(self, X_train, y_train, cv=5, n_jobs=-1):
        """Treina modelos com busca em grade"""
        print("Iniciando treinamento com Grid Search...")
        
        for name, model_info in self.models.items():
            print(f"\nTreinando {name}...")
            
            # Grid Search
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=3, #cv
                scoring='accuracy',
                n_jobs=6, #Específico para a instância da AWS
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Armazena o melhor modelo
            self.best_models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            print(f"{name} - Melhor score CV: {grid_search.best_score_:.4f}")
            print(f"{name} - Melhores parâmetros: {grid_search.best_params_}")
    
    def get_best_models(self):
        """Retorna os melhores modelos treinados"""
        return self.best_models