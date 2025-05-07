import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt





def snake(col):
        col = re.sub(r'[\s\-]', '_', col) 
        col = re.sub(r'([a-z])([A-Z])', r'\1_\2', col)  
        col = re.sub(r'_{2,}', '_', col)  
        return col.lower().strip('_')  



def k_optimo(X_train, y_train, X_test, y_test, k_min=1, k_max=50, k_step=2, metric='f1'):
    """
    Encuentra el mejor valor de K para KNN usando validación cruzada y una métrica especificada.

    Parámetros:
    - X_train, y_train: Datos de entrenamiento.
    - X_test, y_test: Datos de prueba.
    - k_min: Valor mínimo de K a probar (por defecto 1).
    - k_max: Valor máximo de K a probar (por defecto 50).
    - k_step: Paso entre valores de K (por defecto 2).
    - metric: Métrica a optimizar ('accuracy', 'f1', 'roc_auc').

    Retorna:
    - Mejor valor de K.
    - Mejor puntuación de la métrica seleccionada.
    """

    k_values = list(range(k_min, k_max+1, k_step))
    scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_score = cross_val_score(knn, X_train, y_train, cv=5, scoring=metric).mean()
        scores.append(cv_score)

    mejor_k = k_values[np.argmax(scores)]
    mejor_puntuacion = max(scores)

    # Graficar los resultados
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, scores, marker='o', linestyle='-', label=f'{metric} en validación cruzada')
    plt.axvline(mejor_k, linestyle='--', color='r', label=f'Mejor K = {mejor_k}')
    plt.xlabel('Valores K')
    plt.ylabel(f'Puntuación {metric}')
    plt.title(f'Búsqueda del Mejor K ({metric})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mejor_k, mejor_puntuacion




