# 🚢 Titanic Survival Prediction - Kaggle Benchmark

Proyecto de clasificación supervisada para predecir la supervivencia de pasajeros del Titanic, desarrollado como parte de una competencia de Kaggle. El flujo incluye análisis exploratorio, preprocesamiento, entrenamiento y evaluación de modelos, además de comparación mediante métricas clave y ajustes de hiperparámetros.

---

## Objetivos

- Participar en una competencia real de Kaggle (Titanic Classification).
- Realizar análisis exploratorio completo (EDA) y limpieza del dataset.
- Implementar al menos **5 modelos de clasificación supervisada**.
- Evaluar rendimiento con métricas relevantes: Accuracy, ROC-AUC, F1-Score.
- Optimizar hiperparámetros con `GridSearchCV` / `RandomizedSearchCV`.
- Comparar resultados y reflexionar sobre su comportamiento.

---

## Dataset

- **Fuente:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Descripción:** Datos de pasajeros con variables como clase, edad, sexo, tarifa, embarque y número de familiares.  
- **Variable objetivo:** `Survived` (binaria: 1 = sobrevivió, 0 = no sobrevivió)

---

## Herramientas y Técnicas

- **Lenguaje:** Python  
- **Librerías:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost, LightGBM  
- **Técnicas aplicadas:**
  - Limpieza de datos y tratamiento de valores faltantes
  - Codificación de variables categóricas y escalado
  - Validación cruzada (`KFold`)
  - Tuning de hiperparámetros
  - Benchmark interpretativo mediante visualizaciones

---

## Conclusiones y patrones de supervivencia

- **Sexo como variable altamente predictiva:** Las mujeres tuvieron una tasa de supervivencia considerablemente más alta que los hombres. Este patrón fue capturado por todos los modelos, especialmente los basados en árboles.

- **Clase social impacta directamente la probabilidad de sobrevivir:** Los pasajeros de primera clase (`Pclass=1`) tuvieron mayores tasas de supervivencia que los de segunda y tercera clase. Esto refleja no solo diferencias económicas, sino acceso prioritario al rescate.

- **Edad y relaciones familiares influyen en la supervivencia:** Los niños pequeños (edad <10) y pasajeros que viajaban con familiares (variables `SibSp` y `Parch`) presentaron mayor probabilidad de sobrevivir, lo que sugiere atención en grupo durante la evacuación.

- **Valores faltantes y variables como 'Cabin':** La variable `Cabin` tiene muchos valores nulos, pero su presencia suele estar asociada a pasajeros de primera clase, lo que indirectamente puede correlacionarse con la supervivencia. Fue descartada en algunos modelos por falta de completitud.

- **Ubicación de embarque (`Embarked`) como variable secundaria:** Aunque menos determinante, se observaron diferencias leves en las tasas de supervivencia según el puerto de embarque (`C`, `Q`, `S`), posiblemente ligadas a la distribución de clases y nacionalidades.

---

## Modelos Evaluados

Modelos utilizados:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- XGBoost
- LightGBM

## Rendimiento comparativo de modelos

| Modelo                  | Accuracy  | AUC-ROC  | F1-Score  | Std. Dev  |
|------------------------|-------------|------------|-------------|--------------|
| **LightGBM**           | **0.8436**   | 0.8897     | **0.8056**   | 0.0120       |
| XGBoost                | 0.8212      | 0.8820     | 0.7647      | 0.0141       |
| K-Nearest Neighbors    | 0.8101      | **0.8933** | 0.7606      | 0.0210       |
| Decision Tree          | 0.7989      | 0.8463     | 0.7391      | 0.0150       |
| Logistic Regression    | 0.7933      | 0.8777     | 0.7299      | 0.0215       |

> 🏆 **LightGBM** lidera en accuracy y F1-score, con configuración óptima:
```python
{
  'model__num_leaves': 31,
  'model__n_estimators': 100,
  'model__learning_rate': 0.1
}




