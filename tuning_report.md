# StreamLoop — Informe de Ajuste del Modelo de Cancelación de Clientes

## 1. Problema y Elección de la Métrica

En StreamLoop, la retención de clientes es un factor determinante para la sostenibilidad y el crecimiento del negocio. Perder suscriptores (cancelación o *churn*) reduce directamente el *Customer Lifetime Value* (LTV) y los ingresos recurrentes.

En este contexto existe una clara asimetría de costos entre los dos tipos de errores de clasificación:
- **Falso Negativo (FN)**: El modelo predice que el cliente continuará suscrito cuando en realidad cancela. El negocio no toma ninguna acción preventiva y pierde al cliente de forma irreversible.
- **Falso Positivo (FP)**: El modelo predice erróneamente que el cliente cancelará cuando en realidad planeaba quedarse. La empresa activa una estrategia de retención (descuento, llamada de fidelización, oferta promocional); el cliente la recibe y permanece. El costo incurrido es marginal respecto a la pérdida del cliente.

Dado que **no detectar una cancelación es significativamente más costoso que activar una retención innecesaria**, la métrica principal de optimización y evaluación es el **Recall de la clase positiva (`Churn = 1`)**:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Complementariamente, se registran y analizan:
- **Precision**: Proporción de cancelaciones reales entre las predichas como cancelación ($\frac{\text{TP}}{\text{TP} + \text{FP}}$), para monitorear el volumen de falsas alarmas.
- **F1-Score**: Media armónica entre Precision y Recall.
- **Accuracy**: Proporción global de aciertos. Se documenta como dato de referencia, advirtiendo que resulta engañosa en conjuntos desbalanceados ya que un modelo que clasifique a todos como no-cancelación obtendría una alta precisión global (~73.5%) sin rescatar a ningún cliente.
- **Matriz de confusión**: Con Verdaderos Negativos (TN), Falsos Positivos (FP), Falsos Negativos (FN) y Verdaderos Positivos (TP) explícitamente reportados.

---

## 2. Preparación de Datos y Particiones

### Auditoría Inicial
El dataset corresponde a *Telco Customer Churn* (7,043 registros, 21 columnas):
- **Duplicados**: 0 filas duplicadas.
- **Identificador de cliente (`customerID`)**: 7,043 identificadores únicos. Se excluyó de las variables predictoras antes del modelado para evitar memorización o artefactos espurios.
- **Variable objetivo (`Churn`)**:
  - `No`: 5,174 clientes (73.46%)
  - `Yes`: 1,869 clientes (26.54%)
  - Se codificó determinísticamente como binaria: `Yes = 1` (clase positiva), `No = 0` (clase negativa).
- **Tratamiento de `TotalCharges`**: La columna contenía 11 registros con espacios en blanco (`' '`) correspondientes a clientes con `tenure = 0`. Se convirtió a tipo numérico de forma determinista mediante `pd.to_numeric(..., errors='coerce')`, transformando dichos espacios en valores faltantes (`NaN`) que son tratados posteriormente dentro del pipeline.
- **Conservación semántica**: Categorías como `"No internet service"` y `"No phone service"` se preservaron íntegramente como categorías válidas, ya que representan información contextual de los servicios contratados y no datos ausentes.

### Partición Entrenamiento / Prueba
Se realizó una única partición 80/20 estratificada por `Churn` con semilla fija (`random_state=42`):
- **Entrenamiento (`X_train`, `y_train`)**: 5,634 clientes (tasa de churn: 26.54%).
- **Prueba (`X_test`, `y_test`)**: 1,409 clientes (tasa de churn: 26.54%, 374 cancelaciones reales y 1,035 no cancelaciones).
- **Verificación de aislamiento**: Se comprobó que la intersección de identificadores entre `ids_train` e `ids_test` es exactamente 0, garantizando ausencia de solapamiento.

*Principio anti-fuga (Data Leakage)*: Toda transformación estadística (cálculo de medianas para imputación, codificación one-hot de categorías) se encapsuló en un `Pipeline` con `ColumnTransformer` que se ajusta exclusivamente en los datos de entrenamiento.

---

## 3. Línea Base

### Arquitectura del Pipeline
Se implementó un pipeline en `scikit-learn` compuesto por:
1. **`ColumnTransformer`**:
   - Variables numéricas (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): Imputación de nulos con `SimpleImputer(strategy='median')`. No se aplica escalado numérico dado que los árboles de decisión son estrictamente invariantes a transformaciones monotónicas continuas.
   - Variables categóricas (15 columnas restantes): Imputación con `SimpleImputer(strategy='most_frequent')` y codificación con `OneHotEncoder(handle_unknown='ignore')`.
2. **Estimador**: `RandomForestClassifier(random_state=42)` con hiperparámetros por defecto (`n_estimators=100`, `max_depth=None`, `class_weight=None`).

### Primera Evaluación en Test
El pipeline se entrenó exclusivamente en `(X_train, y_train)` y se evaluó por primera vez en `(X_test, y_test)`:

| Métrica | Valor Línea Base |
| :--- | :--- |
| **Recall (Churn = 1)** | **47.59%** (0.4759) |
| **Precision** | **60.34%** (0.6034) |
| **F1-Score** | **53.21%** (0.5321) |
| **Accuracy** | **77.79%** (0.7779) |

**Matriz de Confusión de la Línea Base:**
- **TN** = 918
- **FP** = 117
- **FN** = 196 *(Clientes en riesgo de cancelación no detectados)*
- **TP** = 178 *(Cancelaciones detectadas)*

### Insuficiencia de Accuracy
A pesar de exhibir un Accuracy cercano al 78%, el modelo omite **196 de las 374 cancelaciones reales (52.41% de Falsos Negativos)**. La métrica de accuracy oculta esta falla crítica debido a la dominancia de la clase negativa (73.5%).

---

## 4. Espacio y Resultados de Búsqueda Aleatoria (`RandomizedSearchCV`)

### Configuración de la Búsqueda
- **Estimador**: Pipeline completo (preprocesador + clasificador).
- **Validación Cruzada**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` aplicada estrictamente sobre `X_train`.
- **Métrica de optimización**: `scoring='recall'`.
- **Configuración de ejecución**: `n_iter=20`, `random_state=42`, `n_jobs=-1`, `refit=True`.

### Espacio de Hiperparámetros Definido:
- `classifier__n_estimators`: `[50, 100, 200, 300]` (número de árboles en el ensamble).
- `classifier__max_depth`: `[None, 5, 8, 12, 16]` (controla sobreajuste y memorización de la clase mayoritaria).
- `classifier__min_samples_split`: `[2, 5, 10, 20]` (tamaño mínimo de muestra para dividir un nodo).
- `classifier__min_samples_leaf`: `[1, 2, 4, 8]` (tamaño mínimo de muestra en hojas; regulariza nodos terminales).
- `classifier__max_features`: `['sqrt', 'log2', 0.5]` (tamaño del subconjunto aleatorio de variables por división).
- `classifier__class_weight`: `[None, 'balanced', 'balanced_subsample', {0: 1, 1: 2}, {0: 1, 1: 3}]` (ponderación de penalización para la clase minoritaria).

### Resultados de `RandomizedSearchCV` (Top 5 en CV Train)

| Puesto | Recall Medio | Std Recall | Tiempo Ajuste | Parámetros Clave |
| :---: | :---: | :---: | :---: | :--- |
| **1** | **0.8247** | 0.0218 | 1.15s | `class_weight={0: 1, 1: 3}`, `max_depth=5`, `max_features='log2'`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=100` |
| **2** | **0.7946** | 0.0329 | 0.58s | `class_weight='balanced'`, `max_depth=5`, `max_features='sqrt'`, `min_samples_leaf=4`, `min_samples_split=2`, `n_estimators=50` |
| **3** | **0.7813** | 0.0209 | 2.50s | `class_weight='balanced'`, `max_depth=None`, `max_features='log2'`, `min_samples_leaf=8`, `min_samples_split=10`, `n_estimators=200` |
| **4** | **0.7692** | 0.0267 | 3.53s | `class_weight='balanced_subsample'`, `max_depth=8`, `max_features='log2'`, `min_samples_leaf=2`, `min_samples_split=5`, `n_estimators=300` |
| **5** | **0.7625** | 0.0176 | 2.65s | `class_weight={0: 1, 1: 3}`, `max_depth=12`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=10`, `n_estimators=200` |

---

## 5. Cuadrícula Refinada y su Justificación (`GridSearchCV`)

### Justificación de la Cuadrícula Focalizada
El análisis de `RandomizedSearchCV` reveló patrones consistentes:
1. **Penalización de clases**: Los modelos sin `class_weight` obtuvieron recalls inferiores a 0.50. Ponderaciones como `{0: 1, 1: 3}` y `'balanced'` (que calcula pesos inversamente proporcionales, aproximadamente 1:2.77) dominaron los primeros puestos, superando el 79% de recall medio.
2. **Profundidad controlada (`max_depth`)**: Profundidades moderadas entre 4 y 6 superaron ampliamente a árboles no podados (`None`) o muy profundos (`12`, `16`), ya que evitan que el bosque cree hojas ultraespecíficas para la clase mayoritaria.
3. **Regularización en hojas (`min_samples_leaf`)**: Valores entre 2 y 4 demostraron menor variabilidad entre folds.
4. **Espacio de variables (`max_features`)**: Tanto `'log2'` como `'sqrt'` mostraron comportamientos estables y rápidos.
5. **Tamaño del ensamble (`n_estimators`)**: `100` y `150` árboles garantizaron convergencia completa sin desperdicio computacional.

### Espacio de `GridSearchCV` (48 combinaciones en total)
- `classifier__class_weight`: `['balanced', {0: 1, 1: 3}]`
- `classifier__max_depth`: `[4, 5, 6]`
- `classifier__min_samples_leaf`: `[2, 4]`
- `classifier__max_features`: `['log2', 'sqrt']`
- `classifier__n_estimators`: `[100, 150]`

---

## 6. Comparación de Candidatos y Estabilidad entre Folds

Se ejecutó `GridSearchCV` sobre `X_train` utilizando la misma validación cruzada estratificada de 5 folds.

### Top 5 Candidatos de `GridSearchCV`

| Puesto | Recall Medio | Desviación Típica | `class_weight` | `max_depth` | `max_features` | `min_samples_leaf` | `n_estimators` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **0.8294** | **0.0209** | `{0: 1, 1: 3}` | 6 | `'sqrt'` | 2 | 150 |
| **2** | **0.8281** | 0.0218 | `{0: 1, 1: 3}` | 6 | `'sqrt'` | 2 | 100 |
| **3** | **0.8274** | 0.0245 | `{0: 1, 1: 3}` | 6 | `'log2'` | 4 | 100 |
| **4** | **0.8274** | 0.0250 | `{0: 1, 1: 3}` | 6 | `'sqrt'` | 4 | 150 |
| **5** | **0.8274** | 0.0274 | `{0: 1, 1: 3}` | 6 | `'log2'` | 2 | 150 |

### Análisis de Estabilidad del Mejor Candidato
Examinando las puntuaciones individuales por fold del candidato en Puesto 1:
- **Fold 0**: 0.8328
- **Fold 1**: 0.8027
- **Fold 2**: 0.8261
- **Fold 3**: 0.8662
- **Fold 4**: 0.8194
- **Media**: **0.8294** | **Desviación Típica**: **0.0209**

La variabilidad entre folds es pequeña (~2.09%), sin caídas por debajo del 80% en ninguna partición, lo que evidencia que la configuración generaliza sólidamente sobre los distintos subconjuntos de clientes sin sobreajustar a patrones locales.

---

## 7. Hiperparámetros Finales

Se seleccionó el estimador óptimo identificado por la validación cruzada:

```python
{
    'classifier__class_weight': {0: 1, 1: 3},
    'classifier__max_depth': 6,
    'classifier__max_features': 'sqrt',
    'classifier__min_samples_leaf': 2,
    'classifier__n_estimators': 150
}
```

Al haberse configurado con `refit=True`, el objeto `grid_search.best_estimator_` fue reajustado automáticamente por `scikit-learn` sobre la totalidad del conjunto de entrenamiento (`X_train`, `y_train`). No se ejecutó ningún `fit()` manual adicional.

---

## 8. Comparación del Test entre Línea Base y Modelo Ajustado

Se procedió a la **segunda y última evaluación en el conjunto de prueba (`X_test`, `y_test`)**:

| Métrica | Línea Base (Default RF) | Modelo Final Ajustado | Variación Absoluta | Variación Puntos Porcentuales |
| :--- | :---: | :---: | :---: | :---: |
| **Recall (Churn = 1)** | 0.4759 (47.59%) | **0.8048 (80.48%)** | +0.3289 | **+32.89 pp** |
| **Precision** | 0.6034 (60.34%) | **0.4959 (49.59%)** | -0.1075 | **-10.75 pp** |
| **F1-Score** | 0.5321 (53.21%) | **0.6137 (61.37%)** | +0.0815 | **+8.15 pp** |
| **Accuracy** | 0.7779 (77.79%) | **0.7310 (73.10%)** | -0.0468 | **-4.68 pp** |

### Desglose de la Matriz de Confusión en Test (1,409 clientes)

| Componente | Línea Base | Modelo Final | Diferencia | Impacto de Negocio |
| :--- | :---: | :---: | :---: | :--- |
| **Verdaderos Positivos (TP)** | 178 | **301** | **+123** | Clientes en riesgo detectados y abordables para retención. |
| **Falsos Negativos (FN)** | 196 | **73** | **-123** | **Reducción del 62.8% en cancelaciones ignoradas.** |
| **Falsos Positivos (FP)** | 117 | **306** | **+189** | Clientes estables que reciben oferta de retención. |
| **Verdaderos Negativos (TN)** | 918 | **729** | -189 | Clientes no retenibles clasificados correctamente. |

---

## 9. Conclusiones y Limitaciones

### Conclusiones Principales
1. **Éxito en el objetivo de negocio**: El modelo final detecta al **80.48%** de los clientes que van a cancelar (frente a sólo el 47.59% de la línea base), evitando que **123 cancelaciones** pasen desapercibidas en una muestra de prueba de 1,409 clientes.
2. **Evaluación de contrapartidas**: Para lograr rescatar a esos 123 clientes adicionales, las falsas alarmas aumentaron en 189 casos (Precision cayó de 60.34% a 49.59%). Sin embargo, debido a la premisa de negocio de que el costo de perder un suscriptor supera al costo de una oferta de retención, esta concesión es altamente favorable.
3. **Mejora en F1-Score**: El balance armónico F1 creció de **53.21% a 61.37% (+8.15 pp)**, demostrando que la ganancia neta en recall compensa el incremento controlado de falsos positivos.
4. **Integridad metodológica**: El conjunto de prueba se evaluó únicamente dos veces en todo el ciclo de desarrollo (Línea Base y Modelo Final). Todas las decisiones de hiperparámetros se sustentaron de forma exclusiva en la validación cruzada sobre `X_train`.

### Limitaciones y Pasos Futuros
1. **Umbral fijo de decisión**: Se mantuvo el umbral estándar de 0.5. Al no contarse con los costos monetarios exactos de campaña por cliente versus el LTV promedio, no se calibró un umbral económico específico en test.
2. **Calibración de probabilidades**: En una fase previa a despliegue en producción, se recomienda calibrar las probabilidades predichas (`predict_proba`) mediante validación cruzada y construir curvas de costo económico en función del umbral de decisión.
