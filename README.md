# StreamLoop — Ajuste del Modelo de Cancelación de Clientes (Churn)

## Resumen del Proyecto

En este proyecto optimizamos un modelo de clasificación para predecir la cancelación de clientes (*churn*) priorizando el **Recall de la clase positiva**, debido a que no detectar una cancelación es significativamente más costoso para el negocio que emitir una oferta de retención.

- **Pipeline y Línea Base**: Preprocesamiento encapsulado (`SimpleImputer` + `OneHotEncoder`) con `RandomForestClassifier` por defecto.
- **Búsqueda y Ajuste**: Búsqueda amplia con `RandomizedSearchCV` seguida de refinamiento con `GridSearchCV` sobre 5 folds estratificados en entrenamiento (`scoring='recall'`).
- **Resultados en Test**:
  - **Recall**: subió de **47.59% a 80.48%** (+32.89 pp).
  - **Cancelaciones no detectadas (FN)**: reducidas de **196 a 73** (62.8% de pérdidas evitadas).
  - **F1-Score**: mejoró de **53.21% a 61.37%** (+8.15 pp).

Para detalles completos consulta [tuning_report.md](tuning_report.md) y el notebook [src/explore.ipynb](src/explore.ipynb). Para reproducir: `pip install -r requirements.txt && python main.py`.

---

# Python Hello

The most basic boilerplate to start a Python project at 4Geeks is to start your very first Python project from scratch.

## What to do next?

Open the `main.py` file and start writing your code.

Execute your code by typing the following command on your terminal:

```bash
$ python main.py
```

You can create and include as many python files (a.k.a. modules) as you want using the import statements.

## Requirements

Make sure you have Python installed in your computer. We strongly recommend [installing Python through Pyenv ](https://4geeks.com/how-to/what-is-pyenv-and-how-to-install-pyenv) to avoid version conflicts in the future.

### Contributors

This template was built as part of the [4Geeks Python Resources](https://4geeks.com/technology/python) for learning at [4Geeks.com](https://4geeks.com) by [Alejandro Sanchez](https://twitter.com/alesanchezr) and [many other contributors](https://github.com/4GeeksAcademy/python-hello/graphs/contributors).
