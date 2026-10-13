"""
StreamLoop — Ajuste del Modelo de Cancelación de Clientes (Churn)
Reproducible script for training, tuning, and evaluating the baseline and tuned models.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")


def load_and_prepare_data(url=DATA_URL, path=DATA_PATH):
    """Loads dataset from local file or downloads if absent, audits, cleans, and splits deterministically."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Descargando datos desde {url}...")
        urllib.request.urlretrieve(url, path)

    df = pd.read_csv(path)
    print(f"Dimensiones del dataset cargado: {df.shape}")

    # Auditoria de duplicados e IDs
    print(f"Filas duplicadas: {df.duplicated().sum()}")
    print(f"IDs de cliente unicos: {df['customerID'].nunique()}")

    # Conversion determinista de TotalCharges (espacios en blanco a NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    print(f"Nulos en TotalCharges tras conversion: {df['TotalCharges'].isnull().sum()}")

    # Mapeo de la variable objetivo Churn
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Distribucion de Churn:\n{df['Churn'].value_counts(normalize=True).round(4)}")

    # Separar caracteristicas y objetivo (excluyendo customerID)
    customer_ids = df['customerID']
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    # Particion unica 80/20 estratificada
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.20, random_state=42, stratify=y
    )

    # Validacion de no solapamiento
    overlap = set(ids_train).intersection(set(ids_test))
    assert len(overlap) == 0, "Alerta: Existe fuga de clientes entre entrenamiento y prueba."
    print(f"Particiones: Train={X_train.shape[0]} registros, Test={X_test.shape[0]} registros (Solapamiento ID={len(overlap)})")

    return X_train, X_test, y_train, y_test


def build_pipeline(classifier=None):
    """Construye un pipeline completo con ColumnTransformer para imputacion y codificacion."""
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    if classifier is None:
        classifier = RandomForestClassifier(random_state=42)

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def evaluate_test_set(model, X_test, y_test, label="Modelo"):
    """Evalua un modelo en el conjunto de prueba y calcula metricas orientadas al negocio."""
    y_pred = model.predict(X_test)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        'Modelo': label,
        'Recall': rec,
        'Precision': prec,
        'F1': f1,
        'Accuracy': acc,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    return results


def main():
    print("=" * 70)
    print("1. CARGA Y PREPARACION DE DATOS")
    print("=" * 70)
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("\n" + "=" * 70)
    print("2. LINEA BASE: ENTRENAMIENTO Y PRIMERA EVALUACION DE TEST")
    print("=" * 70)
    baseline_pipeline = build_pipeline(RandomForestClassifier(random_state=42))
    # Ajustar unicamente con X_train, y_train
    baseline_pipeline.fit(X_train, y_train)

    # 1ª EVALUACION EN TEST
    baseline_metrics = evaluate_test_set(baseline_pipeline, X_test, y_test, label="Línea Base (Default RF)")
    print(f"Métricas Línea Base:")
    print(f"  Recall:    {baseline_metrics['Recall']:.4f}")
    print(f"  Precision: {baseline_metrics['Precision']:.4f}")
    print(f"  F1-Score:  {baseline_metrics['F1']:.4f}")
    print(f"  Accuracy:  {baseline_metrics['Accuracy']:.4f}")
    print(f"  Matriz de Confusion: TN={baseline_metrics['TN']}, FP={baseline_metrics['FP']}, FN={baseline_metrics['FN']}, TP={baseline_metrics['TP']}")

    print("\n" + "=" * 70)
    print("3. BUSQUEDA ALEATORIA (RandomizedSearchCV) EN TRAIN")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [None, 5, 8, 12, 16],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2', 0.5],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample', {0: 1, 1: 2}, {0: 1, 1: 3}]
    }

    random_search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=param_dist,
        n_iter=20,
        scoring='recall',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True
    )
    random_search.fit(X_train, y_train)

    rs_results = pd.DataFrame(random_search.cv_results_)
    top_rs = rs_results.sort_values(by='mean_test_score', ascending=False).head(5)
    print("Top 5 Candidatos RandomizedSearchCV (CV Train):")
    for idx, row in top_rs.iterrows():
        print(f"  Rank {row['rank_test_score']}: Recall = {row['mean_test_score']:.4f} +/- {row['std_test_score']:.4f} | Parametros: {row['params']}")

    print("\n" + "=" * 70)
    print("4. REFINAMIENTO CON GridSearchCV EN TRAIN")
    print("=" * 70)
    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [4, 5, 6],
        'classifier__min_samples_leaf': [2, 4],
        'classifier__max_features': ['log2', 'sqrt'],
        'classifier__class_weight': ['balanced', {0: 1, 1: 3}]
    }

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring='recall',
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    gs_results = pd.DataFrame(grid_search.cv_results_)
    top_gs = gs_results.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True]).head(5)
    print("Top 5 Candidatos GridSearchCV (CV Train):")
    for idx, row in top_gs.iterrows():
        print(f"  Rank {row['rank_test_score']}: Recall = {row['mean_test_score']:.4f} +/- {row['std_test_score']:.4f} | Parametros: {row['params']}")

    best_cv_row = gs_results.loc[grid_search.best_index_]
    fold_scores = [best_cv_row[f'split{i}_test_score'] for i in range(5)]
    print(f"\nMejor Candidato GridSearchCV:")
    print(f"  Parámetros: {grid_search.best_params_}")
    print(f"  Recall Medio CV: {best_cv_row['mean_test_score']:.4f} (Std: {best_cv_row['std_test_score']:.4f})")
    print(f"  Puntuaciones por Fold: {[round(s, 4) for s in fold_scores]}")

    print("\n" + "=" * 70)
    print("5. SEGUNDA Y ULTIMA EVALUACION DE TEST: MODELO FINAL")
    print("=" * 70)
    # Reutilizar best_estimator_ reajustado automaticamente por GridSearchCV
    final_model = grid_search.best_estimator_
    final_metrics = evaluate_test_set(final_model, X_test, y_test, label="Modelo Final Ajustado")

    print(f"Métricas Modelo Final:")
    print(f"  Recall:    {final_metrics['Recall']:.4f}")
    print(f"  Precision: {final_metrics['Precision']:.4f}")
    print(f"  F1-Score:  {final_metrics['F1']:.4f}")
    print(f"  Accuracy:  {final_metrics['Accuracy']:.4f}")
    print(f"  Matriz de Confusion: TN={final_metrics['TN']}, FP={final_metrics['FP']}, FN={final_metrics['FN']}, TP={final_metrics['TP']}")

    print("\n" + "=" * 70)
    print("6. COMPARACION FINAL (TEST SET): LINEA BASE vs MODELO AJUSTADO")
    print("=" * 70)
    comparison_df = pd.DataFrame([baseline_metrics, final_metrics])
    print(comparison_df.to_string(index=False))

    delta_recall = (final_metrics['Recall'] - baseline_metrics['Recall']) * 100
    delta_prec = (final_metrics['Precision'] - baseline_metrics['Precision']) * 100
    delta_f1 = (final_metrics['F1'] - baseline_metrics['F1']) * 100
    delta_acc = (final_metrics['Accuracy'] - baseline_metrics['Accuracy']) * 100
    diff_fn = final_metrics['FN'] - baseline_metrics['FN']
    diff_fp = final_metrics['FP'] - baseline_metrics['FP']
    diff_tp = final_metrics['TP'] - baseline_metrics['TP']

    print("\nVariaciones Observadas:")
    print(f"  Recall:        {delta_recall:+.2f} pp")
    print(f"  Precision:     {delta_prec:+.2f} pp")
    print(f"  F1-Score:      {delta_f1:+.2f} pp")
    print(f"  Accuracy:      {delta_acc:+.2f} pp")
    print(f"  Cancelaciones no detectadas (FN): {diff_fn} ({baseline_metrics['FN']} -> {final_metrics['FN']})")
    print(f"  Falsas alarmas (FP):              {diff_fp:+d} ({baseline_metrics['FP']} -> {final_metrics['FP']})")
    print(f"  Cancelaciones detectadas (TP):    {diff_tp:+d} ({baseline_metrics['TP']} -> {final_metrics['TP']})")


if __name__ == "__main__":
    main()
