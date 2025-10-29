# 🧠 Proyecto MLOps – Fase 2 Equipo 42
### Equipo 42 – Predicción de Adquisición de Pólizas de Caravana (CoIL Challenge 2000)

Este proyecto implementa un flujo de Machine Learning bajo prácticas **MLOps** con:
- Código refactorizado en **/src** (enfoque **OOP**),
- **Pipeline** reproducible (Scikit-Learn Pipeline + ColumnTransformer),
- **MLflow** para seguimiento de experimentos,
- **DVC** para versionar pipeline/datos/modelos,
- **Tests** mínimos para validar preprocesamiento.

## ⚙️ Requisitos
- Python ≥ 3.10, Git ≥ 2.40, DVC ≥ 3.0, MLflow ≥ 2.0

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📂 Estructura principal

- `src/` — Código fuente modular (data, features, models, pipelines, utils)
- `data/` — Datos versionados (DVC)
- `models/` — Modelos entrenados y exportados
- `notebooks/` — Análisis exploratorio y prototipos
- `test/` — Pruebas unitarias
- `reports/` — Resultados, figuras y métricas
- `docs/` — Documentación extendida

## 🚀 Ejecución del pipeline

```bash
python src/pipelines/train_pipeline.py
```

## 🧪 Tests

```bash
pytest test/
```

## 📊 Seguimiento de experimentos

- Iniciar MLflow UI:
  ```bash
  mlflow ui
  ```
- Versionado de datos/modelos con DVC:
  ```bash
  dvc repro
  ```
- Para versionar datos y modelos, asegúrate de tener inicializado DVC en el proyecto:

  ```powershell
  dvc init
  dvc repro
  ```

- El pipeline está definido en `dvc.yaml` y versiona los outputs clave.
- Los archivos y carpetas ignorados por DVC están en `.dvcignore`.

- Para seguimiento de experimentos ejecuta:

  ```powershell
  mlflow ui
  ```

## 📖 Documentación

Ver detalles en `docs/` y comentarios en los scripts de `src/`.
