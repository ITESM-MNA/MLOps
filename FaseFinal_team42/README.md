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

---

## 🛠️ Resumen de la Reestructuración y Mejores Prácticas Implementadas

### 1) Estructuración de Proyectos con Cookiecutter

- Se utilizó la plantilla oficial de Cookiecutter para proyectos de ciencia de datos, implementando la estructura recomendada de carpetas y archivos.
- El proyecto ahora cuenta con directorios bien definidos: `src/` (código fuente), `data/` (datos en distintas etapas), `models/` (modelos entrenados), `notebooks/` (análisis exploratorio), `reports/` (resultados y visualizaciones), `docs/` (documentación), `test/` (pruebas), y archivos de configuración como `requirements.txt`, `dvc.yaml`, y `README.md`.
- Esta organización facilita la colaboración, el mantenimiento y la escalabilidad del proyecto.

### 2) Estructuración y Refactorización del Código

- El código fue reorganizado en módulos y funciones con responsabilidades claras dentro de la carpeta `src/`.
- Se aplicaron principios de programación orientada a objetos (POO) donde fue pertinente, mejorando la reutilización y extensibilidad.
- Se refactorizó el código para mejorar su eficiencia, legibilidad y mantenibilidad, asegurando que cada módulo tenga una función específica y bien documentada.

### 3) Aplicación de Mejores Prácticas en el Pipeline de Modelado

- Se implementó un pipeline de Scikit-Learn utilizando `Pipeline` y `ColumnTransformer` para automatizar el preprocesamiento, entrenamiento y evaluación de modelos.
- Cada etapa del pipeline está documentada y modularizada, permitiendo la reproducción y comprensión por parte de cualquier colaborador.
- El pipeline es ejecutable desde scripts en `src/` y está versionado con DVC para asegurar la trazabilidad de los datos y modelos.

### 4) Seguimiento de Experimentos, Visualización de Resultados y Gestión de Modelos

- Se integró MLflow para el registro y comparación de experimentos, permitiendo visualizar métricas, parámetros y resultados de manera clara.
- DVC se utiliza para el versionado de datos y modelos, asegurando la reproducibilidad y el control de versiones.
- Se documentan las configuraciones, hiperparámetros y métricas relevantes de cada modelo entrenado.
- Los resultados y visualizaciones generados se almacenan en la carpeta `reports/`, facilitando el análisis comparativo y la toma de decisiones informadas.

---

Este enfoque garantiza que el proyecto sea reproducible de principio a fin, permitiendo a cualquier persona ejecutar el código, replicar los experimentos y obtener resultados equivalentes a los presentados.
