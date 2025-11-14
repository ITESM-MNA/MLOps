# 🧪 Guía para Pruebas Unitarias y de Integración

Este documento describe cómo implementar y ejecutar pruebas automatizadas para validar los componentes críticos del proyecto.

## 📋 Implementación de Pruebas

### 1. Pruebas Unitarias
- **Objetivo:** Validar funciones o módulos individuales.
- **Ejemplo:** Validar el preprocesamiento, cálculo de métricas o inferencia.
- **Ubicación:** Guardar en `test/` con el prefijo `test_` (ej.: `test_preprocessing.py`).

### 2. Pruebas de Integración
- **Objetivo:** Validar el flujo extremo a extremo del pipeline.
- **Ejemplo:** Carga de datos → Preprocesamiento → Predicción → Cálculo de métricas.
- **Ubicación:** Guardar en `test/` con el prefijo `test_integration_` (ej.: `test_integration_pipeline.py`).

## 🚀 Ejecución de Pruebas

Ejecutar todas las pruebas:
```bash
pytest -q
```

Ejecutar pruebas específicas:
```bash
pytest test/test_nombre_archivo.py
```

## 📂 Estructura Recomendada

- `test/test_load_data.py`: Prueba unitaria para la función `load_data`.
- `test/test_preprocessing.py`: Prueba unitaria para el preprocesamiento.
- `test/test_metrics.py`: Prueba unitaria para el cálculo de métricas.
- `test/test_integration_pipeline.py`: Prueba de integración para el pipeline completo.

## 🛠️ Ejemplo de Prueba de Integración

```python
import pytest
from src.pipelines.train_pipeline import main_pipeline

def test_integration_pipeline():
    """Prueba de integración para el pipeline completo."""
    result = main_pipeline()
    assert result is not None
    assert "accuracy" in result
    assert result["accuracy"] > 0.7
```

## 🔍 Notas
- Asegúrate de que todas las dependencias estén instaladas:
  ```bash
  pip install -r requirements.txt
  ```
- Documenta las pruebas con comentarios claros para facilitar su mantenimiento.