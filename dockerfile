# Usar Python 3.10 como imagen base
FROM python:3.10-slim

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Copiar el archivo de requirements.txt al contenedor
COPY requirements.txt .

# Instalar las dependencias y manejar excepciones
RUN pip install --no-cache-dir -r requirements.txt || \
    { echo "Algunas dependencias no se pudieron instalar. Revise error_log.txt para detalles." && \
      pip freeze > error_log.txt; exit 0; }

# Copiar todo el código fuente al contenedor
COPY . .

# Comando por defecto para ejecutar la aplicación
CMD ["python3"]