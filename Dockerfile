# Usar una imagen oficial de Python con una versión estable (Python 3.11)
FROM python:3.11-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias si alguna librería lo requiere
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto al contenedor
COPY . .

# Como es un script interactivo de consola, usamos python -u (unbuffered) 
# para que los prints se muestren en tiempo real en la terminal
CMD ["python", "-u", "main.py"]
