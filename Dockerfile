FROM python:3.12-slim

# ---------------------------------------------------------
# Install only the minimal system libs Matplotlib + NumPy need
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Set up app directory
# ---------------------------------------------------------
WORKDIR /app
COPY . /app

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Matplotlib: force headless (server-safe) backend
# ---------------------------------------------------------
ENV MPLBACKEND=Agg

# ---------------------------------------------------------
# Cloud Run requires port 8080
# ---------------------------------------------------------
EXPOSE 8080

# ---------------------------------------------------------
# Start Streamlit with correct Cloud Run flags
# ---------------------------------------------------------
CMD ["streamlit", "run", "main_streamlit_v3.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]