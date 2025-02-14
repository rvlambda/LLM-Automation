FROM python:3.12-slim-bookworm

# Install Node.js, npm, tesseract, and other dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get update && apt-get install -y \
    nodejs \
    npm \
    tesseract-ocr \
    libtesseract-dev \
    curl \
    ca-certificates

# Install Prettier globally
RUN npm install -g prettier@3.4.2

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app
# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code to the container
COPY . /app/


ENV PATH="/usr/local/lib/python3.12/site-packages/:$PATH"
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV AIPROXY_TOKEN=""

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000","--log-level", "debug"]
