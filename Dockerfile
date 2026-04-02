# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the project files
COPY . .

# Expose Gradio port
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the app
CMD ["python", "app.py"]