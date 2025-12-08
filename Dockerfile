# Use the official Manim image (comes with Python, FFmpeg, LaTeX pre-installed)
# v0.19.0 includes Python 3.9+ with support for modern type hints
FROM manimcommunity/manim:v0.19.0

# Switch to root to install dependencies and configure permissions
USER root

# Set the working directory
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create necessary directories for video storage
# "777" permissions ensure the app can write to these folders even if the user changes
RUN mkdir -p videos temp && chmod -R 777 videos temp

# Expose the port FastAPI will run on
# Cloud Run uses PORT env variable (defaults to 8080)
EXPOSE 8080

# Start the application
# We use 0.0.0.0 to allow external access
# Cloud Run sets PORT env variable, defaulting to 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
