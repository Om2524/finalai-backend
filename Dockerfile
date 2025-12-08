# Use the official Manim image (comes with Python, FFmpeg, LaTeX pre-installed)
# v0.19.0 includes Python 3.9+ with support for modern type hints
FROM manimcommunity/manim:v0.19.0

# ============================================================================
# CLOUD RUN DEPLOYMENT NOTE:
# When deploying to Google Cloud Run, you MUST set the request timeout to 600s
# to allow complex physics problems to render (MANIM_TIMEOUT=300s + buffer).
#
# Deploy command:
#   gcloud run deploy ask-doubt-backend \
#     --image gcr.io/PROJECT_ID/ask-doubt-backend \
#     --timeout=600s \
#     --memory=2Gi \
#     --cpu=2 \
#     --platform managed \
#     --region us-central1
#
# Or via Cloud Console: Service > Edit & Deploy > Container > Request timeout: 600
# ============================================================================

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
