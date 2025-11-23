# Ask Doubt Backend

Backend service for generating Manim animation solutions from images and questions using Gemini AI.

## Features

- Image + text question upload
- Gemini 3 API integration for code generation
- Automatic Manim rendering
- Video solution delivery
- FastAPI REST API

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg** - Required for video rendering
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
3. **Manim** - Installed via pip (automatic)

## Quick Start

### 1. Setup

```bash
cd ask-doubt-backend

# Option A: Use the startup script (recommended)
./start.sh

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Make sure `.env` file exists with:
```
GEMINI_API_KEY=your_api_key_here
PORT=8000
```

### 3. Run

```bash
# If using startup script
./start.sh

# If manual
uvicorn app.main:app --reload --port 8000
```

### 4. Test

Open browser to:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## API Endpoints

### POST /api/ask-doubt

Submit a doubt with image and question.

**Request:**
- `image`: Image file (multipart/form-data)
- `question`: Question text (form field)

**Response:**
```json
{
  "status": "success",
  "video_url": "/videos/solution_abc123_20251121.mp4",
  "filename": "solution_abc123_20251121.mp4",
  "duration": 45.2,
  "generated_at": "2025-11-21T16:30:00Z",
  "message": "Solution generated successfully"
}
```

### GET /api/health

Health check endpoint.

## Project Structure

```
ask-doubt-backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── api/
│   │   └── routes.py        # API routes
│   └── services/
│       ├── gemini_client.py # Gemini AI integration
│       └── manim_renderer.py # Manim rendering
├── videos/                  # Generated videos
├── temp/                    # Temporary files
├── requirements.txt
├── .env
└── start.sh
```

## Development

```bash
# Activate venv
source venv/bin/activate

# Run with auto-reload
uvicorn app.main:app --reload

# Run tests (if added)
pytest
```

## Troubleshooting

### Manim not found
```bash
pip install manim
```

### FFmpeg not found
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### Gemini API errors
- Check API key in `.env`
- Verify quota/limits on Google AI Studio

### Port already in use
```bash
# Change port in .env
PORT=8001
```

## License

MIT
