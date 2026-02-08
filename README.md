# Food Recognition Backend

Local Python backend server for food image recognition using machine learning.

## Overview

This backend replaces the external Hugging Face API with a local server running the `BinhQuocNguyen/food-recognition-model` (101 food categories). The React Native app calls this backend for food image analysis.

## Features

- **Local inference**: No external API dependencies
- **Fast responses**: Model runs locally (GPU accelerated if available)
- **Compatible API**: Drop-in replacement for Hugging Face Inference API
- **Configurable**: Easy to switch models or adjust settings

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Optional: CUDA-compatible GPU for faster inference

## Installation

1. **Install Python dependencies**:

```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment** (optional):

```bash
cp .env.example .env
# Edit .env to customize HOST, PORT, or MODEL_NAME
```

## Running the Server

Start the server with:

```bash
python server.py
```

The server will:
- Download the food recognition model on first run (~400MB)
- Start on `http://localhost:8000` by default
- Enable hot reload for development

### Production Deployment

For production, use uvicorn directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### `POST /analyze`

Analyze food image and return classifications.

**Request**:
```json
{
  "inputs": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response**:
```json
[
  {
    "label": "pizza",
    "score": 0.95
  },
  {
    "label": "pasta",
    "score": 0.03
  }
]
```

### `GET /health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "BinhQuocNguyen/food-recognition-model"
}
```

## Configuration

### Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `MODEL_NAME`: Hugging Face model name (default: `BinhQuocNguyen/food-recognition-model`)

### Using Different Models

To use a different food recognition model:

1. Edit `.env` or set environment variable:
```bash
MODEL_NAME=nateraw/food
```

2. Restart the server

## Connecting React Native App

The React Native app should be configured to connect to your backend:

### Development (Local Network)

Find your computer's IP address and use:
```
http://192.168.1.x:8000
```

### Production

Deploy to a cloud service and use the public URL:
```
https://your-backend.com
```

## Troubleshooting

### Model download fails
- Check internet connection
- Verify model name is correct
- Some models may require authentication

### Out of memory errors
- Reduce batch size (default is 1)
- Use CPU instead of GPU
- Use a smaller model

### Port already in use
- Change PORT in `.env`
- Kill process using the port: `lsof -ti:8000 | xargs kill` (macOS/Linux)

## Model Information

**BinhQuocNguyen/food-recognition-model**:
- 101 food categories
- Based on Food-101 dataset
- High accuracy for common foods
- Model size: ~400MB

## Development

### Testing the model directly

```bash
python model_loader.py
```

### Hot reload

The server runs with `reload=True` by default, automatically restarting when code changes.

## License

MIT
