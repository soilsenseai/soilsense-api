# SoilSense Prediction API

This folder adds an HTTP API alongside the existing Streamlit app.

## Deploy on Render
1. Put `main.py`, `requirements.txt`, `render.yaml`, `soilsense_mobilenetv2.keras`, and `class_names.json` in a dedicated GitHub repository (or set Render's Root Directory to this folder if using a monorepo).
2. On Render, create a **New Web Service** connected to that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. After deployment, test `https://YOUR-SERVICE.onrender.com/health` and open `/docs`.
6. Copy the service URL into `API_BASE_URL` in `camera.html` (no trailing slash), then commit the updated file to the website repo.

Note: TensorFlow/model installation and the first model load may take longer than 10 minutes on a free host. Free services may sleep when idle. If deployment fails due to memory/build limits, use a compatible host with enough RAM/storage.
