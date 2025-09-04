# 🖥️ Training & Prediction Server

This server exposes a REST API (via **FastAPI**) for:
- Returning available class labels from the model checkpoint.
- Predicting the class of an uploaded image.
- Saving annotated images + labels into the training dataset.

---

## ⚙️ Setup

1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt

2. Run the server with uvicorn:
     uvicorn server.app:app --reload --port 8000
    
The server will be available at:
👉 http://localhost:8000

