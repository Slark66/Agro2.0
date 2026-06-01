# AgriSense AI Crop Recommendation System

A professional website for AI-based crop recommendation. It includes:

- Farmer registration and login
- Secure password hashing
- Crop recommendation from soil and climate inputs
- Saved recommendation history in SQLite
- One-click reuse of previous field values
- Mineral and soil reference tables
- Printable and downloadable PDF recommendation reports
- Kaggle-format dataset support
- IoT device simulation with saved sensor readings

## Run The Project

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Kaggle Dataset

Download the common Kaggle **Crop Recommendation Dataset** and place the CSV here:

```text
data/Crop_recommendation.csv
```

The CSV should contain these columns:

```text
N,P,K,temperature,humidity,ph,rainfall,label
```

If that file is not present, the app uses `data/sample_crop_recommendation.csv` so the demo still works.

## Google Colab Model Training

For model training proof, use:

```text
colab/train_model_colab.py
```

Paste that code into Google Colab, upload `Crop_recommendation.csv`, and run it. It trains a Random Forest model, prints accuracy, displays a confusion matrix, shows feature importance, and saves `crop_recommendation_model.pkl`.

## Best Place To Build And Present This

For development, use **VS Code** on your laptop because it is easiest for editing Flask, HTML, CSS, and the database together.

For a reliable live demo, run it locally from your laptop using:

```powershell
python app.py
```

This avoids internet or hosting problems during demos. If you want an online deployment later, use **Render** or **PythonAnywhere** for Flask hosting. Use **GitHub** to store your code and project report safely.

## Deploy Online

### Option 1: Render

1. Push this folder to GitHub.
2. Create a new Render Web Service.
3. Connect your GitHub repository.
4. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

5. Add an environment variable:

```text
SECRET_KEY=choose-a-long-random-secret
```

Render is good for an online link, but the free instance may sleep when inactive. SQLite is fine for demo use, but a production project should use PostgreSQL.

### Option 2: PythonAnywhere

1. Upload the project files or clone from GitHub.
2. Create a virtual environment.
3. Install requirements with `pip install -r requirements.txt`.
4. Create a Flask web app from the PythonAnywhere Web tab.
5. Point the WSGI file to `app.py` and expose the Flask variable named `app`.
6. Reload the web app.

PythonAnywhere is simple for college demos and Flask projects.

## Project Flow

1. User registers or logs in.
2. User enters N, P, K, temperature, humidity, pH, and rainfall.
3. The backend loads the Kaggle-format crop dataset.
4. The AI recommender compares the input with known crop patterns.
5. The best crop, confidence, advisory, and alternatives are shown.
6. The recommendation is stored in SQLite and appears in history.
7. The farmer can reopen the report, print it, download it as PDF, or reuse the same field values.

## Suggested IoT Extension

Use an ESP32 with NPK, pH, soil moisture, DHT22 temperature/humidity, and optional rain sensor modules. The current website includes a simulator at `/iot` that stores readings through `/api/iot/simulate`; those readings can fill the analysis form automatically.
