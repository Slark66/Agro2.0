from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import random
from collections import Counter, defaultdict
from datetime import datetime
from functools import wraps
from io import BytesIO
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE = BASE_DIR / "crop_recommendation.db"
KAGGLE_DATASET = DATA_DIR / "Crop_recommendation.csv"
SAMPLE_DATASET = DATA_DIR / "sample_crop_recommendation.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

MINERAL_REFERENCES = [
    {
        "key": "N",
        "name": "Nitrogen",
        "unit": "kg/ha",
        "low": "0-49",
        "ideal": "50-120",
        "high": "121+",
        "role": "Leaf growth, chlorophyll, and early crop vigor.",
    },
    {
        "key": "P",
        "name": "Phosphorus",
        "unit": "kg/ha",
        "low": "0-24",
        "ideal": "25-75",
        "high": "76+",
        "role": "Root development, flowering, and seed formation.",
    },
    {
        "key": "K",
        "name": "Potassium",
        "unit": "kg/ha",
        "low": "0-34",
        "ideal": "35-120",
        "high": "121+",
        "role": "Water regulation, disease tolerance, and fruit quality.",
    },
    {
        "key": "ph",
        "name": "Soil pH",
        "unit": "pH",
        "low": "0-5.4",
        "ideal": "5.5-7.5",
        "high": "7.6-14",
        "role": "Controls nutrient availability and microbial activity.",
    },
]

REFERENCE_BOUNDS = {
    "N": (50, 120),
    "P": (25, 75),
    "K": (35, 120),
    "ph": (5.5, 7.5),
}

CROP_DETAILS = {
    "rice": {
        "season": "Kharif / monsoon",
        "water": "High",
        "notes": "Best suited to warm, humid fields with reliable water availability.",
    },
    "maize": {
        "season": "Kharif or Rabi",
        "water": "Moderate",
        "notes": "Performs well in drained soil and benefits from balanced nitrogen.",
    },
    "chickpea": {
        "season": "Rabi",
        "water": "Low to moderate",
        "notes": "A good pulse crop for slightly dry conditions and moderate fertility.",
    },
    "kidneybeans": {
        "season": "Rabi / cool season",
        "water": "Moderate",
        "notes": "Needs cooler temperatures and avoids waterlogged fields.",
    },
    "pigeonpeas": {
        "season": "Kharif",
        "water": "Low to moderate",
        "notes": "Drought-tolerant pulse crop suitable for warm climates.",
    },
    "mothbeans": {
        "season": "Kharif",
        "water": "Low",
        "notes": "Hardy crop for arid and semi-arid regions.",
    },
    "mungbean": {
        "season": "Kharif / summer",
        "water": "Moderate",
        "notes": "Short-duration pulse crop that prefers warm weather.",
    },
    "blackgram": {
        "season": "Kharif / Rabi",
        "water": "Moderate",
        "notes": "Suitable for warm conditions with well-drained soil.",
    },
    "lentil": {
        "season": "Rabi",
        "water": "Low to moderate",
        "notes": "Cool-season pulse crop for neutral to slightly alkaline soils.",
    },
    "pomegranate": {
        "season": "Perennial",
        "water": "Low to moderate",
        "notes": "Thrives in semi-arid regions with careful irrigation scheduling.",
    },
    "banana": {
        "season": "Perennial",
        "water": "High",
        "notes": "Requires high humidity, warm temperature, and strong potassium supply.",
    },
    "mango": {
        "season": "Perennial",
        "water": "Moderate",
        "notes": "Prefers warm climates, good drainage, and seasonal dry spells.",
    },
    "grapes": {
        "season": "Perennial",
        "water": "Moderate",
        "notes": "Needs managed irrigation and well-drained soil.",
    },
    "watermelon": {
        "season": "Summer",
        "water": "Moderate to high",
        "notes": "Warm-season crop that benefits from sandy loam and consistent moisture.",
    },
    "muskmelon": {
        "season": "Summer",
        "water": "Moderate",
        "notes": "Prefers warm, sunny conditions and well-drained soil.",
    },
    "apple": {
        "season": "Temperate perennial",
        "water": "Moderate",
        "notes": "Best for cooler hill climates, not hot plains.",
    },
    "orange": {
        "season": "Perennial",
        "water": "Moderate",
        "notes": "Requires good drainage and balanced soil pH.",
    },
    "papaya": {
        "season": "Perennial",
        "water": "Moderate to high",
        "notes": "Grows well in warm and humid climates with fertile soil.",
    },
    "coconut": {
        "season": "Perennial",
        "water": "High",
        "notes": "Best in coastal humid climates with consistent rainfall.",
    },
    "cotton": {
        "season": "Kharif",
        "water": "Moderate",
        "notes": "Requires warm weather, good sunlight, and careful pest monitoring.",
    },
    "jute": {
        "season": "Kharif",
        "water": "High",
        "notes": "Needs warm, humid conditions and alluvial soil.",
    },
    "coffee": {
        "season": "Perennial",
        "water": "Moderate to high",
        "notes": "Best under shade in humid hilly regions.",
    },
}


class CropRecommender:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.rows = self._load_rows(dataset_path)
        self.stats = self._feature_stats()
        self.label_counts = Counter(row["label"] for row in self.rows)

    @staticmethod
    def _load_rows(dataset_path: Path) -> list[dict]:
        with dataset_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = []
            for row in reader:
                rows.append(
                    {
                        **{feature: float(row[feature]) for feature in FEATURES},
                        "label": row["label"].strip().lower(),
                    }
                )
        if not rows:
            raise RuntimeError("Dataset is empty. Add valid crop records to the CSV file.")
        return rows

    def _feature_stats(self) -> dict[str, tuple[float, float]]:
        values_by_feature = defaultdict(list)
        for row in self.rows:
            for feature in FEATURES:
                values_by_feature[feature].append(row[feature])

        stats = {}
        for feature, values in values_by_feature.items():
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            stats[feature] = (mean, math.sqrt(variance) or 1)
        return stats

    def recommend(self, inputs: dict[str, float], k: int = 7) -> dict:
        distances = []
        for row in self.rows:
            distance = math.sqrt(
                sum(
                    ((inputs[feature] - row[feature]) / self.stats[feature][1]) ** 2
                    for feature in FEATURES
                )
            )
            distances.append((distance, row["label"]))

        nearest = sorted(distances, key=lambda item: item[0])[:k]
        weighted_scores = defaultdict(float)
        for distance, label in nearest:
            weighted_scores[label] += 1 / (distance + 0.001)

        ranked = sorted(weighted_scores.items(), key=lambda item: item[1], reverse=True)
        total_score = sum(score for _, score in ranked) or 1
        top_label, top_score = ranked[0]

        alternatives = [
            {"crop": label.title(), "score": round((score / total_score) * 100, 1)}
            for label, score in ranked[1:4]
        ]

        details = CROP_DETAILS.get(
            top_label,
            {
                "season": "Check local agriculture advisory",
                "water": "Depends on local conditions",
                "notes": "Use this AI result with local soil testing and expert guidance.",
            },
        )

        return {
            "crop": top_label.title(),
            "confidence": round((top_score / total_score) * 100, 1),
            "alternatives": alternatives,
            "details": details,
            "dataset_name": self.dataset_path.name,
            "dataset_rows": len(self.rows),
            "crop_count": len(self.label_counts),
        }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-before-deployment")
    DATA_DIR.mkdir(exist_ok=True)
    init_db()

    @app.context_processor
    def inject_now():
        return {"current_year": datetime.now().year}

    @app.route("/")
    def index():
        return render_template("index.html", dataset_status=get_dataset_status())

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not name or not email or len(password) < 6:
                flash("Please enter your name, email, and a password of at least 6 characters.", "error")
                return redirect(url_for("register"))

            try:
                with get_db() as db:
                    cursor = db.execute(
                        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                        (name, email, generate_password_hash(password)),
                    )
                    db.commit()
                    session["user_id"] = cursor.lastrowid
                    session["user_name"] = name
                    flash("Account created. You can now generate crop recommendations.", "success")
                    return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("An account with this email already exists.", "error")

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            with get_db() as db:
                user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                flash("Welcome back. Your farm dashboard is ready.", "success")
                return redirect(url_for("dashboard"))

            flash("Invalid email or password.", "error")

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for("index"))

    @app.route("/dashboard", methods=["GET", "POST"])
    @login_required
    def dashboard():
        recommendation = None
        prefill = get_prefill_values()
        if request.method == "POST":
            try:
                inputs = parse_inputs(request.form)
                recommender = CropRecommender(active_dataset_path())
                recommendation = recommender.recommend(inputs)
                recommendation["soil_health"] = calculate_soil_health(inputs)
                recommendation["inputs"] = inputs
                save_recommendation(session["user_id"], inputs, recommendation)
                flash("Recommendation generated and saved to your history.", "success")
                return redirect(url_for("report", recommendation_id=get_latest_recommendation_id(session["user_id"])))
            except (ValueError, RuntimeError, KeyError) as error:
                flash(str(error), "error")

        recent_history = get_history(session["user_id"], limit=5)
        return render_template(
            "dashboard.html",
            recommendation=recommendation,
            prefill=prefill,
            history=recent_history,
            dataset_status=get_dataset_status(),
            mineral_references=MINERAL_REFERENCES,
            latest_iot=get_latest_iot_reading(session["user_id"]),
        )

    @app.route("/history")
    @login_required
    def history():
        return render_template(
            "history.html",
            history=get_history(session["user_id"], limit=50),
            mineral_references=MINERAL_REFERENCES,
        )

    @app.route("/report/<int:recommendation_id>")
    @login_required
    def report(recommendation_id: int):
        item = get_recommendation(session["user_id"], recommendation_id)
        if not item:
            flash("Report not found for this account.", "error")
            return redirect(url_for("history"))
        return render_template("report.html", item=item, mineral_references=MINERAL_REFERENCES)

    @app.route("/report/<int:recommendation_id>/pdf")
    @login_required
    def report_pdf(recommendation_id: int):
        item = get_recommendation(session["user_id"], recommendation_id)
        if not item:
            flash("Report not found for this account.", "error")
            return redirect(url_for("history"))
        pdf = build_report_pdf(item)
        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"agrisense-report-{recommendation_id}.pdf",
        )

    @app.route("/api/iot/simulate", methods=["POST"])
    @login_required
    def simulate_iot():
        reading = generate_iot_reading()
        save_iot_reading(session["user_id"], reading)
        return jsonify(reading)

    @app.route("/iot")
    @login_required
    def iot():
        latest = get_latest_iot_reading(session["user_id"])
        return render_template("iot.html", latest=latest)

    return app


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                ph REAL NOT NULL,
                rainfall REAL NOT NULL,
                crop TEXT NOT NULL,
                confidence REAL NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                ph REAL NOT NULL,
                rainfall REAL NOT NULL,
                soil_moisture REAL NOT NULL,
                device_status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        db.commit()


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the farmer dashboard.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapper


def active_dataset_path() -> Path:
    return KAGGLE_DATASET if KAGGLE_DATASET.exists() else SAMPLE_DATASET


def get_dataset_status() -> dict:
    path = active_dataset_path()
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = max(sum(1 for _ in file) - 1, 0)
    except FileNotFoundError:
        rows = 0

    return {
        "file": path.name,
        "rows": rows,
        "is_kaggle": path == KAGGLE_DATASET,
    }


def parse_inputs(form) -> dict[str, float]:
    labels = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall",
    }
    ranges = {
        "N": (0, 200),
        "P": (0, 200),
        "K": (0, 250),
        "temperature": (-10, 60),
        "humidity": (0, 100),
        "ph": (0, 14),
        "rainfall": (0, 400),
    }
    values = {}
    for feature in FEATURES:
        try:
            value = float(form.get(feature, ""))
        except ValueError as exc:
            raise ValueError(f"{labels[feature]} must be a number.") from exc

        low, high = ranges[feature]
        if value < low or value > high:
            raise ValueError(f"{labels[feature]} must be between {low} and {high}.")
        values[feature] = value
    return values


def save_recommendation(user_id: int, inputs: dict[str, float], recommendation: dict) -> None:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO recommendations (
                user_id, nitrogen, phosphorus, potassium, temperature, humidity,
                ph, rainfall, crop, confidence, details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                inputs["N"],
                inputs["P"],
                inputs["K"],
                inputs["temperature"],
                inputs["humidity"],
                inputs["ph"],
                inputs["rainfall"],
                recommendation["crop"],
                recommendation["confidence"],
                json.dumps(recommendation),
            ),
        )
        db.commit()


def get_latest_recommendation_id(user_id: int) -> int | None:
    with get_db() as db:
        row = db.execute(
            """
            SELECT id FROM recommendations
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    return row["id"] if row else None


def get_history(user_id: int, limit: int = 25) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT * FROM recommendations
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item["details"])
        history.append(item)
    return history


def get_recommendation(user_id: int, recommendation_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM recommendations WHERE user_id = ? AND id = ?",
            (user_id, recommendation_id),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["details"] = json.loads(item["details"])
    item["soil_health"] = item["details"].get(
        "soil_health",
        calculate_soil_health(
            {
                "N": item["nitrogen"],
                "P": item["phosphorus"],
                "K": item["potassium"],
                "ph": item["ph"],
            }
        ),
    )
    return item


def get_prefill_values() -> dict[str, float]:
    history_id = request.args.get("history_id", type=int)
    source = request.args.get("source", "")
    if history_id:
        item = get_recommendation(session["user_id"], history_id)
        if item:
            return {
                "N": item["nitrogen"],
                "P": item["phosphorus"],
                "K": item["potassium"],
                "temperature": item["temperature"],
                "humidity": item["humidity"],
                "ph": item["ph"],
                "rainfall": item["rainfall"],
            }
    if source == "iot":
        latest = get_latest_iot_reading(session["user_id"])
        if latest:
            return {
                "N": latest["nitrogen"],
                "P": latest["phosphorus"],
                "K": latest["potassium"],
                "temperature": latest["temperature"],
                "humidity": latest["humidity"],
                "ph": latest["ph"],
                "rainfall": latest["rainfall"],
            }
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 24,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202,
    }


def classify_value(key: str, value: float) -> dict:
    low, high = REFERENCE_BOUNDS[key]
    if value < low:
        return {"status": "Low", "class": "low", "message": "Needs attention"}
    if value > high:
        return {"status": "High", "class": "high", "message": "Above ideal"}
    return {"status": "Optimal", "class": "optimal", "message": "Within range"}


def calculate_soil_health(inputs: dict[str, float]) -> dict:
    indicators = []
    score = 0
    for key in ("N", "P", "K", "ph"):
        value = float(inputs[key])
        classification = classify_value(key, value)
        if classification["class"] == "optimal":
            score += 25
        elif classification["class"] == "high":
            score += 14
        else:
            score += 9
        indicators.append(
            {
                "key": key,
                "name": next(item["name"] for item in MINERAL_REFERENCES if item["key"] == key),
                "value": round(value, 1),
                **classification,
            }
        )
    if score >= 80:
        label = "Excellent"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Needs Care"
    return {"score": score, "label": label, "indicators": indicators}


def generate_iot_reading() -> dict[str, float | str]:
    return {
        "N": round(random.uniform(35, 115), 1),
        "P": round(random.uniform(24, 70), 1),
        "K": round(random.uniform(32, 115), 1),
        "temperature": round(random.uniform(18, 34), 1),
        "humidity": round(random.uniform(50, 92), 1),
        "ph": round(random.uniform(5.6, 7.4), 1),
        "rainfall": round(random.uniform(55, 245), 1),
        "soil_moisture": round(random.uniform(28, 76), 1),
        "device_status": "Online",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_iot_reading(user_id: int, reading: dict) -> None:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO sensor_readings (
                user_id, nitrogen, phosphorus, potassium, temperature, humidity,
                ph, rainfall, soil_moisture, device_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                reading["N"],
                reading["P"],
                reading["K"],
                reading["temperature"],
                reading["humidity"],
                reading["ph"],
                reading["rainfall"],
                reading["soil_moisture"],
                reading["device_status"],
            ),
        )
        db.commit()


def get_latest_iot_reading(user_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            """
            SELECT * FROM sensor_readings
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return dict(row)


def build_report_pdf(item: dict) -> BytesIO:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("Install reportlab to download PDF reports.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AgriSense AI Field Recommendation Report", styles["Title"]),
        Paragraph(f"Generated: {item['created_at']}", styles["Normal"]),
        Spacer(1, 0.18 * inch),
        Paragraph(f"Recommended crop: {item['crop']}", styles["Heading2"]),
        Paragraph(f"Confidence: {item['confidence']}%", styles["Normal"]),
        Paragraph(f"Soil health: {item['soil_health']['label']} ({item['soil_health']['score']}%)", styles["Normal"]),
        Spacer(1, 0.18 * inch),
    ]

    data = [
        ["Parameter", "Value"],
        ["Nitrogen", f"{item['nitrogen']} kg/ha"],
        ["Phosphorus", f"{item['phosphorus']} kg/ha"],
        ["Potassium", f"{item['potassium']} kg/ha"],
        ["Temperature", f"{item['temperature']} C"],
        ["Humidity", f"{item['humidity']}%"],
        ["Soil pH", item["ph"]],
        ["Rainfall", f"{item['rainfall']} mm"],
    ]
    table = Table(data, colWidths=[2.2 * inch, 3.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f8f4d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c8bb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Advisory", styles["Heading2"]))
    story.append(Paragraph(item["details"]["details"]["notes"], styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
