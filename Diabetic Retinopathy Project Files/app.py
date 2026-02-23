from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.secret_key = "change_this_secret_to_secure_key"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "updated_xception_diabetic_retinopathy.h5")
model = load_model(MODEL_PATH)

CLASS_LABELS = [
    "No disease visible",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
        
        if row and check_password_hash(row[2], password):
            session["user"] = {"id": row[0], "name": row[1], "email": email}
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid email or password")
    return render_template("login.html", error=None)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        
        if not name or not email or not password:
            return render_template("register.html", error="All fields are required")
        
        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, hashed)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already registered")
    return render_template("register.html", error=None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    
    prediction_text = None
    uploaded_filename = None
    
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            
            uploaded_filename = filename
            img = load_img(path, target_size=(299, 299))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)
            class_index = int(np.argmax(preds, axis=1)[0])
            prediction_text = CLASS_LABELS[class_index]
    
    return render_template(
        "predict.html",
        prediction=prediction_text,
        uploaded_image=uploaded_filename
    )

@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)