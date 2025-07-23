from flask import Flask, request, jsonify, send_file, send_from_directory, make_response, redirect, url_for, session, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import threading
import pandas as pd
import requests
import time
import os
import tempfile
import zipfile
import shutil
import logging
import csv
import sqlite3
import asyncio
import aiohttp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import redis
import pickle
from functools import lru_cache
import hashlib

app = Flask(__name__, static_folder='../frontend')
app.secret_key = 'supersecretkey'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Redis for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Redis connected successfully")
    REDIS_AVAILABLE = True
except:
    print("Redis not available, using memory cache")
    REDIS_AVAILABLE = False
    memory_cache = {}

# Load model
try:
    model_options = [
        "paraphrase-albert-small-v2",
        "all-distilroberta-v1",
        "paraphrase-TinyBERT-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    model = None
    for model_name in model_options:
        try:
            model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                model = model.half()
                print(f"Loaded {model_name} model with GPU acceleration")
            else:
                print(f"Loaded {model_name} model (CPU)")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    if model is None:
        raise Exception("No models could be loaded")
except Exception as e:
    print(f"Model loading error: {e}")
    model = SentenceTransformer("all-MiniLM-L6-v2")

pause_event = threading.Event()
processing_thread = None
in_memory_results = {}
uploaded_file_path = None
results_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'results')
os.makedirs(results_dir, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=10)

def get_db():
    db = sqlite3.connect('users.db', timeout=20)
    db.execute('PRAGMA journal_mode=WAL')
    return db

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user'
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_role ON users(role)')
    db.commit()
    db.close()

init_db()

logging.basicConfig(level=logging.INFO, filename=os.path.join(results_dir, 'processing.log'), filemode='w',
                    format='%(asctime)s - %(message)s')

def get_cache_key(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_cached_embedding(content):
    cache_key = get_cache_key(content)
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(f"embedding:{cache_key}")
            if cached:
                return pickle.loads(cached.encode('latin-1'))
        except:
            pass
    else:
        if cache_key in memory_cache:
            return memory_cache[cache_key]
    return None

def cache_embedding(content, embedding):
    cache_key = get_cache_key(content)
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(f"embedding:{cache_key}", 3600, pickle.dumps(embedding).decode('latin-1'))
        except:
            pass
    else:
        memory_cache[cache_key] = embedding

async def scrape_webpage_async(session, url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return text[:1000] if len(text) > 1000 else text
            return None
    except Exception as e:
        logging.warning(f"Failed to scrape {url}: {e}")
        return None

def scrape_webpage(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:1000] if len(text) > 1000 else text
    except requests.RequestException as e:
        logging.warning(f"Failed to scrape {url}: {e}")
        return None

async def process_urls_batch_async(urls, keywords, keyword_embeddings):
    results = {kw.strip(): [] for kw in keywords}
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_webpage_async(session, url) for url in urls]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            content = await task
            url = urls[i]
            if content and not pause_event.is_set():
                content_embedding = get_cached_embedding(content)
                if content_embedding is None:
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    cache_embedding(content, content_embedding)
                for keyword in keywords:
                    keyword_key = keyword.strip()
                    similarity = util.pytorch_cos_sim(content_embedding, keyword_embeddings[keyword_key]).item()
                    if similarity > 0.4:
                        results[keyword_key].append(url)
    return results

def process_urls_batch(urls_batch, keywords, keyword_embeddings):
    batch_results = {kw.strip(): [] for kw in keywords}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_webpage, url): url for url in urls_batch}
        for future in as_completed(future_to_url):
            if pause_event.is_set():
                break
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    content_embedding = get_cached_embedding(content)
                    if content_embedding is None:
                        content_embedding = model.encode(content, convert_to_tensor=True)
                        cache_embedding(content, content_embedding)
                    for keyword in keywords:
                        keyword_key = keyword.strip()
                        similarity = util.pytorch_cos_sim(content_embedding, keyword_embeddings[keyword_key]).item()
                        if similarity > 0.4:
                            batch_results[keyword_key].append(url)
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
    return batch_results

def process_file(file_path, keywords):
    global in_memory_results, pause_event
    in_memory_results = {kw.strip(): [] for kw in keywords}
    for keyword in keywords:
        keyword_dir = os.path.join(results_dir, keyword.strip())
        os.makedirs(keyword_dir, exist_ok=True)
        csv_file_path = os.path.join(keyword_dir, "results.csv")
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["URL"])

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            socketio.emit("processing_error", {"error": "Unsupported file format. Please use CSV or Excel files."})
            return
        urls = df.iloc[:, 0].dropna().tolist()
        if not urls:
            socketio.emit("processing_error", {"error": "No URLs found in the file. Make sure URLs are in the first column."})
            return
    except Exception as e:
        socketio.emit("processing_error", {"error": f"Error reading file: {str(e)}"})
        return

    total_urls = len(urls)
    processed_urls = 0
    keyword_embeddings = {}
    for keyword in keywords:
        keyword_key = keyword.strip()
        cached_embedding = get_cached_embedding(keyword_key)
        if cached_embedding is not None:
            keyword_embeddings[keyword_key] = cached_embedding
        else:
            embedding = model.encode(keyword_key, convert_to_tensor=True)
            keyword_embeddings[keyword_key] = embedding
            cache_embedding(keyword_key, embedding)

    batch_size = 10
    for i in range(0, len(urls), batch_size):
        if pause_event.is_set():
            while pause_event.is_set():
                time.sleep(0.5)
        batch_urls = urls[i:i + batch_size]
        batch_results = process_urls_batch(batch_urls, keywords, keyword_embeddings)
        for keyword_key, urls_found in batch_results.items():
            in_memory_results[keyword_key].extend(urls_found)
            if urls_found:
                keyword_dir = os.path.join(results_dir, keyword_key)
                csv_file_path = os.path.join(keyword_dir, "results.csv")
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for url in urls_found:
                        writer.writerow([url])

        processed_urls += len(batch_urls)
        progress = int((processed_urls / total_urls) * 100)
        socketio.emit("update_progress", {
            "progress": progress,
            "processed_urls": processed_urls,
            "total_urls": total_urls,
            "current_batch": f"Batch {i // batch_size + 1}",
            "percentage": f"{progress}%"
        })
        logging.info(f"Processed batch {i // batch_size + 1} ({progress}%): {len(batch_urls)} URLs")

    socketio.emit("processing_ended")

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password)
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
            db.commit()
            db.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            db.close()
            return "Email already exists!"
    return send_from_directory('../frontend', 'signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        db.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            if user[3] == 'admin':
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('index'))
        else:
            return "Invalid credentials!"
    return send_from_directory('../frontend', 'login.html')

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route("/upload", methods=["POST"])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    global processing_thread, pause_event, uploaded_file_path
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    keywords = request.form.get("keywords", "").split(",")
    if file.filename == "" or not keywords:
        return jsonify({"error": "Invalid input"}), 400
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": "File type not supported. Please upload CSV or Excel files."}), 400
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    file.save(temp.name)
    uploaded_file_path = temp.name
    pause_event.clear()
    processing_thread = threading.Thread(target=process_file, args=(temp.name, keywords))
    processing_thread.start()
    return jsonify({"message": "Processing started! Using optimized parallel processing."})

@app.route("/download_results", methods=["POST"])
def download_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    global uploaded_file_path, in_memory_results, results_dir
    keywords = request.json.get("keywords", [])
    temp_dir = tempfile.mkdtemp()
    uploaded_files_dir = os.path.join(temp_dir, "uploaded_files")
    results_files_dir = os.path.join(temp_dir, "results_files")
    os.makedirs(uploaded_files_dir)
    os.makedirs(results_files_dir)
    uploaded_file_name = os.path.basename(uploaded_file_path)
    shutil.copy(uploaded_file_path, os.path.join(uploaded_files_dir, uploaded_file_name))
    for keyword in keywords:
        keyword_dir = os.path.join(results_files_dir, keyword.strip())
        os.makedirs(keyword_dir)
        with open(os.path.join(keyword_dir, "results.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["URL"])
            for url in in_memory_results.get(keyword.strip(), []):
                writer.writerow([url])
    zip_path = os.path.join(temp_dir, "results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file != "results.zip":
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
    def cleanup():
        time.sleep(1)
        shutil.rmtree(temp_dir, ignore_errors=True)
    threading.Thread(target=cleanup).start()
    response = make_response(send_from_directory(temp_dir, "results.zip", as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename=results.zip"
    return response

@app.route("/get_results", methods=["POST"])
def get_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    global in_memory_results
    keywords = request.json.get("keywords", [])
    return jsonify({kw: in_memory_results.get(kw, []) for kw in keywords})


@app.route("/admin")
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, email, role FROM users")
    users = cursor.fetchall()
    db.close()
    return render_template('admin.html', users=users)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Prevent deleting yourself
    if session['user_id'] == user_id:
        return "You cannot delete yourself!", 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    db.close()
    return redirect(url_for('admin'))

@app.route("/register_admin", methods=["GET", "POST"])
def register_admin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password)
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, password, role) VALUES (?, ?, 'admin')",
                (email, hashed_password)
            )
            db.commit()
            db.close()
            return "Admin registration successful! You can now <a href='/login'>login</a>."
        except sqlite3.IntegrityError:
            db.close()
            return "Email already exists!"
    return '''
        <form method="post">
            Email: <input type="email" name="email" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Register Admin">
        </form>
    '''

@app.route("/stats")
def stats():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    cache_stats = {}
    if REDIS_AVAILABLE:
        try:
            info = redis_client.info()
            cache_stats = {
                "cache_type": "Redis",
                "used_memory": info.get('used_memory_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 'N/A')
            }
        except:
            cache_stats = {"cache_type": "Redis (Error)", "status": "Connection failed"}
    else:
        cache_stats = {
            "cache_type": "Memory",
            "cached_items": len(memory_cache)
        }
    return jsonify({
        "model": str(model),
        "cache_stats": cache_stats,
        "gpu_available": torch.cuda.is_available(),
        "thread_pool_size": executor._max_workers
    })

@socketio.on("pause")
def handle_pause():
    pause_event.set()
    emit("pause_acknowledged")

@socketio.on("resume")
def handle_resume():
    pause_event.clear()
    emit("resume_acknowledged")

if __name__ == "__main__":
    port = 5000
    host = '127.0.0.1'
    print(f" * Running on http://{host}:{port}/ (Press CTRL+C to quit)")
    print(f" * Click the link to access the app: http://{host}:{port}/")
    print(f" * Using model: {model}")
    print(f" * Cache: {'Redis' if REDIS_AVAILABLE else 'Memory'}")
    print(f" * GPU: {'Available' if torch.cuda.is_available() else 'Not available'}")
    print(f" * Thread pool: {executor._max_workers} workers")
    socketio.run(app, debug=True, port=port, host=host, allow_unsafe_werkzeug=True)
