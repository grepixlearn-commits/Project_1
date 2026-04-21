import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, text
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────
# 1. DATABASE CONNECTION
# ─────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT", 3306))

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
    raise ValueError("Missing required database config in .env file. "
                     "Copy .env.example to .env and fill in your credentials.")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True
)

# ─────────────────────────────────────────
# 2. MODEL LOADING
# ─────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH")

if not MODEL_PATH:
    raise ValueError(" MODEL_PATH not set in .env file")

print(f"\nUsing MODEL PATH: {MODEL_PATH}")

tokenizer = None
model = None

def load_model():
    global tokenizer, model

    print("\n--- Loading Model ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if MODEL_PATH.startswith("/") and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    print("Model Loaded Successfully")
    print("Model device:", next(model.parameters()).device)

# ─────────────────────────────────────────
# 3. TABLE MAPPING
# ─────────────────────────────────────────
TABLE_MAP = {
    "city": "cities", "cities": "cities",
    "category": "categories", "categories": "categories",
    "promo": "promos", "promos": "promos", "promo code": "promos",
    "user": "users", "users": "users",
    "driver": "drivers", "drivers": "drivers",
    "trip": "m_trips", "trips": "m_trips",
    "car": "cars", "cars": "cars",
    "constant": "constants", "constants": "constants"
}

def detect_table(question):
    q = question.lower()
    for key, table in TABLE_MAP.items():
        if key in q:
            return table
    return None

# ─────────────────────────────────────────
# 4. DYNAMIC COLUMN FETCH
# ─────────────────────────────────────────
def get_table_columns(table_name):
    try:
        df = pd.read_sql(f"SHOW COLUMNS FROM {table_name}", engine)
        return ", ".join(df["Field"].tolist())
    except Exception as e:
        return ""

# ─────────────────────────────────────────
# 5. SQL GENERATION
# ─────────────────────────────────────────
def generate_sql(user_question):
    current_date = datetime.now().strftime("%Y-%m-%d")
    available_tables = list(set(TABLE_MAP.values()))
    table = detect_table(user_question)

    column_info = ""
    if table:
        cols = get_table_columns(table)
        column_info = f"Table {table} has columns: {cols}\n"

    prompt = (
        "<|im_start|>system\n"
        "You are a SQL expert. Output ONLY a valid SQL query — no explanation, no markdown, no backticks.\n"
        "COLUMN SELECTION RULE (HIGHEST PRIORITY):\n"
        "- Read the user question and identify exactly which columns they asked for.\n"
        "- SELECT only those exact columns — no more, no less.\n"
        "- If user asks for 1 column → SELECT 1 column only.\n"
        "- If user asks for 3 columns → SELECT exactly those 3 columns only.\n"
        "- NEVER add extra columns the user did not mention.\n"
        "- 'all users' or 'list users' means all ROWS — still SELECT only the asked/identifier column.\n\n"
        "STRICT RULES:\n"
        f"- You can ONLY use these tables: {available_tables}\n"
        "- NEVER invent or use a table not in the list above\n"
        "- NEVER use SELECT *\n"
        "- Use DISTINCT when needed to remove duplicate results\n"
        "- Select ONLY columns that answer the question\n"
        "- Use ONLY column names that exist in the table\n"
        "- Do NOT guess or create column names\n"
        "- Always limit results to 20 rows using LIMIT 20\n"
        "- End query with a semicolon\n"
        "- Understand natural language queries instead of relying on fixed phrases\n"
        f"{column_info}"
        "IMPORTANT FILTER RULES:\n"
        "- DO NOT apply any filters like status, active, is_delete, or date conditions by default\n"
        "- ONLY apply filters when the user explicitly asks\n"
        "PROMO CODE RULES:\n"
        f"- Current date is: {current_date}\n"
        "- 'active promo codes' → WHERE promo_end_date >= current date AND is_delete = 0\n"
        "- 'expired promo codes' → WHERE promo_end_date < current date\n"
        "CITY RULES:\n"
        "- 'list of cities' → return city_name only, no filters\n"
        "- Do NOT apply city_active or is_delete unless explicitly asked\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {user_question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "SELECT"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Uncomment for TPU:
    # torch_xla.sync()

    input_len = inputs["input_ids"].shape[1]
    output_ids = outputs[0][input_len:].detach().cpu()
    sql_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    sql = sql_text.replace("```sql", "").replace("```", "").strip()
    sql = sql.split(";")[0].strip()

    if not sql or len(sql) < 10:
        return "SELECT 'Model failed to generate SQL' AS error;"

    if not sql.upper().startswith("SELECT"):
        sql = "SELECT " + sql

    return sql + ";"

# ─────────────────────────────────────────
# 6. RUN QUERY
# ─────────────────────────────────────────
def run_query(sql):
    try:
        df = pd.read_sql(sql, engine)
        if df.empty:
            return {"columns": [], "rows": [], "message": "No results found."}
        return {
            "columns": df.columns.tolist(),
            "rows": df.values.tolist(),
            "message": f"{len(df)} row(s) returned"
        }
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────
# 7. FLASK ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 503

    try:
        sql = generate_sql(question)
        result = run_query(sql)
        result["sql"] = sql
        result["question"] = question
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "sql": ""}), 500

@app.route("/tables", methods=["GET"])
def get_tables():
    tables = list(set(TABLE_MAP.values()))
    table_info = {}
    for t in tables:
        cols = get_table_columns(t)
        table_info[t] = cols.split(", ") if cols else []
    return jsonify(table_info)

@app.route("/health", methods=["GET"])
def health():
    model_status = "loaded" if model is not None else "not loaded"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    return jsonify({"model": model_status, "database": db_status})

# ─────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"
    )
