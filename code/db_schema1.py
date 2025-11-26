import mysql.connector
from sqlalchemy import create_engine, inspect, text
from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# ==============================================
# 1️⃣  MySQL Functions
# ==============================================
def list_mysql_databases(user, password, host="localhost", port=3308):
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password, port=port)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [db[0] for db in cursor.fetchall()]
        cursor.close()
        conn.close()
        return [db for db in dbs if db not in {"information_schema", "mysql", "performance_schema", "sys"}]
    except Exception as e:
        print(f"❌ MySQL connection error: {e}")
        return []

def extract_mysql_schema(conn_str):
    try:
        engine = create_engine(conn_str)
        inspector = inspect(engine)
        schema = {}
        for table in inspector.get_table_names():
            columns = inspector.get_columns(table)
            schema[table] = [{"Field": col["name"], "Type": str(col["type"])} for col in columns]
        return schema
    except Exception as e:
        print(f"❌ MySQL schema extraction error: {e}")
        return {}

# ==============================================
# 2️⃣  MongoDB Functions
# ==============================================
def list_mongo_databases(uri="mongodb://localhost:27017"):
    try:
        client = MongoClient(uri)
        return [db for db in client.list_database_names() if db not in {"admin", "config", "local"}]
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return []

def extract_mongo_schema(uri, dbname, sample_size=5):
    try:
        client = MongoClient(uri)
        db = client[dbname]
        schema = {}
        for coll_name in db.list_collection_names():
            coll = db[coll_name]
            sample_docs = list(coll.find().limit(sample_size))
            fields = set()
            for doc in sample_docs:
                fields.update(doc.keys())
            schema[coll_name] = [{"Field": f, "Type": "Unknown"} for f in fields]
        return schema
    except Exception as e:
        print(f"❌ Mongo schema extraction error: {e}")
        return {}

# ==============================================
# 3️⃣  Semantic Index
# ==============================================
def build_embeddings_and_index(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return embedder, index, embeddings

def semantic_search(query, embedder, index, docs, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ==============================================
# 4️⃣  Model Loader (Auto-select)
# ==============================================
def load_ai_model():
    if torch.cuda.is_available():
        print("⚡ GPU detected — loading CodeLlama (fast on GPU)...")
        model_name = "codellama/CodeLlama-7b-Instruct-hf"
    else:
        print("💨 No GPU detected — using faster CPU model (Mistral).")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1
    )
    return generator, model_name

# ==============================================
# 5️⃣  Query Generation
# ==============================================
import re

# =================================================
# 5️⃣  Query Generation (cleaned)
# =================================================
def generate_query(generator, natural_query, schema_hint):
    prompt = f"""
You are an expert database assistant.
Given this schema info, decide if the query should be SQL (for MySQL) or MongoDB (for NoSQL).
Then generate ONLY the query — no explanation.

Schema:
{schema_hint}

User request:
{natural_query}

Output only the query, nothing else.
"""
    output = generator(prompt, max_new_tokens=200, do_sample=False)
    gen_text = output[0]['generated_text']

    # Extract only the part after prompt
    query = gen_text.split("Output only the query, nothing else.")[-1].strip()

    # 🔹 Clean unwanted prefixes like "SQL:" or "Query:"
    query = re.sub(r"^(SQL|Query|MongoDB|Mongo|Command)[:\- ]+", "", query, flags=re.IGNORECASE).strip()

    # 🔹 Ensure it ends with semicolon for SQL
    if not query.endswith(";") and "find(" not in query:
        query += ";"

    return query


# =================================================
# 6️⃣  Execute Query (improved messages)
# =================================================
def execute_sql_query(conn_str, query):
    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                print("ℹ️ No results found.")
            return [dict(r._mapping) for r in rows]
    except Exception as e:
        print("\n❌ SQL Execution Error:")
        print(f"   {str(e).splitlines()[0]}")
        print("💡 Tip: The generated SQL might include extra words or wrong syntax.")
        print("🔎 Please review this query:\n", query)
        return []
def execute_mongo_query(uri, dbname, query_text):
    client = MongoClient(uri)
    db = client[dbname]
    try:
        # Expect query_text like db.collection.find({...})
        result = eval(query_text)
        return list(result)
    except Exception as e:
        return [f"❌ Mongo query execution error: {e}"]

# ==============================================
# 7️⃣  Main
# ==============================================
if __name__ == "__main__":
    mysql_user = "root"
    mysql_password = "root"
    mysql_host = "localhost"
    mysql_port = 3308
    mongo_uri = "mongodb://localhost:27017"

    print("🔍 Scanning available databases...\n")

    all_schemas = {"MySQL": {}, "MongoDB": {}}
    mysql_dbs = list_mysql_databases(mysql_user, mysql_password, mysql_host, mysql_port)
    mongo_dbs = list_mongo_databases(mongo_uri)

    # Load MySQL schemas
    for dbname in mysql_dbs:
        conn_str = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{dbname}"
        all_schemas["MySQL"][dbname] = extract_mysql_schema(conn_str)
        print(f"✅ MySQL schema loaded: {dbname}")

    # Load Mongo schemas
    for dbname in mongo_dbs:
        all_schemas["MongoDB"][dbname] = extract_mongo_schema(mongo_uri, dbname)
        print(f"✅ MongoDB schema loaded: {dbname}")

    # Build documents for semantic context
    docs = []
    for db_type, dbs in all_schemas.items():
        for dbname, schema in dbs.items():
            for table, cols in schema.items():
                schema_text = f"{db_type}::{dbname}::{table} -> " + ", ".join([c['Field'] for c in cols])
                docs.append(schema_text)

    print("\n⚙️ Building semantic index...")
    embedder, index, _ = build_embeddings_and_index(docs)
    print("✅ Semantic index built successfully.\n")

    generator, model_used = load_ai_model()
    print(f"🤖 Using model: {model_used}\n")

    while True:
        user_query = input("\n🧠 Enter your natural language query (or 'exit'): ")
        if user_query.lower() == "exit":
            print("👋 Exiting.")
            break

        matches = semantic_search(user_query, embedder, index, docs)
        best_match = matches[0][0]
        print(f"\n🔎 Closest schema: {best_match}\n")

        db_type = "MySQL" if "MySQL::" in best_match else "MongoDB"
        dbname = best_match.split("::")[1]

        gen_query = generate_query(generator, user_query, best_match)
        print(f"💡 Generated Query:\n{gen_query}\n")

        if db_type == "MySQL":
            conn_str = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{dbname}"
            try:
                results = execute_sql_query(conn_str, gen_query)
                print("📊 Query Results:")
                for r in results:
                    print(r)
            except Exception as e:
                print(f"❌ SQL Execution Error: {e}")
        else:
            try:
                results = execute_mongo_query(mongo_uri, dbname, gen_query)
                print("📊 Query Results:")
                for r in results:
                    print(r)
            except Exception as e:
                print(f"❌ MongoDB Execution Error: {e}")
