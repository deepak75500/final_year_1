from flask import Flask, render_template, request, jsonify
from db_schema1 import (
    list_mysql_databases, list_mongo_databases,
    extract_mysql_schema, extract_mongo_schema,
    build_embeddings_and_index, semantic_search,
    load_ai_model, generate_query, execute_sql_query, execute_mongo_query
)

app = Flask(__name__)

# ----------------------------
# 🔧 Initialization
# ----------------------------
mysql_user = "root"
mysql_password = "root"
mysql_host = "localhost"
mysql_port = 3308
mongo_uri = "mongodb://localhost:27017"

print("🔍 Initializing schemas and model...")

all_schemas = {"MySQL": {}, "MongoDB": {}}
mysql_dbs = list_mysql_databases(mysql_user, mysql_password, mysql_host, mysql_port)
mongo_dbs = list_mongo_databases(mongo_uri)

for dbname in mysql_dbs:
    conn_str = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{dbname}"
    all_schemas["MySQL"][dbname] = extract_mysql_schema(conn_str)

for dbname in mongo_dbs:
    all_schemas["MongoDB"][dbname] = extract_mongo_schema(mongo_uri, dbname)

docs = []
for db_type, dbs in all_schemas.items():
    for dbname, schema in dbs.items():
        for table, cols in schema.items():
            schema_text = f"{db_type}::{dbname}::{table} -> " + ", ".join([c['Field'] for c in cols])
            docs.append(schema_text)

embedder, index, _ = build_embeddings_and_index(docs)
generator, model_used = load_ai_model()
print(f"✅ Model loaded: {model_used}")


# ----------------------------
# 🌍 Routes
# ----------------------------
@app.route("/")
def chat_ui():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    matches = semantic_search(user_message, embedder, index, docs)
    best_match = matches[0][0]
    db_type = "MySQL" if "MySQL::" in best_match else "MongoDB"
    dbname = best_match.split("::")[1]

    gen_query = generate_query(generator, user_message, best_match)

    # Execute query
    try:
        if db_type == "MySQL":
            conn_str = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{dbname}"
            results = execute_sql_query(conn_str, gen_query)
        else:
            results = execute_mongo_query(mongo_uri, dbname, gen_query)
    except Exception as e:
        results = [f"❌ Execution Error: {str(e)}"]

    # Prepare bot response
    bot_reply = (
        f"**Closest Schema:** {best_match}\n\n"
        f"**Generated Query:**\n```\n{gen_query}\n```\n"
        f"**Results:**\n{results if results else 'No data found.'}"
    )

    return jsonify({"reply": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
