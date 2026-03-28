import json
from pathlib import Path
from elasticsearch import Elasticsearch, helpers

# ── Configuration ────────────────────────────────────────────────
ES_HOST = "http://localhost:9200"
INDEX_NAME = "embedding_stats"
INPUT_DIR = Path(__file__).parent.parent / "results" / "embedding" / "embedding_outputs"

# ── Connexion ─────────────────────────────────────────────────────
es = Elasticsearch(ES_HOST)

if not es.ping():
    raise ConnectionError(f"Impossible de se connecter à Elasticsearch sur {ES_HOST}")

print(f"Connecté à Elasticsearch : {ES_HOST}")

# ── Mapping de l'index ────────────────────────────────────────────
mapping = {
    "mappings": {
        "properties": {
            "id":                   {"type": "integer"},
            "model":                {"type": "keyword"},
            "BehaviorID":           {"type": "keyword"},
            "category":             {"type": "keyword"},
            "SemanticCategory":     {"type": "keyword"},
            "label":                {"type": "keyword"},
            "harm":                 {"type": "float"},
            "refusal":              {"type": "float"},
            "actionability":        {"type": "float"},
            "copyright_reproduction": {"type": "float"},
            "score":                {"type": "float"},
            "score_base":           {"type": "float"},
            "score_context":        {"type": "float"},
            "context_resistance":   {"type": "float"},
            "delta_score":          {"type": "float"},
            "source_file":          {"type": "keyword"},
        }
    }
}

if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Index '{INDEX_NAME}' créé.")
else:
    print(f"Index '{INDEX_NAME}' déjà existant, on réutilise.")

# ── Chargement des fichiers ───────────────────────────────────────
def load_jsonl(file_path: Path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().strip()

    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    for line in data.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def generate_actions(directory: Path):
    files = sorted(directory.glob("*.jsonl"))
    total = 0
    for file_path in files:
        records = load_jsonl(file_path)
        for record in records:
            if isinstance(record, dict):
                record["source_file"] = file_path.name
                yield {
                    "_index": INDEX_NAME,
                    "_source": record,
                }
                total += 1
    print(f"{total} documents chargés depuis {len(files)} fichiers.")


# ── Indexation ────────────────────────────────────────────────────
success, errors = helpers.bulk(es, generate_actions(INPUT_DIR), raise_on_error=False)
print(f"Documents indexés avec succès : {success}")
if errors:
    print(f"Erreurs ({len(errors)}) :")
    for err in errors[:5]:
        print(" ", err)
