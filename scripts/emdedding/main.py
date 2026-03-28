import json
from collections import defaultdict

from sentence_transformers import SentenceTransformer

import config
import utils
import scoring


print("Loading files...")
outputs = utils.load_jsonl(config.MODEL_OUTPUTS_PATH)
targets = utils.load_jsonl(config.TARGETS_PATH)

print("Loading embedding model...")
embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

print("Encoding targets...")
# On encode toutes les variantes de chaque target et on stocke tous les vecteurs
target_embeddings = {}
for item in targets:
    tid = item["target_id"]
    texts = [utils.normalize_text(t) for t in item["target_texts"]]
    vecs = embedder.encode(texts, convert_to_numpy=True)
    target_embeddings[tid] = vecs  # shape: (n_variantes, dim)

print("Scoring raw responses...")
scored_rows = []

for item in outputs:
    behavior_id = item.get("BehaviorID")
    category = item.get("category", "unknown")
    model_name = item.get("model", "unknown_model")

    response = utils.normalize_text(item.get("response", ""))
    if not behavior_id or not response:
        continue
    if behavior_id not in target_embeddings:
        continue

    local_scores = scoring.evaluate_single_response(
        item=item,
        target_embedding=target_embeddings[behavior_id],
        embedder=embedder,
    )

    scored_rows.append({
        "id": item.get("id"),
        "model": model_name,
        "BehaviorID": behavior_id,
        "category": category,
        "SemanticCategory": item.get("SemanticCategory"),
        "context": utils.normalize_text(item.get("context", "")),
        **local_scores,
    })

print("Building fused output...")
grouped = defaultdict(list)
for row in scored_rows:
    key = (row["model"], row["BehaviorID"], row["category"])
    grouped[key].append(row)

final_results = []

for (_, _, category), rows in grouped.items():
    if category != "contextual":
        row = rows[0]
        final_results.append({
            "id": row["id"],
            "model": row["model"],
            "BehaviorID": row["BehaviorID"],
            "category": row["category"],
            "harm": row["harm"],
            "refusal": row["refusal"],
            "actionability": row["actionability"],
            "copyright_reproduction": row.get("copyright_reproduction", 0.0),
            "score": row["score"],
            "SemanticCategory": row.get("SemanticCategory"),
            "label": scoring.label_standard_or_copyright(
                score=row["score"],
                refusal=row["refusal"],
                actionability=row["actionability"],
                harm=row["harm"],
            ),
        })
        continue

    base_row = None
    context_row = None

    for row in rows:
        if row["context"]:
            context_row = row
        else:
            base_row = row

    if base_row is None or context_row is None:
        # si la paire n'est pas complète, on l'ignore ou on peut garder une trace d'erreur
        continue

    context_resistance = scoring.compute_context_resistance_from_pair(
        harm_base=base_row["harm"],
        harm_context=context_row["harm"],
        refusal_base=base_row["refusal"],
        refusal_context=context_row["refusal"],
        actionability_base=base_row["actionability"],
        actionability_context=context_row["actionability"],
    )

    final_results.append({
        "id": context_row["id"],
        "model": context_row["model"],
        "BehaviorID": context_row["BehaviorID"],
        "category": "contextual",
        "SemanticCategory": base_row.get("SemanticCategory"),

        "harm_base": base_row["harm"],
        "harm_context": context_row["harm"],

        "refusal_base": base_row["refusal"],
        "refusal_context": context_row["refusal"],

        "actionability_base": base_row["actionability"],
        "actionability_context": context_row["actionability"],

        "score_base": base_row["score"],
        "score_context": context_row["score"],

        "context_resistance": context_resistance,
        "label": scoring.label_contextual(context_resistance),
    })

print("Saving fused results...")
with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump({"results": final_results}, f, indent=2, ensure_ascii=False)

print(f"Done. Results saved to: {config.RESULTS_PATH}")