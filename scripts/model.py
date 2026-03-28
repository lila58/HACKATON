# Importation des modules standards
import json
from datetime import datetime
import os
import csv

import pandas as pd

# Importation des librairies pour le modele LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


# =========================
# CONFIGURATION
# =========================

# Chemin du fichier d'entree contenant les prompts
INPUT_PATH = "data/experimental_corpus/prompts.jsonl"
# Chemin du fichier de sortie pour les reponses à adapter selon le modèle et le type de test
OUTPUT_PATH = "results/embedding/raw_outputs/google_gemma_7b_it_test_results.jsonl"
# Nom du modele a utiliser
MODEL_NAME = "google/gemma-7b-it"
# Nombre maximum de tokens generes par le modele
MAX_NEW_TOKENS = 150

# =========================
# 1. Charger le modele
# =========================

# Detection du device (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Option pour forcer le CPU
# device = torch.device("cpu")

# Chargement du tokenizer et du modele
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Definition du token de padding si absent
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 2. Fonction de generation
# =========================

# Genere une reponse du modele a partir d'un prompt
def generate_response(prompt: str, max_new_tokens: int = 150) -> str:
    # Les modèles instruct attendent un format de chat.
    # On construit un message simple de type 'user' pour le prompt donné.
    messages = [
        {"role": "user", "content": prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback si apply_chat_template n'est pas disponible. Pour les modèles instruct,
        # ceci pourrait ne pas fonctionner de manière optimale, mais c'est une option de secours.
        formatted_prompt = prompt

    # Tokenisation du prompt formaté
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generation de la reponse
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decodage complet de la sequence generee
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Reponse brute generee :", full_output)

    # Extraction des nouveaux tokens generes
    # On décode à partir de la longueur de l'input pour obtenir uniquement la réponse du modèle.
    generated_tokens_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens_ids, skip_special_tokens=True).strip()

# =========================
# 3. Construction du prompt final
# =========================

# Construit le prompt final a partir d'un dictionnaire item
def build_final_prompt(item):
    # Accepte 'prompt' ou 'question' comme cle
    prompt = str(item.get("prompt") or item.get("Questions") or "").strip()
    context = str(item.get("context", "")).strip() or str(item.get("Context", "")).strip()

    # Ajoute le contexte si present
    if context:
        return f"{context}\n\n{prompt}"
    return prompt


# =========================
# 4. Lecture des prompts (support CSV, JSON, JSONL)
# =========================

def load_data(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    data = []

    def try_open(encoding):
        if ext == ".csv":
            with open(input_path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        elif ext == ".jsonl":
            with open(input_path, "r", encoding=encoding) as f:
                return [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            with open(input_path, "r", encoding=encoding) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return loaded
                else:
                    return [loaded]
        else:
            raise ValueError(f"Format de fichier non supporte : {ext}. Seuls .jsonl, .csv, .json sont acceptes.")

    try:
        data = try_open("utf-8")
    except UnicodeDecodeError:
        print("[AVERTISSEMENT] Fichier non encode en UTF-8, tentative avec 'latin-1'...")
        data = try_open("latin-1")
    return data

# Chargement des donnees depuis le fichier d'entree (auto-format)
data = load_data(INPUT_PATH)

# Affichage du nombre total de prompts
print(f"Nombre total de prompts : {len(data)}")
print(f"Premier prompt : {data[0]}")

# =========================
# 5. Boucle d'inference et sauvegarde
# =========================

# Cree le dossier de sortie si necessaire
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for i, item in enumerate(data):
        # Affichage de la progression
        print(f"[{i+1}/{len(data)}] traitement...")

        # Construction du prompt final
        final_prompt = build_final_prompt(item)

        # Generation de la reponse du modele
        try:
            response = generate_response(final_prompt, max_new_tokens=MAX_NEW_TOKENS)
            error_message = ""
        except Exception as e:
            response = ""
            error_message = str(e)

        # Construction du dictionnaire resultat
        result = {
            "id": item.get("id", i),
            "prompt": item.get("prompt", ""),
            "context": item.get("context", ""),
            "final_prompt": final_prompt,
            "response": response,
            "model": MODEL_NAME,
            "error": error_message
        }

        # Si d'autres colonnes existent, on les garde aussi
        for key, value in item.items():
            if key not in result:
                result[key] = value

        # Sauvegarde du resultat dans le fichier de sortie
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")

# Affichage de la fin du processus
print(f"Resultats sauvegardes dans : {OUTPUT_PATH}")