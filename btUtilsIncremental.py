from sentence_transformers import SentenceTransformer
from btUtils import train_or_update_bertopic

# Modell laden
print("Lade Hugging Face Modell...")
hf_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# TAG 1: INITIALES TRAINING (Historischer Korpus)
# ==========================================
docs_batch_1 = [
    "Künstliche Intelligenz revolutioniert den Arbeitsmarkt.",
    "Der DAX startet mit Verlusten in die neue Woche.",
    "Machine Learning Algorithmen werden immer effizienter.",
    "Die Inflation im Euroraum sinkt unerwartet deutlich."
]
# Embeddings für Tag 1 berechnen
embeddings_batch_1 = hf_model.encode(docs_batch_1)

# Funktion ohne "existing_model" aufrufen -> Erstellt neues Modell
# Wir setzen n_clusters hier künstlich niedrig auf 2 für dieses winzige Beispiel
mein_topic_model = train_or_update_bertopic(
    docs=docs_batch_1, 
    embeddings=embeddings_batch_1, 
    n_clusters=2 
)

print("\nThemen an Tag 1:")
print(mein_topic_model.get_topic_info())


# ==========================================
# TAG 14: NEUE DATEN TREFFEN EIN (Update)
# ==========================================
docs_batch_2 = [
    "Neue Regulierung für KI-Systeme in der EU beschlossen.",
    "Aktienmärkte erholen sich nach der Leitzinsentscheidung der EZB."
]
# Embeddings für Tag 14 berechnen
embeddings_batch_2 = hf_model.encode(docs_batch_2)

# Funktion MIT dem bestehenden Modell aufrufen -> Updatet das Modell
mein_topic_model = train_or_update_bertopic(
    docs=docs_batch_2, 
    embeddings=embeddings_batch_2, 
    existing_model=mein_topic_model # Hier übergeben wir das alte Modell!
)

print("\nThemen an Tag 14 (Aktualisiert):")
print(mein_topic_model.get_topic_info())
