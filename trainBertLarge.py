import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

def train_bertopic_large_scale(
    df: pl.DataFrame,
    text_col: str = "text",
    train_size: int = 1_000_000,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    save_path: str = "./bertopic_5m_model"
):
    """
    Trainiert ein BERTopic Modell auf einem großen Datensatz (z.B. 5M Dokumente),
    indem das rechenintensive Clustering auf einem Subset (z.B. 1M) durchgeführt 
    und der Rest via Inference zugeordnet wird.
    """
    
    # 1. Daten vorbereiten (Splitten in Train und Inference)
    print("Schritt 1: Extrahiere Texte aus Polars...")
    docs = df[text_col].to_list()
    
    train_docs = docs[:train_size]
    infer_docs = docs[train_size:]
    
    # 2. Embeddings berechnen (Batchweise auf der GPU)
    print(f"Schritt 2: Berechne Embeddings für das {train_size}-Training-Subset...")
    embedding_model = SentenceTransformer(model_name)
    # batch_size je nach VRAM der GPU anpassen (512 oder 1024 bei großen GPUs)
    train_embeddings = embedding_model.encode(train_docs, show_progress_bar=True, batch_size=512)
    
    # 3. cuML Modelle initialisieren (GPU-beschleunigt)
    print("Schritt 3: Initialisiere cuML UMAP und HDBSCAN...")
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    
    # WICHTIG: prediction_data=True ist zwingend erforderlich, 
    # damit wir später .transform() für die restlichen 4M und 95M Dokumente nutzen können!
    hdbscan_model = HDBSCAN(min_samples=50, min_cluster_size=500, prediction_data=True)
    
    # 4. BERTopic Modell definieren
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="multilingual",
        calculate_probabilities=False, # Spart massiv RAM und Laufzeit
        verbose=True
    )
    
    # 5. Training auf dem 1M Subset
    print(f"Schritt 4: Trainiere BERTopic auf {train_size} Dokumenten...")
    train_topics, _ = topic_model.fit_transform(train_docs, train_embeddings)
    
    # 6. Inference auf den restlichen 4M Dokumenten
    print(f"Schritt 5: Ordne die restlichen {len(infer_docs)} Dokumente zu...")
    infer_embeddings = embedding_model.encode(infer_docs, show_progress_bar=True, batch_size=512)
    infer_topics, _ = topic_model.transform(infer_docs, infer_embeddings)
    
    # 7. Daten zusammenführen
    print("Schritt 6: Führe alle 5 Millionen Zuweisungen zusammen...")
    all_docs = train_docs + infer_docs
    all_topics = train_topics + infer_topics
    # np.vstack ist speichereffizienter als Listen-Konkatenation bei Arrays
    all_embeddings = np.vstack((train_embeddings, infer_embeddings)) 
    
    # 8. Outlier Reduction (Topic -1 reduzieren)
    print("Schritt 7: Reduziere Outliers (Topic -1)...")
    # Nutze die Embeddings-Strategie, um Ausreißer in die semantisch nächstgelegenen Topics zu verschieben
    new_topics = topic_model.reduce_outliers(
        all_docs, 
        all_topics, 
        strategy="embeddings", 
        embeddings=all_embeddings
    )
    
    # 9. Themenrepräsentationen auf Basis aller 5M Dokumente aktualisieren
    print("Schritt 8: Aktualisiere c-TF-IDF Repräsentationen für alle 5M Dokumente...")
    # Hier lernt das Modell die finalen Wörter für die Topics basierend auf dem vollen 5M Datensatz
    topic_model.update_topics(all_docs, topics=new_topics)
    
    # 10. Modell speichern
    print(f"Schritt 9: Speichere Modell unter {save_path}...")
    # Wir nutzen Pickle, da cuML-Objekte für die spätere 95M-Inference erhalten bleiben müssen
    topic_model.save(save_path, serialization="pickle")
    
    print("Training abgeschlossen!")
    return topic_model, new_topics

# === Aufruf-Beispiel ===
# topic_model, final_topics = train_bertopic_large_scale(
#     df=df_train_polars, 
#     text_col="text_inhalt", 
#     train_size=1_000_000
# )
