import polars as pl
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

def analyze_outliers_validation(
    inferred_results_path: str,
    original_texts_path: str,
    text_col: str = "text",
    id_col: str = "id_column",
    sample_size: int = 1_000_000
):
    """
    Isoliert die Ausreißer (Topic -1) aus den Inference-Ergebnissen, 
    zieht ein Sample und trainiert ein diagnostisches BERTopic-Modell, 
    um auf übersehene Makro-Themen zu prüfen.
    """
    print("Schritt 1: Lade und filtere Ausreißer (Topic -1) via Polars Lazy API...")
    
    # Lazy Loading liest die Daten nicht sofort in den RAM, sondern erstellt einen Ausführungsplan
    df_inferred = pl.scan_parquet(inferred_results_path)
    df_original = pl.scan_parquet(original_texts_path)
    
    # Wir isolieren direkt nur die Dokumente, die in Topic -1 gelandet sind
    outliers_lazy = df_inferred.filter(pl.col("topic_id") == -1)
    
    # Join mit den Originaldaten, um die echten Texte für diese IDs zu holen
    joined_lazy = outliers_lazy.join(
        df_original.select([id_col, text_col]),
        on=id_col,
        how="inner"
    )
    
    print(f"Schritt 2: Führe Join aus und ziehe ein Sample von {sample_size} Dokumenten...")
    
    # .collect(streaming=True) verarbeitet den Join in Batches (Out-of-Core),
    # was bei 100M Reihen das Überlaufen des Arbeitsspeichers verhindert.
    try:
        df_outliers = joined_lazy.collect(streaming=True)
    except Exception as e:
        print(f"Streaming fehlgeschlagen, wechsle zu Standard-Collect: {e}")
        df_outliers = joined_lazy.collect()
        
    # Begrenze das Sample auf die angegebene Größe, falls es mehr Ausreißer gibt
    if df_outliers.height > sample_size:
        df_outliers = df_outliers.sample(n=sample_size, with_replacement=False, seed=42)
        
    outlier_docs = df_outliers[text_col].to_list()
    print(f"-> {len(outlier_docs)} Ausreißer-Dokumente erfolgreich geladen.")
    
    print("Schritt 3: Trainiere diagnostisches BERTopic-Modell auf den Ausreißern...")
    # Wir nutzen wieder die GPU für maximale Geschwindigkeit
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedding_model.encode(outlier_docs, show_progress_bar=True, batch_size=512)
    
    # WICHTIG: Wir suchen hier NICHT nach Nischen, sondern nach massiven Lücken.
    # Daher setzen wir min_cluster_size sehr hoch (z. B. 1000 oder 2000). 
    # Wenn ein Thema kleiner ist, ist es legitimes "Rauschen".
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_samples=50, min_cluster_size=1000)
    
    diagnostic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="multilingual",
        verbose=True
    )
    
    # Fit auf das Outlier-Sample
    topics, _ = diagnostic_model.fit_transform(outlier_docs, embeddings)
    
    print("\n=== Analyse-Ergebnis (Top 10 gefundene Cluster im Rauschen) ===")
    topic_info = diagnostic_model.get_topic_info()
    print(topic_info.head(10))
    
    return diagnostic_model, topic_info

# === Aufruf-Beispiel ===
# diag_model, diag_info = analyze_outliers_validation(
#     inferred_results_path="./95m_docs_mit_topics.parquet",
#     original_texts_path="hdfs://data/100m_docs.parquet",
#     sample_size=1_000_000
# )
