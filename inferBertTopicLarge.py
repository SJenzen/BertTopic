import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa
import gc
import torch
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def infer_bertopic_large_scale(
    input_parquet_path: str,
    output_parquet_path: str,
    model_path: str = "./bertopic_5m_model",
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    text_col: str = "text",
    id_col: str = "id_column",
    chunk_size: int = 500_000
):
    """
    Lädt ein trainiertes BERTopic-Modell und führt eine speichereffiziente 
    Inference über einen massiven Datensatz (z.B. 95M Dokumente) durch.
    Die Daten werden gestreamt und die Ergebnisse blockweise gespeichert.
    """
    
    print("Schritt 1: Lade Modelle in den Speicher (GPU)...")
    # Das Embedding-Modell separat laden, um die Batch-Size bei der Inference zu kontrollieren
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # BERTopic Modell laden (aus Phase 3)
    topic_model = BERTopic.load(model_path)
    
    print(f"Schritt 2: Öffne Datenstrom für {input_parquet_path}...")
    # pyarrow.parquet erlaubt es, Dateien zu lesen, ohne sie in den RAM zu laden
    parquet_file = pq.ParquetFile(input_parquet_path)
    
    # Prüfen, ob die Ausgabedatei schon existiert (für einen sauberen Append-Start)
    if os.path.exists(output_parquet_path):
        print(f"Warnung: {output_parquet_path} existiert bereits. Wird überschrieben/erweitert.")
        os.remove(output_parquet_path)
        
    chunk_counter = 1
    total_processed = 0
    
    # ParquetFile.iter_batches() ist der Goldstandard für das Streaming massiver Daten
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        print(f"\n--- Verarbeite Chunk {chunk_counter} (Größe: {batch.num_rows} Docs) ---")
        
        # Batch in ein Polars DataFrame umwandeln (Zero-Copy dank Arrow)
        df_chunk = pl.from_arrow(batch)
        docs = df_chunk[text_col].to_list()
        doc_ids = df_chunk[id_col].to_list()
        
        # 1. Embeddings auf der GPU berechnen
        print("   -> Berechne Embeddings...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=512)
        
        # 2. BERTopic Inference (Zuweisung der Topics)
        print("   -> Ordne Topics zu...")
        # Transform liefert die Topics und (falls nicht deaktiviert) die Probabilities
        topics, _ = topic_model.transform(docs, embeddings)
        
        # 3. Ergebnisse in einem neuen Polars DataFrame strukturieren
        print("   -> Speichere Ergebnisse...")
        df_results = pl.DataFrame({
            id_col: doc_ids,
            "topic_id": topics
        })
        
        # 4. Chunk an die Ziel-Parquet-Datei anhängen (Append)
        # pyarrow wird hier genutzt, da Polars das direkte Appenden an Parquets nur über Umwege unterstützt
        table = df_results.to_arrow()
        if chunk_counter == 1:
            # Beim ersten Chunk den Writer initialisieren
            writer = pq.ParquetWriter(output_parquet_path, table.schema)
            writer.write_table(table)
        else:
            # Bei allen weiteren Chunks einfach anhängen
            writer.write_table(table)
            
        # 5. RAM und VRAM (GPU Speicher) rigoros leeren!
        total_processed += batch.num_rows
        del df_chunk, docs, doc_ids, embeddings, topics, df_results, table
        gc.collect()
        torch.cuda.empty_cache() # Zwingt PyTorch, den GPU-Speicher freizugeben
        
        print(f"   -> Chunk {chunk_counter} abgeschlossen. Bisher verarbeitet: {total_processed}")
        chunk_counter += 1

    # Writer sauber schließen
    if 'writer' in locals():
        writer.close()
        
    print(f"\n✅ Inference komplett! {total_processed} Dokumente wurden verarbeitet und unter {output_parquet_path} gespeichert.")

# === Aufruf-Beispiel ===
# infer_bertopic_large_scale(
#     input_parquet_path="hdfs://data/95m_docs_inference.parquet",
#     output_parquet_path="./95m_docs_mit_topics.parquet",
#     chunk_size=500_000 # Reduzieren, falls die GPU out-of-memory geht
# )
