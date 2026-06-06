import polars as pl
import numpy as np
from bertopic import BERTopic

def stream_spark_data_to_bertopic(parquet_dir_path: str, topic_model: BERTopic, micro_batch_size: int = 50000):
    """
    Streamt Daten aus einem von Spark geschriebenen Parquet-Verzeichnis mittels Polars
    in ein inkrementelles BERTopic-Modell.
    
    Parameter:
    - parquet_dir_path: Pfad zum Verzeichnis mit den Spark-Parquet-Partitionen.
    - topic_model: Ein bereits initialisiertes BERTopic-Modell (bereit für partial_fit).
    - micro_batch_size: Größe der Häppchen, die an BERTopic übergeben werden.
    """
    print("Nutze Polars LazyFrame, um das Spark-Verzeichnis zu scannen...")
    # scan_parquet liest die Daten NICHT sofort in den Speicher, sondern erstellt einen Ausführungsplan
    lazy_df = pl.scan_parquet(f"{parquet_dir_path}/*.parquet")
    
    # Wir ermitteln die Gesamtanzahl der Zeilen hocheffizient über die Metadaten
    total_rows = lazy_df.select(pl.len()).collect().item()
    print(f"Gesamtanzahl zu verarbeitender Zeilen: {total_rows:,}")
    
    # Schleife über die großen Chunks (z.B. in Schritten von 5 Millionen Zeilen)
    # Oder direkt in Mikro-Batches aufgeteilt, um Polars die Speicherverwaltung zu überlassen
    offset = 0
    while offset < total_rows:
        print(f"\nHole nächsten Datenblock ab Zeile {offset:,}...")
        
        # Polars zieht sich hocheffizient genau den benötigten Bereich (Slice)
        # Erst durch .collect() werden die Daten tatsächlich in den RAM geladen
        chunk_df = (
            lazy_df
            .slice(offset, micro_batch_size)
            .select(["text_column", "embedding_column"])
            .collect()
        )
        
        if chunk_df.is_empty():
            break
            
        # Konvertierung in Python-Listen und NumPy-Arrays für BERTopic
        texte = chunk_df["text_column"].to_list()
        
        # Annahme: Embeddings liegen in Spark/Polars als Liste von Floats in einer Spalte vor
        # pl.Series.to_numpy() konvertiert das hocheffizient in ein 2D NumPy-Array
        embeddings = np.vstack(chunk_df["embedding_column"].to_numpy())
        
        # Inkrementelles Training
        print(f"Trainiere BERTopic mit Mikro-Batch von {len(texte):,} Dokumenten...")
        topic_model.partial_fit(texte, embeddings=embeddings)
        
        offset += micro_batch_size
        
    print("\nStreaming-Pipeline erfolgreich beendet!")
    return topic_model
