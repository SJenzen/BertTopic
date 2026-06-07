import polars as pl
import pyarrow as pa
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

def semantic_diversity_sampling_to_polars(
    spark: SparkSession,
    df_100m, # Dein initiales PySpark DataFrame
    text_col: str = "text",
    total_sample_size: int = 5_000_000,
    intermediate_sample_size: int = 20_000_000,
    n_clusters: int = 10_000,
    seed: int = 42
) -> pl.DataFrame:
    """
    Führt ein semantisches Sampling auf einem großen Spark DataFrame durch 
    und gibt ein lokales Polars DataFrame für das BERTopic Training zurück.
    """
    
    print("Schritt 1: Initiales Subset ziehen (20 Mio. Dokumente)...")
    # Um ein .count() auf 100M zu vermeiden (was in Spark dauern kann), 
    # geben wir die Fraction direkt an (20M / 100M = 0.2)
    fraction = intermediate_sample_size / 100_000_000 
    df_20m = df_100m.sample(withReplacement=False, fraction=fraction, seed=seed)
    
    print("Schritt 2: Leichtgewichtige Vektorisierung vorbereiten...")
    # Tokenizer zerlegt den Text in Wörter
    tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
    # HashingTF erstellt Vektoren ohne teures Vokabular-Mapping (2^14 = 16384 Dimensionen)
    hashing_tf = HashingTF(inputCol="words", outputCol="features", numFeatures=16384)
    
    print(f"Schritt 3: K-Means Clustering in {n_clusters} Cluster...")
    # maxIter gering halten, da wir keine perfekten Cluster brauchen, nur grobe semantische Zonen
    kmeans = KMeans(featuresCol="features", predictionCol="cluster_id", k=n_clusters, seed=seed, maxIter=5)
    
    # ML-Pipeline ausführen
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, kmeans])
    model = pipeline.fit(df_20m)
    df_clustered = model.transform(df_20m)
    
    print("Schritt 4: Stratifiziertes Sampling aus den Clustern...")
    # Wie viele Dokumente müssen wir pro Cluster ziehen?
    docs_per_cluster = max(1, total_sample_size // n_clusters)
    
    # Window-Funktion: Gruppiere nach Cluster-ID und sortiere innerhalb des Clusters zufällig
    window_spec = Window.partitionBy("cluster_id").orderBy(rand(seed))
    
    df_sampled = (
        df_clustered
        .withColumn("row_num", row_number().over(window_spec))
        .filter(col("row_num") <= docs_per_cluster)
        .select("id_column", text_col, "cluster_id") # Passe "id_column" an deine ID an
    )
    
    print("Schritt 5: Transfer von Spark zu Polars via PyArrow...")
    # PyArrow für maximalen Speed und RAM-Effizienz aktivieren
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # .toPandas() nutzt intern Arrow, wenn oben aktiviert. 
    # Polars kann das direkt und extrem schnell einlesen.
    pandas_df = df_sampled.toPandas()
    polars_df = pl.from_pandas(pandas_df)
    
    print(f"Fertig! Ziel-Dataset hat {polars_df.height} Zeilen.")
    return polars_df

# === Aufruf-Beispiel ===
# spark = SparkSession.builder \
#     .appName("SemanticSampling") \
#     .config("spark.driver.memory", "16g") \
#     .getOrCreate()
#
# df_raw = spark.read.parquet("hdfs://dein/pfad/zu/den/100m_docs.parquet")
# df_train_polars = semantic_diversity_sampling_to_polars(spark, df_raw, text_col="text_inhalt")
