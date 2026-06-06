from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
import numpy as np
from typing import List, Union, Optional

def train_or_update_bertopic(
    docs: List[str], 
    embeddings: Union[np.ndarray, List[List[float]]], 
    existing_model: Optional[BERTopic] = None,
    n_clusters: int = 50
) -> BERTopic:
    """
    Trainiert ein neues BERTopic-Modell für Online-Learning oder aktualisiert ein bestehendes.
    
    Parameter:
    - docs: Liste der Originaltexte.
    - embeddings: Hugging Face Embeddings passend zu den Texten.
    - existing_model: Ein bereits trainiertes BERTopic-Modell (für Updates). Wenn None, wird ein neues erstellt.
    - n_clusters: Ziel-Anzahl der Themen (wichtig für MiniBatchKMeans beim initialen Training).
    
    Rückgabe:
    - Das trainierte oder aktualisierte BERTopic-Modell.
    """
    # Sicherstellen, dass Embeddings als NumPy-Array vorliegen
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
        
    if len(docs) != len(embeddings):
        raise ValueError("Die Anzahl der Dokumente muss exakt mit den Embeddings übereinstimmen.")

    # FALL A: Es gibt noch kein Modell -> Initiales Training
    if existing_model is None:
        print("Erstelle und trainiere ein neues Basis-Modell für Online-Learning...")
        
        # Für Online-Learning müssen wir feste Algorithmen definieren
        cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        dim_model = IncrementalPCA(n_components=5)
        
        topic_model = BERTopic(
            language="german",
            umap_model=dim_model,
            hdbscan_model=cluster_model
        )
        
        # Wichtig: Hier nutzen wir fit(), nicht fit_transform()
        topic_model.fit(docs, embeddings=embeddings)
        print("Basis-Modell erfolgreich trainiert.")
        return topic_model

    # FALL B: Ein Modell wurde übergeben -> Inkrementelles Update
    else:
        print(f"Aktualisiere bestehendes Modell mit {len(docs)} neuen Dokumenten...")
        
        # partial_fit fügt die neuen Daten in das bestehende Modell ein
        existing_model.partial_fit(docs, embeddings=embeddings)
        print("Modell erfolgreich aktualisiert.")
        return existing_model
