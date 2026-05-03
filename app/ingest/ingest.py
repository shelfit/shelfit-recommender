import ast
import os.path
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(host='qdrant', port=6333)
collection_name = 'books'

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

base_dir = os.path.dirname(__file__)
file_path = os.path.join(str(base_dir), 'books.csv')

df = pd.read_csv(file_path)
df = df.dropna(subset=['title', 'author', 'description']).reset_index(drop=True)

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

BATCH_SIZE = 32

texts = []
payloads = []
for index, row in df.iterrows():
    title = row['title']
    authors = row['author']
    description = row['description']
    genres = row['genres']

    if isinstance(genres, str):
        genres = ast.literal_eval(genres)
    genres_str = ', '.join(genres)

    texts.append(f"{title} by {authors}. {description}. Genres: {genres_str}.")
    payloads.append({
        'title': title,
        'authors': authors,
        'description': description,
        'genres': genres_str
    })

embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

points = [
    PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
    for i in range(len(texts))
]

client.upload_points(
    collection_name=collection_name,
    wait=True,
    points=points
)
