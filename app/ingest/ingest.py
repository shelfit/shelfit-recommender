import json

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from app.models import Settings

settings = Settings() # noqa

VECTOR_SIZE = 384
MYSQL_BATCH_SIZE = 256
EMBEDDING_BATCH_SIZE = 32

VECTOR_FORMAT = "{title} by {author}. {description}. Genres: {genres}"

def format_genres(genres: str|list|None)-> str:
    if genres is None:
        return ""

    genres_str = genres if isinstance(genres, list) else json.loads(genres)
    return ",".join(genres_str)


qdrant_client = QdrantClient(host=settings.qdrant_settings.qdrant_host, port=settings.qdrant_settings.qdrant_port)
collection = settings.qdrant_settings.qdrant_collection_name

if qdrant_client.collection_exists(collection):
    qdrant_client.delete_collection(collection)

qdrant_client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

vector_data = []

mysql_engine = create_engine(settings.gateway_mysql_url)
with mysql_engine.connect() as conn:
    with conn.execution_options(stream_results=True, yield_per=MYSQL_BATCH_SIZE).execute(
        text("select id, title, author, genres, description from books where visibility='public'")
    ) as result:
        for row in result:
            vector_data.append({
                "vector_str": VECTOR_FORMAT.format(title=row.title, author=row.author, description=row.description, genres=format_genres(row.genres)),
                "payload": {
                    "id": row.id,
                    "title": row.title,
                    "author": row.author,
                    "description": row.description,
                    "genres": format_genres(row.genres),
                }
            })


embedding_model = SentenceTransformer(settings.qdrant_settings.embedding_model)
embeddings = embedding_model.encode(
    [item["vector_str"] for item in vector_data],
    batch_size=EMBEDDING_BATCH_SIZE,
    show_progress_bar=True
)

points = [
    PointStruct(
        id=item["payload"]["id"],
        vector=embedding.tolist(),
        payload=item["payload"]
    )
    for item, embedding in zip(vector_data, embeddings)
]

qdrant_client.upload_points(
    collection_name=collection,
    points=points,
    wait=True
)
