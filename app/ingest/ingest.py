import json

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from app.core.config import Settings
from app.utils.string_formatter_util import StringFormatterUtil

settings = Settings() # noqa

VECTOR_SIZE = 384
MYSQL_BATCH_SIZE = 256
EMBEDDING_BATCH_SIZE = 32

VECTOR_FORMAT = "{title} by {author}. {description}. Genres: {genres}"

def genres_to_list(genres: str|list|None)-> list[str]:
    if genres is None:
        return []

    return genres if isinstance(genres, list) else json.loads(genres)

def format_genres_string(genres: list[str])-> str:
    return ", ".join(genres)

def normalize_genres(genres: list[str])-> list[str]:
    return [StringFormatterUtil.normalize_term(genre) for genre in genres]

def main():
    qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection = settings.qdrant_collection_name

    if qdrant_client.collection_exists(collection):
        qdrant_client.delete_collection(collection)

    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    qdrant_client.create_payload_index(
        collection_name=collection,
        field_name="title_normalized",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    qdrant_client.create_payload_index(
        collection_name=collection,
        field_name="author_normalized",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    qdrant_client.create_payload_index(
        collection_name=collection,
        field_name="genres_normalized",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    vector_data = []

    mysql_engine = create_engine(settings.gateway_mysql_url)
    with mysql_engine.connect() as conn:
        with conn.execution_options(stream_results=True, yield_per=MYSQL_BATCH_SIZE).execute(
            text("select id, title, author, genres, description, rating, num_ratings from books where visibility='public'")
        ) as result:
            for row in result:
                genres_list = genres_to_list(row.genres)

                vector_data.append({
                    "vector_str": VECTOR_FORMAT.format(title=row.title, author=row.author, description=row.description or "", genres=format_genres_string(genres_list)),
                    "payload": {
                        "id": row.id,
                        "title": row.title,
                        "author": row.author,
                        "genres": genres_list,
                        "title_normalized": StringFormatterUtil.normalize_term(row.title),
                        "author_normalized": StringFormatterUtil.normalize_term(row.author),
                        "genres_normalized": normalize_genres(genres_list),
                        "description": row.description or "",
                        "rating": float(row.rating) if row.rating is not None else None,
                        "num_ratings": int(row.num_ratings) if row.num_ratings is not None else None
                    }
                })

    # embedding the entire table at once instead of in chunks to avoid long embed times timing out the db connection
    embedding_model = SentenceTransformer(settings.embedding_model)
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

if __name__ == "__main__":
    main()
