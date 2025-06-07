from RAG_project_practice.llm_models.embedding_model import bge_embedding

embeddings = bge_embedding.embed_documents(
    [
        "hi there!",
        "On, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

print(len(embeddings), len(embeddings[0]))