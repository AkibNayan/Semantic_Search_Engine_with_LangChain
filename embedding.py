from langchain_huggingface import HuggingFaceEmbeddings
from document import documents

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_1 = embeddings.embed_query(documents[0].page_content)
vector_2 = embeddings.embed_query(documents[1].page_content)

assert len(vector_1) == len(vector_2)

print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
