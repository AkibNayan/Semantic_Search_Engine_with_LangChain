from embedding import embeddings
from langchain_core.vectorstores import InMemoryVectorStore
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

vector_store = InMemoryVectorStore(embeddings)


def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]


file_path = "E:/Semantic_Search_Engine_with_LangChain/nke-10k-2023.pdf"
docs = load_pdf_pages(file_path)
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

# We can now index the chunks into the vector store.
ids = vector_store.add_documents(documents=all_splits)
print(ids)

# Return documents based on similarity to a string query:
results = vector_store.similarity_search(
    "How many distribution centers does Nike has in the US?"
)
print(results[0])


# Async query
async def main():
    results = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(results[0])


asyncio.run(main())

# Return scores:
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

# Return documents based on similarity to an embedded query:
embed_query = embeddings.embed_query("How were Nike's margin impacted in 2023?")
results = vector_store.similarity_search_by_vector(embed_query)
print(results[0])
