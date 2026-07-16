from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain
from vector import vector_store


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
