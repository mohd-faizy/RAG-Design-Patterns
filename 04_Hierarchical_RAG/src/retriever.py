from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"


class HierarchicalRetriever:
    def __init__(self, parent_map):
        self.parent_map = parent_map

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.retriever = (
            vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        )

    def retrieve(self, query):
        # -------------------------
        # RETRIEVE CHILD CHUNKS
        # -------------------------
        child_docs = self.retriever.invoke(query)

        # -------------------------
        # FIND PARENT DOCS
        # -------------------------
        parent_contexts = []
        seen = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")

            if parent_id and parent_id not in seen:
                parent_contexts.append(
                    self.parent_map[parent_id]
                )
                seen.add(parent_id)

        return parent_contexts
