from ddgs import DDGS
from langchain_core.documents import Document


class WebRetriever:
    def retrieve(self, query):
        docs = []

        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=3
            )

            for r in results:
                body = r.get("body", "")
                if body:
                    docs.append(
                        Document(
                            page_content=body
                        )
                    )

        return docs
