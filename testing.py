from dotenv import load_dotenv
load_dotenv()  # must run before instantiating VectorStore

from medical_graph_rag.nlp.vectorstore import VectorStore

vs = VectorStore()
print(len(vs.vector_index.index_to_docstore_id))
docs = vs.vector_index.similarity_search("How does Covid-19 affect mental health?", k=5)
print([d.page_content[:80] for d in docs])
