from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.document_stores import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline


def init_retriever(doc_file):

    document_store = InMemoryDocumentStore()
    retriever = TfidfRetriever(document_store=document_store)

    doc_dir = doc_file
    docs = convert_files_to_docs(
        dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    document_store.write_documents(docs)
    retriever = TfidfRetriever(document_store=document_store)

    return retriever
    # object
