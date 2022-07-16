from django.shortcuts import HttpResponse, render
from mainapp import retrievers
from haystack.nodes import FARMReader
from haystack.utils import print_answers
from haystack.pipelines import ExtractiveQAPipeline
import os
from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.document_stores import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
# Create your views here.


def index(request):
    return render(request, "index.html")


def about(request):
    retriever = retrievers.init_retriever("./docs")

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                        use_gpu=True)
    # document_store = InMemoryDocumentStore()
    # retriever = TfidfRetriever(document_store=document_store)

    if request.method == "POST":

        # get question from the user
        question = request.POST.get("question")

        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(query=question, params={"Retriever": {
            "top_k": 5}, "Reader": {"top_k": 2}})

        answers = print_answers(prediction, details="minimum")
        for ans in answers:
            answer = ans.get('answer')
            bgcontext = ans.get('context')

        context = {
            'answer': answer,
            'bgcontext': bgcontext
        }

        return render(request, "about.html", context)
    return render(request, 'about.html')

# def queryIt(q):
#     retriever = retrievers.init_retriever()
#     reader = FARMReader(model_name_or_path="my_model", use_gpu=True)

#     print("Work in progress........................................")
#     pipe = ExtractiveQAPipeline(reader, retriever)

#     print("Running pipeline.........................................")
#     # The higher top_k_retriever, the better (but also the slower) your answers.
#     # prediction = pipe.run(
#     #     query="What is network layer?", params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
#     # )
#     prediction = pipe.run(
#         query=q, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 2}}
#     )

#     print_answers(prediction, details="minimum")


# if __name__ == '__main__':
#     queryIt("WHat is OSI model?")
