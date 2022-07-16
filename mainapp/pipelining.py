from haystack.utils import print_answers
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import retrievers


def queryIt(q):
    retriever = retrievers.init_retriever()
    reader = FARMReader(model_name_or_path="my_model", use_gpu=True)

    print("Work in progress........................................")
    pipe = ExtractiveQAPipeline(reader, retriever)

    print("Running pipeline.........................................")
    # The higher top_k_retriever, the better (but also the slower) your answers.
    # prediction = pipe.run(
    #     query="What is network layer?", params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    # )
    prediction = pipe.run(
        query=q, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 2}}
    )

    print_answers(prediction, details="minimum")


# if __name__ == '__main__':
#     queryIt("WHat is OSI model?")
