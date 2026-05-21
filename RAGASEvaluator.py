from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os


class RAGASEvaluator:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def evaluate_rag(
        self,
        question,
        answer,
        contexts,
        ground_truth
    ):

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        })

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=self.llm,
            embeddings=self.embeddings
        )

        return result