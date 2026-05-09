"""Handles Performance evaluation"""
import asyncio
import time
import itertools
from pathlib import Path
import pandas as pd
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from ..core.utils.config import benchmark_const, google_genai_config
from ..core.utils.helpers import timeout, export_to_csv

# load required variables
CONSTANTS = benchmark_const()
GOOGLE_VARIABLES = google_genai_config()

# benchmark constants
FILENAME = CONSTANTS['filename'] + '.csv'
PATH = Path(CONSTANTS['path'])
FILE = PATH / FILENAME

# data for export
SAVE_PATH = Path.home() / "data/processed"
SAVE_NAME = "evaluation_result"

# google variables
google_api_key = GOOGLE_VARIABLES['api key']


# load the data in chunks
def load_file(batch_size: int = 2):
    """Serves data in batches to conserve memory usage"""
    data_iter = pd.read_csv(FILE, chunksize=batch_size)
    for chunk in data_iter:
        yield chunk
        # rate limiting: only invoke twice per minute
        time.sleep(30)


# model to be invoked for evaluation
@timeout(30)
@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(5))
def load_judge(model: str = "gemini-2.5-pro"):
    """Loads model from Google for evaluation"""
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=google_api_key
    )


# eval function to carry out evaluation
def google_evaluator():
    """carries out evaluations"""
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=load_judge()
        )


# evalutaion pipeline to be activated on push
def evaluate_correctness():
    """Evaluate correctness of responses using LLM as judge"""
    evaluator = google_evaluator()
    for batch in load_file():
        for _, row in batch.iterrows():
            # Example: evaluate each row’s response
            query = row["query"]
            response = row["answer"]
            reference = row["reference"]

            # run correctness eval
            eval_result = evaluator(
                inputs=query,
                outputs=response,
                reference_outputs=reference,
            )


            result= [query, response, reference, eval_result['score'].map({"True":1, "False":0})]
            export_to_csv(result, SAVE_PATH, SAVE_NAME, evaluation=True)


if __name__ == "__main__":
    evaluate_correctness()