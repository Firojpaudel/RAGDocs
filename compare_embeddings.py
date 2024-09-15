from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("under")
    print(f"Vector for 'under': {vector}")
    print(f"Vector length: {len(vector)}")
    
    ## Comparing vector of two words:
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("under", "below")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()