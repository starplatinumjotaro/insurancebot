import os
import glob
import shutil
from openai import OpenAI
from thirdai import neural_db as ndb

db = ndb.NeuralDB()
insertable_docs = []

def load_documents(directory: str):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for file in pdf_files:
        pdf_doc = ndb.PDF(file)
        insertable_docs.append(pdf_doc)
    
    checkpoint_dir = "./data/sample_checkpoint"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=False,
        checkpoint_interval=3,
    )

    # Insert documents into the NeuralDB and create checkpoint
    db.insert(insertable_docs, train=True, checkpoint_config=checkpoint_config)
    print(f"Inserted {len(insertable_docs)} documents into the database.")

def generate_queries_chatgpt(original_query):
    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )
    return response.choices[0].message.content.strip().split("\n")

def get_references(query):
    search_results = db.search(query, top_k=50)
    return search_results

def rag_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, result in enumerate(results):
            doc_str = result.text
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    return [
        (doc_str, score)
        for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

def generate_answers(query, references):
    context = "\n\n".join(references[:3])
    prompt = (
        f"You are a helpful assistant for the indiafirst life insurance company that generates answers based on the user's query from the provided pdf documents only."
        f"Don't take any information from the internet. Only use the provided documents. "
        f"you must try to find the exact answer from the given documents for the user query."
        f"you will never make the user feel irritated by your answer."
        f"if the user doesn't mention any policy name suggest some policies or ask the user for the policy name. "
        f"if the user says hi make a friendly conversation about their policy looking for. "
        f"Provide the most specific answer to the following question based on the documents. "
        f"If the exact answer is found in the text, return only that specific answer. "
        f"you have to specify the plan name for every answer what the plan name is. "
        f"if you gave the answer ask the user for any suggestions. "
        f"If the answer requires additional context, include it:\n"
        f"Question: {query} \nContext: {context}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content
