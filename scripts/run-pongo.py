from datasets import load_dataset
import pinecone
import csv
from openai import OpenAI
import openai
from llama_index.embeddings import OpenAIEmbedding
import time
import cohere
import requests
from datasets import load_dataset
import math
import json
miracl = load_dataset('miracl/miracl', 'en', use_auth_token=True)

cohere_api_key = ''
oai_client = OpenAI(api_key='')

co = cohere.Client(cohere_api_key)

openai.api_key = ''

pinecone_api_key = ''
pinecone_environment = 'us-east1-gcp'
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = 'pongo-miracl'

pinecone_index = pc.Index(index_name)



file_exists = False
try:
    with open('./miracl-pongo.csv', 'r') as file:
        file_exists = True
except FileNotFoundError:
    pass

if not file_exists:
    with open('./miracl-pongo.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", 'relevant_passages', 'pongo_answer', 'pongo_mrr', 'pogno_DCG10', 'iDCG10'])


vectorized_questions = []
starting_datapoint_index = 0
datapoint_index = 0 

# try:
with open('./miracl-pongo.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    for datapoint in miracl['dev']:

        if datapoint_index%10==0:
            print('datapoint index: '+str(datapoint_index))
        if(datapoint_index < starting_datapoint_index):
            datapoint_index+=1
            continue

        question = datapoint['query']

        vector = oai_client.embeddings.create(input=question, model="text-embedding-3-large", dimensions=3072).data[0].embedding
        docs=pinecone_index.query(vector=vector, include_metadata=True, top_k=100)
        context_string = ''
        source_index = 1


        docs_for_pongo = []

        for doc in docs['matches']:
            j_doc = json.loads(doc['metadata']['_node_content'])

            docs_for_pongo.append({
                'text': j_doc['text'], 
                'original_text': j_doc['text'],
                'id': j_doc['docid'],
                'metadata': {'Title': j_doc['metadata']['title']}
                })



        pongo_body = {
        "query": question,
        "vec_sample_size": 25,
        "plaintext_sample_size": 5,
        "num_results": 10,
        "public_metadata_field": "metadata",
        "text_field": "text",
        "key_field": "id",
        "docs": docs_for_pongo}


        pongo_response = requests.post('https://api.joinpongo.com/api/v1/rerank', headers={'secret': ''}, json=pongo_body).json()


        pongo_doc_relevances = [] #-1 means irrelevant, all else is the index of the relevant source
        relevant_source_count = len(datapoint['positive_passages'])
        sources_string = ''
        i  = 1

        pongo_mrr = -1
        pongo_string= ''
        for doc in pongo_response:
            curr_str  = f'Source #{i}: {doc["metadata"]["Title"]}\n\n{doc["text"]}'
            found = False
            for passage in datapoint['positive_passages']:
                if doc['text'] == passage['text']:
                    if pongo_mrr == -1:
                        pongo_mrr = i
                    found = True
                    curr_str = '(relevant) ' + curr_str
                    break

            if found:
                pongo_doc_relevances.append(1)
            else:
                pongo_doc_relevances.append(0)
            pongo_string += curr_str  + '\n\n'
            i+=1
        
        pongo_DCG10= 0
        iDCG10 = 0

        #count irrelevant sources and DCG@10
        i = 1
        for relevance in pongo_doc_relevances:
            pongo_DCG10 += relevance/math.log2(i+1)
            i+=1
        


        #calculate ideal score
        i = 1
        for relevant in datapoint['positive_passages']:
            iDCG10 += 1/math.log2(i+1)
            i+=1

        relevant_passages_string = ''
        i = 1
        for doc in datapoint['positive_passages']:
            relevant_passages_string += f'{i}. {doc["title"]}:\n\n{doc["text"]}\n\n\n'

        writer.writerow([question, relevant_passages_string, pongo_string, pongo_mrr, pongo_DCG10, iDCG10])
        time.sleep(0.1)

        
        datapoint_index+=1
