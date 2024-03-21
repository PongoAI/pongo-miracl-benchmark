from datasets import load_dataset
from pinecone import Pinecone
import csv
from openai import OpenAI
import openai
from llama_index.embeddings import OpenAIEmbedding
import time
import cohere
import json
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
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'pongo-miracl'

pinecone_index = pc.Index(index_name)

file_exists = False
try:
    with open('./miracl-cohere.csv', 'r') as file:
        file_exists = True
except FileNotFoundError:
    pass

if not file_exists:
    with open('./miracl-cohere.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", 'relevant_passages', 'cohere_answer', 'cohere_mrr', 'cohere_DCG10', 'iDCG10'])



vectorized_questions = []
starting_datapoint_index = 0
datapoint_index = 0 

# try:
with open('./miracl-cohere.csv', 'a', newline='') as file:
    writer = csv.writer(file)


    for datapoint in miracl['dev']:

        if datapoint_index%10==0:
            print('datapoint index: '+str(datapoint_index))
        if(datapoint_index < starting_datapoint_index):
            datapoint_index+=1
            continue

        question = datapoint['query']

        #llamaindex was misbehaving so do vector search raw
        vector = oai_client.embeddings.create(input=question, model="text-embedding-3-large", dimensions=3072).data[0].embedding
        docs=pinecone_index.query(vector=vector, include_metadata=True, top_k=100)
        context_string = ''
        source_index = 1


        docs_for_cohere = []
        for doc in docs['matches']:
            j_doc = json.loads(doc['metadata']['_node_content'])

            docs_for_cohere.append({
                'text': f"Title: {j_doc['metadata']['title']}\nBody: {j_doc['text']}", 
                'original_text': j_doc['text'],
                'title': j_doc['metadata']['title']
                })

        cohere_docs = co.rerank(query=question, documents=docs_for_cohere, top_n=10, model='rerank-english-v2.0')




        cohere_doc_relevances = [] #-1 means irrelevant, all else is the index of the relevant source
        relevant_source_count = len(datapoint['positive_passages'])
        sources_string = ''
        i  = 1

        cohere_mrr = -1
        cohere_string= ''
        while i < 10:
            doc = docs_for_cohere[cohere_docs[i].index]

            curr_str  = f'Source #{i}:\n {doc["title"]}\n\n{doc["original_text"]}'
            found = False
            for passage in datapoint['positive_passages']:
                if doc['original_text'] == passage['text']:
                    if cohere_mrr == -1:
                        cohere_mrr = i
                    found = True
                    curr_str = '(relevant) ' + curr_str
                    break

            if found:
                cohere_doc_relevances.append(1)
            else:
                cohere_doc_relevances.append(0)
            cohere_string += curr_str  + '\n\n'
            i+=1
        


        cohere_DCG10= 0
        iDCG10 = 0

        #count irrelevant sources and DCG@10
        i = 1
        for relevance in cohere_doc_relevances:
            cohere_DCG10 += relevance/math.log2(i+1)
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

        writer.writerow([question, relevant_passages_string, cohere_string, cohere_mrr, cohere_DCG10, iDCG10])
        time.sleep(0.1)

        
        datapoint_index+=1
