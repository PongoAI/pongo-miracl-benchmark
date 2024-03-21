from datasets import load_dataset
from pinecone import Pinecone
import csv
from openai import OpenAI
import openai
from llama_index.embeddings import OpenAIEmbedding
import time
import json
from datasets import load_dataset
import math


API_URL = "used a private huggingface endpoint for this"



def run_rerank(query, docs):
    
    import asyncio
    import aiohttp

    scores_to_collect = [{'text': query, 'text_pair': doc['text'], 'title': doc['title']} for doc in docs]

    async def get_score(score_to_collect):
        async with aiohttp.ClientSession() as session:
            # Initialize an empty list to hold scores for this text_pair
            scores = []

            # Split the text_pair if it's longer than 1600 characters into segments of up to 1600 characters, add title later
            text_pair = score_to_collect['text_pair']
            segments = [text_pair[i:i+1600] for i in range(0, len(text_pair), 1600)]

            # Process each segment asynchronously
            for segment in segments:
                async with session.post(API_URL, json={'inputs': {'text': query, 'text_pair': score_to_collect['title']+'\n\n'+segment}}) as response:
                    response_json = await response.json()
                    # print(response_json)
                    scores.append(response_json['score'])

            # Return the highest score among the segments
            return max(scores)

    async def gather_scores():
        tasks = [get_score(score_to_collect) for score_to_collect in scores_to_collect]
        return await asyncio.gather(*tasks)

    scores = asyncio.run(gather_scores())

    for i, doc in enumerate(docs):
        doc['score'] = scores[i]
    
    docs.sort(key=lambda x: x['score'], reverse=True)

    return docs[:10]



miracl = load_dataset('miracl/miracl', 'en', use_auth_token=True)

oai_client = OpenAI(api_key='')


openai.api_key = ''

pinecone_api_key = ''
pinecone_environment = 'us-east1-gcp'
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'pongo-miracl'

pinecone_index = pc.Index(index_name)

file_exists = False
try:
    with open('./miracl-bge.csv', 'r') as file:
        file_exists = True
except FileNotFoundError:
    pass

if not file_exists:
    with open('./miracl-bge.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", 'relevant_passages', 'bge_answer', 'bge_mrr', 'bge_DCG10', 'iDCG10'])



vectorized_questions = []
starting_datapoint_index = 0
datapoint_index = 0 

# try:
with open('./miracl-bge.csv', 'a', newline='') as file:
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


        docs_for_bge = []
        for doc in docs['matches']:
            j_doc = json.loads(doc['metadata']['_node_content'])

            docs_for_bge.append({
                'text': f"Title: {j_doc['metadata']['title']}\nBody: {j_doc['text']}", 
                'original_text': j_doc['text'],
                'title': j_doc['metadata']['title']
                })

        bge_docs = run_rerank(question, docs_for_bge)




        bge_doc_relevances = [] #-1 means irrelevant, all else is the index of the relevant source
        relevant_source_count = len(datapoint['positive_passages'])
        sources_string = ''
        i  = 1

        bge_mrr = -1
        bge_string= ''
        for doc in bge_docs:
            curr_str  = f'Source #{i}:\n {doc["title"]}\n\n{doc["original_text"]}'
            found = False
            for passage in datapoint['positive_passages']:
                if doc['original_text'] == passage['text']:
                    if bge_mrr == -1:
                        bge_mrr = i
                    found = True
                    curr_str = '(relevant) ' + curr_str
                    break

            if found:
                bge_doc_relevances.append(1)
            else:
                bge_doc_relevances.append(0)
            bge_string += curr_str  + '\n\n'
            i+=1


        bge_DCG10= 0
        iDCG10 = 0

        #count irrelevant sources and DCG@10
        i = 1
        for relevance in bge_doc_relevances:
            bge_DCG10 += relevance/math.log2(i+1)
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

        writer.writerow([question, relevant_passages_string, bge_string, bge_mrr, bge_DCG10, iDCG10])
        time.sleep(0.1)

        
        datapoint_index+=1
