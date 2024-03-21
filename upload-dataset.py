from llama_index.vector_stores import PineconeVectorStore
from llama_index.schema import TextNode
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings import OpenAIEmbedding
from datasets import load_dataset




miracl = load_dataset('miracl/miracl', 'en', use_auth_token=True)

embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key='', dimensions=3072)

pinecone_api_key = ''
pinecone_environment = 'us-east1-gcp'
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'pongo-miracl'
pc.create_index(index_name, dimension=3072, metric="euclidean", spec=ServerlessSpec(
    cloud="aws",
    region="us-west-2"
  ) )


pinecone_index = pc.Index(index_name)



vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

nodes = []

for data in miracl['dev']:  # or 'dev', 'testA'
    query_id = data['query_id']
    passages = []
    passages.extend(data['positive_passages'])
    passages.extend(data['negative_passages'])

    for passage in passages: # OR 'negative_passages'
        docid = passage['docid']
        title = passage['title']
        data = passage['text']



        cur_node = TextNode(text=data)
        cur_node.metadata['docid'] = docid
        cur_node.metadata['title'] = title

        nodes.append(cur_node)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

vector_store.add(nodes)