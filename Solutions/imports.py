import requests
import json
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from pymilvus import MilvusClient, AnnSearchRequest

def get_stream(url, data):
    session = requests.Session()

    with session.post(url, data=json.dumps(data), stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                token = json.loads(line)["response"]
                print(token, end='')

def query_llm(query, server='ollama:11434', model='llama3', prompt=None):
    if not query:
        query = "I don't know what to ask."
    if not prompt:
        prompt = f"""You are yoda. Respond to the prompt below:

prompt: {query}
"""
    data = {"model":model, "prompt": prompt, "stream":True}
    url = f'http://{server}/api/generate'
    session = requests.Session()
    with session.post(url, data=json.dumps(data), stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                token = json.loads(line)["response"]
                print(token, end='')

class RAG:

    def __init__(self, 
                 server='milvus-standalone:19530',
                 database='RAG_Default',
                 collection='Default_Collection',
                 recreate_collection=False,
                 chunk_size=100,
                 chunk_overlap=25,
                 embeddings_model='sentence-transformers/multi-qa-distilbert-cos-v1',
                 embeddings_dimensions = 768, 
                 llm_server = 'ollama:11434',
                 llm_name = 'llama3'
                ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection = collection
        self.llm_server = llm_server
        self.llm_name = llm_name
        
        try:
            self.embeddings_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
        except Exception as e:
            print(f'Could not initialize embeddings model: {e}')
            
        try:
            self.database = MilvusClient(f"http://{server}")
        except Exception as e:
            print(f'Problem connecting to Milvus server: {e}')
        if database in self.database.list_databases():
            print(f"Connecting to {database}")
            self.database.using_database(database)
        else:
            print(f'Creating {database}')
            self.database.create_database(database)
            self.database.using_database(database)
        if recreate_collection:
            self.database.drop_collection(collection)
        if not self.database.has_collection(collection):
            self.database.create_collection(collection_name = self.collection,
                                            dimension = embeddings_dimensions,
                                            auto_id = True)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )


    def store_embeddings(self, document, document_name, rights):
        print(f'Ingesting {document_name}... Please be patient.')
        for page_num, page in enumerate(document.pages):
            if (page_num + 1) % 10 == 0:
                print(f'Page {page_num+1}')
            data = [
                {
                    "vector": vector, 
                    "text":text, 
                    "rights":rights,
                    "page":page_num + 1, 
                    "publication":document_name
                } 
                for text,vector in self.get_embeddings(page)
            ]
            self.database.insert(collection_name=self.collection, data=data)
            
    def get_text(self, page, lines_to_skip = 4):
        """
        Here's the logic of the one liner below:
            Extract the text (page.extract_text())
            Split the result on newlines (.split('\n'))
            Ignore the element at position 0 ([1:])
            Join that list with newlines to create a single string ('\n'.join())
                Note that we are preserving all of the original newlines since they
                should tell us where paragraphs are. Semantically, we expect
                all of the sentences in a paragraph to be somewhat related
                and a new paragraph to indicate a change in thought.
        """
        return '\n'.join(page.extract_text().split('\n')[lines_to_skip:])
    
    def get_chunks(self, page):
        return self.splitter.split_text(self.get_text(page))
    
    def get_embeddings(self, page):
        results = []
        chunks = self.get_chunks(page)
        for chunk in chunks:
            results.append((chunk, self.embeddings_model.encode(chunk)))
        return results
    
    def get_stream(self, url, data):
        session = requests.Session()
    
        with session.post(url, data=data, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    token = json.loads(line)["response"]
                    print(token, end='')
    
    def query(self, question, num_results = 5, include_attributions=False, rights=0, debug=False):

        
        result = self.database.search(collection_name=self.collection, 
                               data=[self.embeddings_model.encode(question)],
                               filter=f'{rights} >= rights',
                               limit=num_results, 
                               output_fields=['text', 'publication', 'page', 'rights'])
        chunks = [i['entity']['text']  for i in result[0] if i['entity']['rights'] & rights]
        references = [(i['entity']['publication'], i['entity']['page']) for i in result[0] if i['entity']['rights'] & rights]
        chunks = '\n'.join(chunks)
        if debug:
            print('-------------------------')
            print(f' Input:    {question}')
            print(f' Rights:   {rights}')
            print(f' Limit:    {num_results}')
            print(f' Attrs:    {include_attributions}')
            print(f' Chunks:')
            print(chunks)
            print(f' References:')
            print(references)
            print('-------------------------')
        prompt = f"""
            Answer the following question using only the datasource provided. Be concise. Do not guess. 
            If you cannot answer the question from the datasource, tell the user the information they want is not
            in your dataset. Refer to the datasource as 'my sources' any time you might use the word 'datasource'.
    
            question: <{question}>
    
            datasource: <{chunks}>
            """
        data = {"model":self.llm_name, "prompt": prompt, "stream":True}
        url = f'http://{self.llm_server}/api/generate'
        self.get_stream(url, json.dumps(data))
        if include_attributions:
            print('\n\n-----------------------\nThis response is based on material found in:\n')
            refs = {}
            for publication, page in references:
                if refs.get(publication):
                    refs[publication].add(page)
                else:
                    refs[publication] = {page}
            for pub, pages in refs.items():
                print(f'{pub} page(s) ', end='')
                print(*sorted(pages), sep=', ')



class ContextualRAG(RAG):
    def contextual_query(self, question, num_results = 2, include_attributions=False, rights=0, debug=False):
        # Begin by performing a typical query. Since we know some results might be filtered by the rights,
        # we will not use the configured num_results but use a much larger number to ensure we have results
        # to filter later. We will also exclude the text chunks from our results since we really don't need them
        # and will never use them in this function:
        result = self.database.search(collection_name=self.collection, 
                               data=[self.embeddings_model.encode(question)],
                               filter=f'{rights} >= rights',
                               limit=num_results*5, 
                               output_fields=['publication', 'page', 'rights'])
        # Based on these results, we want the best matches. The results are typically returned from a
        # vector database from greatest similarity to smallest. Let's just take num_results of these after
        # filtering for rights. Let's also use a set here so we know they are unique and don't end up
        # retrieving the same page multiple times.
        refs_for_context = [(i['entity']['publication'], i['entity']['page']) for i in result[0] if i['entity']['rights'] & rights]
        refs_for_context = set(refs_for_context[:num_results])

        # Next we want to retrieve all of the chunks for the matches. We no longer need the rights since we
        # have prefiltered for only documents the user can see:
        results = []
        for publication, page in refs_for_context:
            results = results + self.database.query(collection_name=self.collection,
                           filter = f'page == {page} and publication == "{publication}"', 
                           offset = 0,
                           limit = 500, 
                           output_fields = ['publication', 'page', 'text'])
        # Now we aggregate all of the text:
        text = ''
        for result in results:
            text = f'{text} {result["text"]}'
            
        prompt = f"""
            Answer the following question using only the datasource provided. Do not guess. 
            If you cannot answer the question from the datasource, tell the user the information they want is not
            in your dataset. Refer to the datasource as 'my sources' any time you might use the word 'datasource'.
    
            question: <{question}>
    
            datasource: <{text}>
            """
        data = {"model":self.llm_name, "prompt": prompt, "stream":True}
        url = f'http://{self.llm_server}/api/generate'
        self.get_stream(url, json.dumps(data))
        if include_attributions:
            print('\n\n-----------------------\nThis response is based on material found in:\n')
            refs = {}
            for publication, page in refs_for_context:
                if refs.get(publication):
                    refs[publication].add(page)
                else:
                    refs[publication] = {page}
            for pub, pages in refs.items():
                print(f'{pub} page(s) ', end='')
                print(*sorted(pages), sep=', ')
    