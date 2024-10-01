from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

#from sentence_transformers import SentenceTransformer

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="llama3")
    # ollama.embeddings(
    #     model='mxbai-embed-large',
    #     prompt='Llamas are members of the camelid family',
    # )
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    # hf = HuggingFaceInstructEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs  #replace embeddings
    # )
    return embeddings


# def get_embedding_function():
#     """Get the embedding function using Sentence Transformers."""
#     embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode
#
#     return embeddings