# Importing Stuff needed to Import
import os, dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Loading environment Variables
dotenv.load_dotenv()
COHERE_KEY = os.environ["COHERE_API_KEY"]
GOOGLE_KEY = os.environ["GOOGLE_API_KEY"]

# Setting up the LLM, Embedding Model and Vector Store
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=1, api_key = GOOGLE_KEY)
embedding_model = CohereEmbeddings(model="embed-v4.0", cohere_api_key=COHERE_KEY)
file_path = './HarryPotter.txt'
loader = TextLoader(file_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
vectorstore = Chroma.from_documents(
    documents=splitter.split_documents(loader.load()),
    collection_name="ReRankingRAG",
    embedding=embedding_model,
)

# Making the prompt and RAG pipeline
prompt = ChatPromptTemplate.from_template('''
    You are Albus Dumbledore, a very wise and sincere Potterhead who knows everything regarding Harry Potter.
    Your primary task is to answer the user's question **EXCLUSIVELY based on the provided context.**
    Context: {context}
    Question: {question}
    Answer with the wisdom and eloquence befitting Albus Dumbledore, and conclude warmly (don't be formal though. Be casual).
    Sprinkle some Dumbledore knowledge and quotes here and there (don't make it too long though)
'''
)
rag = prompt | llm | StrOutputParser()
os.system('clear')

#Setting up Re-ranking
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs = {"k":15}
)
compressor = CohereRerank(
    cohere_api_key=COHERE_KEY,
    model="rerank-english-v3.0", 
    top_n=5
)
context_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

#Giving Query and Output
query = input("Enter query: ")
context = "\n".join([doc.page_content for doc in context_retriever.invoke(query)])
print("\n",rag.invoke({"context":context, "question": query}),"\n\n")
