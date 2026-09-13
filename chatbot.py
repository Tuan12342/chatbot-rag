
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True ,
    use_multithreading=False,

)
docs =loader.load()
MARKDOWN_SEPARATORs = [
    "\n#{1,6}",
    "```\n",
    "\n\\*\\*\\**\n",
    "\n--+\n",
    "\n_ _ _+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORs,
    is_separator_regex=True,

)
splits= text_splitter.split_documents(docs)
load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

retriever=vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":5,"score_threshold":0.2}
)
