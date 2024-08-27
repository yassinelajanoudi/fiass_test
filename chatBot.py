from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import load_config
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import GooglePalmEmbeddings
from prompts import qa_template1
import time
cfg = load_config()
# os.environ['GOOGLE_API_KEY'] = cfg.PALLM_API
# Initialize embeddings and vectorstore
embeddings = GooglePalmEmbeddings(google_api_key=cfg.PALLM_API)
vectorstore = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
# print("vectorestore",vectorstore)
# Define the prompt template
qa_prompt = PromptTemplate(
    template=qa_template1, input_variables=["context", "question"]
)


def build_llm():

    # Local CTransformers model
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=cfg.PALLM_API, temperature=cfg.TEMPERATURE)

    return llm


def build_retrieval_qa(llm, qa_prompt, vectordb):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": qa_prompt},
    )
    return qa_chain


def main():
    llm = build_llm()
    qa_bot = build_retrieval_qa(llm, qa_prompt, vectorstore)

    print("Chatbot is ready. Type your questions below (type 'exit' to quit):")

    while True:
        q_time = time.time()
        question = input(f"{q_time} You: ")
        if question.lower() in ["exit", "quit"]:
            break
        result = qa_bot({"query": question})
        a_time = time.time()
        print(f"{a_time} Bot:", result["result"])
        # print(result["source_documents"])
        for doc in result["source_documents"]:
            source_path = doc.metadata['source'].replace("\\", "/")
            print(f"Source: {source_path.split('/')[1]} (Page {doc.metadata.get('page', 'Unknown')})")



if __name__ == "__main__":
    main()
