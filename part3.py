import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


CONFIG = {
    "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT"),
    "API_KEY": os.getenv("AZURE_API_KEY"),
    "DEPLOYMENT": "gpt-4-turbo", 
    "VERSION": "2024-02-15-preview"
}

class SupportEngine:
    def __init__(self):
       
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG["AZURE_ENDPOINT"],
            api_key=CONFIG["API_KEY"],
            azure_deployment=CONFIG["DEPLOYMENT"],
            api_version=CONFIG["VERSION"]
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=CONFIG["AZURE_ENDPOINT"],
            api_key=CONFIG["API_KEY"]
        )


        loader = CSVLoader(file_path="customer_support_data.csv")
        docs = loader.load()
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        
  
        system_prompt = (
            "You are a helpful support assistant. Use the context to answer. "
            "If you don't know, say you'll connect them to a human.\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.vector_db.as_retriever(), doc_chain)

    def fetch_crm_record(self, cid: str) -> Dict:
        """Simulates a CRM database lookup."""
        return {
            "uid": cid,
            "user_name": "Amit Sharma",
            "tier": "Gold Member",
            "status": "Active"
        }

    def log_support_ticket(self, description: str) -> str:
        """Simulates ticket creation in Jira/Zendesk."""
        return f"🎫 Support ticket #XP-992 generated for: {description}"

    def process_request(self, user_query: str) -> str:
        """Intelligent routing logic."""
        text = user_query.lower()


        if any(word in text for word in ["ticket", "complaint", "report"]):
            return self.log_support_ticket(user_query)


        if any(word in text for word in ["account", "profile", "subscription"]):
            user_info = self.fetch_crm_record("C-404")
            return f"Retrieved Account Details: {user_info}"

        
        result = self.rag_chain.invoke({"input": user_query})
        return result["answer"]



app = FastAPI(title="Pro-Active Support Bot")
engine = SupportEngine()

class QueryModel(BaseModel):
    prompt: str

@app.get("/health")
def status_check():
    return {"status": "online", "engine": "active"}

@app.post("/v1/chat")
async def handle_chat(payload: QueryModel):
    try:
        output = engine.process_request(payload.prompt)
        return {"input": payload.prompt, "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))