import streamlit as st
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

GENERIC_FALLBACK_MESSAGE = "I'm not sure about that. Please contact our support team for more details."
SIMILARITY_THRESHOLD = 0.5 

load_dotenv()

faq_entries = [
    ("What does the eligibility verification agent (EVA) do?",
     "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."),

    ("What does the claims processing agent (CAM) do?",
     "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."),

    ("How does the payment posting agent (PHIL) work?",
     "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."),

    ("Tell me about Thoughtful AI's Agents.",
     "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."),

    ("What are the benefits of using Thoughtful AI's agents?",
     "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting.")
]

docs = [
    Document(page_content=f"{question}\n{answer}")
    for question, answer in faq_entries
]


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)


retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    retriever=retriever
)

st.title("Thoughtful AI - Support Agent (MINI-RAG)")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask me something about Thoughtful AI:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    query_embedding = embeddings.embed_query(user_input)
    docs_and_scores = vectorstore.similarity_search_with_score_by_vector(query_embedding, k=1)

    if docs_and_scores[0][1] > (1 - SIMILARITY_THRESHOLD):
        response = GENERIC_FALLBACK_MESSAGE
    else:
        response = qa_chain.invoke(user_input)["result"]

    st.session_state["history"].append(("You", user_input))
    st.session_state["history"].append(("Agent", response))

st.markdown("---")

for sender, msg in reversed(st.session_state["history"]):
    st.markdown(f"**{sender}:** {msg}")