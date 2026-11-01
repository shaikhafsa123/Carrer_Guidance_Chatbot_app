# ==========================================
# CAREER GUIDANCE CHATBOT
# Streamlit + ChromaDB + TF-IDF + Groq
# ==========================================

import os
import io
import streamlit as st
import chromadb

from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. STREAMLIT INTERFACE
# ==========================================

st.set_page_config(page_title="Career Guidance Chatbot")

st.title("Career Guidance Chatbot")

st.markdown(
    "Get personalized career advice using RAG and AI."
)


# ==========================================
# 2. GROQ API KEY
# ==========================================

api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)


# ==========================================
# 3. INITIALIZE CHROMADB
# ==========================================

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()


client = get_chroma_client()


# ==========================================
# 4. PDF TEXT EXTRACTION
# ==========================================

def ingest_pdf(file):

    reader = PdfReader(io.BytesIO(file.getvalue()))

    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    return text


# ==========================================
# 5. PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload Career Resources (PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    file_signature = tuple(
        (file.name, file.size) for file in uploaded_files
    )

    # Process only when uploaded files change

    if st.session_state.get("file_signature") != file_signature:

        with st.spinner("Processing PDF files..."):

            all_chunks = []
            all_metadata = []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80
            )

            for file in uploaded_files:

                text = ingest_pdf(file)

                chunks = splitter.split_text(text)

                for i, chunk in enumerate(chunks):

                    all_chunks.append(chunk)

                    all_metadata.append({
                        "filename": file.name
                    })

            if not all_chunks:

                st.warning(
                    "No readable text found in the uploaded PDFs."
                )

            else:

                try:

                    # Create TF-IDF embeddings
                    vectorizer = TfidfVectorizer()

                    document_embeddings = vectorizer.fit_transform(
                        all_chunks
                    ).toarray().tolist()

                    # Recreate collection
                    try:
                        client.delete_collection(
                            "career_knowledge_base"
                        )
                    except Exception:
                        pass

                    collection = client.create_collection(
                        "career_knowledge_base"
                    )

                    ids = [
                        f"document_{i}"
                        for i in range(len(all_chunks))
                    ]

                    collection.add(
                        ids=ids,
                        documents=all_chunks,
                        metadatas=all_metadata,
                        embeddings=document_embeddings
                    )

                    st.session_state.vectorizer = vectorizer
                    st.session_state.file_signature = file_signature
                    st.session_state.document_count = len(all_chunks)

                    st.success(
                        f"{len(uploaded_files)} PDF(s) processed successfully!"
                    )

                except ValueError as e:

                    st.error(
                        f"Could not process PDF text: {e}"
                    )


# ==========================================
# 6. USER CAREER QUESTION
# ==========================================

user_query = st.text_input(
    "Ask your career question:"
)


# ==========================================
# 7. GENERATE CAREER ADVICE
# ==========================================

if st.button("Get Advice"):

    if not api_key:

        st.warning("Please enter your Groq API key.")

    elif not user_query.strip():

        st.warning("Please enter a career question.")

    elif "vectorizer" not in st.session_state:

        st.warning("Please upload career PDF files first.")

    else:

        try:

            with st.spinner("Searching documents..."):

                collection = client.get_collection(
                    "career_knowledge_base"
                )

                vectorizer = st.session_state.vectorizer

                query_embedding = vectorizer.transform(
                    [user_query]
                ).toarray().tolist()

                # ChromaDB vector search
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(
                        5,
                        st.session_state.document_count
                    )
                )

                retrieved_docs = results["documents"][0]

                retrieved_metadata = results["metadatas"][0]

            # ==================================
            # 8. GROQ AI RESPONSE
            # ==================================

            with st.spinner("Generating career advice..."):

                llm = ChatGroq(
                    temperature=0,
                    model="openai/gpt-oss-20b",
                    api_key=api_key
                )

                context = "\n\n".join(retrieved_docs)

                prompt = f"""
                You are an expert career guidance AI assistant.

                Use the following retrieved career resources
                to answer the user's question.

                CAREER RESOURCES:
                {context}

                USER QUESTION:
                {user_query}

                Provide a useful and personalized roadmap covering:

                1. Skills to learn
                2. Recommended companies
                3. Learning resources
                4. Steps to improve career readiness

                If the resources do not contain enough information,
                clearly mention that instead of inventing facts.
                """

                response = llm.invoke(prompt)

            # ==================================
            # 9. DISPLAY RESULTS
            # ==================================

            st.subheader("Retrieved Documents")

            for i, doc in enumerate(retrieved_docs):

                st.write(f"**Document {i + 1}**")
                st.write(doc)
                st.caption(
                    "Source: " +
                    retrieved_metadata[i]["filename"]
                )

            st.subheader("Personalized Career Advice")

            st.write(response.content)

        except Exception as e:

            st.error(f"An error occurred: {e}")