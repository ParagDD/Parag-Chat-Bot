import streamlit as st
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi

import streamlit as st

# Inject custom CSS to hide the GitHub logo
st.markdown(
    """
    <style>
    /* Hide the GitHub logo */
    .st-emotion-cache-1p1m4ay.e3g6aar0 {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)





# Your Streamlit app code here...





# Define a Streamlit title
st.title("YouTube Transcript Search")

# Load the model and tokenizer from Hugging Face Transformers
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a global variable for storing the FAISS index
faiss_index = None

# Define a function to retrieve answers
def retrieve_answer(question, video_url):
    global faiss_index

    # Extract subtitles from the YouTube video using youtube_transcript_api
    video_id = video_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    subtitles = " ".join([item['text'] for item in transcript])

    # Split subtitles into chunks
    chunk_size = 1000
    chunk_overlap = 200
    chunks = [subtitles[i:i + chunk_size] for i in range(0, len(subtitles), chunk_size - chunk_overlap)]

    # Tokenize all documents at once for efficiency
    doc_texts = [doc if isinstance(doc, str) else ' '.join(map(str, doc)) for doc in chunks]
    input_tokens = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True)

    # Create embeddings for the chunks
    with torch.no_grad():
        output = model(**input_tokens)

    # Extract the [CLS] token's representations as document embeddings
    embeddings = output.last_hidden_state[:, 0, :].numpy()

    # If the FAISS index is not yet created, create it
    if faiss_index is None:
        embedding_dim = len(embeddings[0])
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_vectors = embeddings.astype('float32')
        faiss_index.add(faiss_vectors)

    # Perform retrieval using FAISS and Hugging Face Transformers
    input_tokens = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**input_tokens).last_hidden_state[:, 0, :].numpy()

    # Search for similar embeddings in the FAISS index
    D, I = faiss_index.search(query_embedding.reshape(1, -1), k=1)

    # I[0][0] contains the index of the most similar document
    similar_doc_index = I[0][0]
    similar_doc = chunks[similar_doc_index]

    return similar_doc

# Define a Streamlit sidebar with user input fields
st.sidebar.title("User Input")
query = st.sidebar.text_input("Enter your question:")
video_url = st.sidebar.text_input("Enter the YouTube video URL:")

# Create a placeholder for displaying retrieved answers
answer_placeholder = st.empty()

# Check if user input exists
if query and video_url:
    # Retrieve and display the answer when the user clicks the "Search" button
    if st.sidebar.button("Search"):
        answer = retrieve_answer(query, video_url)
        # Display the retrieved answer
        answer_placeholder.header("Retrieved Answer:")
        answer_placeholder.write(answer)
