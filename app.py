from flask import Flask, render_template, request
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# Load the model and tokenizer from Hugging Face Transformers
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a global variable for storing the FAISS index
faiss_index = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        video_url = request.form.get("video_url")
        answer = retrieve_answer(query, video_url)
        return render_template("index.html", query=query, answer=answer)
    return render_template("index.html")

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

    # You can now use the similar_doc to generate a response
    return similar_doc

if __name__ == "__main__":
    app.run(debug=True)
