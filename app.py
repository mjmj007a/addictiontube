from flask import Flask, request, jsonify
from pinecone import Pinecone  # Import Pinecone class
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone and OpenAI with environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("addictiontube-index")
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

@app.route('/search_stories', methods=['GET'])
def search_stories():
    query = request.args.get('q', 'recovery from addiction')
    category = request.args.get('category', '1028')
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, filter={"category": {"$eq": category}})
    stories = [
        {
            "id": match['id'],
            "score": match['score'],
            "title": match['metadata'].get('title', 'N/A'),
            "description": match['metadata'].get('description', '')
        } for match in results['matches']
    ]
    return jsonify(stories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)