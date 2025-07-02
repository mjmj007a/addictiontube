from flask import Flask, request, jsonify
from pinecone import Pinecone
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI and Pinecone using environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Connect to Pinecone index
index_name = "addictiontube-index"
index = pc.Index(index_name)

# Function to get embedding from OpenAI
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print("[ERROR] Failed to generate embedding:", e)
        return None

# Search endpoint
@app.route('/search_stories', methods=['GET'])
def search_stories():
    try:
        query = request.args.get('q', 'recovery from addiction')
        category = request.args.get('category', '1028')
        print(f"[DEBUG] Received query: '{query}', category: '{category}'")

        query_embedding = get_embedding(query)
        if query_embedding is None:
            return jsonify({"error": "Embedding generation failed"}), 500
        print("[DEBUG] Successfully got embedding")

        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )
        print(f"[DEBUG] Retrieved {len(results['matches'])} matches from Pinecone")

        stories = [
            {
                "id": match['id'],
                "score": match['score'],
                "title": match['metadata'].get('title', 'N/A'),
                "description": match['metadata'].get('description', '')
            } for match in results['matches']
        ]
        return jsonify(stories)

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500

# Local dev only
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
