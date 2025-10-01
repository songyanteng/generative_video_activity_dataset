#!/usr/bin/env python3
"""
Flask API for video search functionality.

This API wraps the existing VideoSearchEngine to provide HTTP endpoints
that can be used by HTML snippets embedded in surveys or other web applications.
"""

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from video_search import VideoSearchEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global search engine instance
search_engine = None

def initialize_search_engine():
    """Initialize the search engine on startup."""
    global search_engine
    try:
        print("Initializing video search engine...")
        search_engine = VideoSearchEngine()
        print("✅ Search engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return False

@app.route('/')
def home():
    """Simple home page with API documentation."""
    return """
    <h1>Video Search API</h1>
    <p>This API provides video search functionality for embedding in surveys.</p>
    <h2>Endpoints:</h2>
    <ul>
        <li><code>GET /search?query=your_query&top_n=5</code> - Search for videos</li>
        <li><code>GET /health</code> - Check API health</li>
    </ul>
    """

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if search_engine is not None else "unhealthy",
        "search_engine_loaded": search_engine is not None
    })

@app.route('/search')
def search_videos():
    """
    Search for videos matching the given query.
    
    Query parameters:
    - query: Text prompt to search for (required)
    - top_n: Number of results to return (default: 5)
    - github_repo: GitHub repository (default: from config)
    - branch: Git branch (default: main)
    """
    if search_engine is None:
        return jsonify({"error": "Search engine not initialized"}), 500
    
    # Get query parameters
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    top_n = int(request.args.get('top_n', 5))
    github_repo = request.args.get('github_repo', 'generative_video_activity_dataset-8280')
    branch = request.args.get('branch', 'main')
    
    try:
        # Perform search
        results = search_engine.search(query, top_n)
        
        # Generate GitHub URLs
        results = search_engine.generate_github_urls(
            results, 
            github_repo, 
            branch
        )
        
        # Add direct video URLs for embedding
        for result in results:
            video_id = result["video_id"]
            # Use raw GitHub URLs for direct video access
            result["video_url"] = result.get("raw_url", "")
            result["embed_url"] = result["video_url"]  # For HTML5 video embedding
        
        return jsonify({
            "query": query,
            "results": results,
            "total_results": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/embed')
def embed_snippet():
    """
    Serve the HTML embed snippet with configuration.
    
    Query parameters:
    - api_url: Base URL of this API (for CORS requests)
    - default_top_n: Default number of results (1-1000)
    - github_repo: GitHub repository
    - branch: Git branch
    - show_controls: Show full video controls (true/false)
    """
    api_url = request.args.get('api_url', request.url_root.rstrip('/'))
    default_top_n = int(request.args.get('default_top_n', 3))
    github_repo = request.args.get('github_repo', 'generative_video_activity_dataset-8280')
    branch = request.args.get('branch', 'main')
    show_controls = request.args.get('show_controls', 'false').lower() == 'true'
    
    # Read the HTML widget template
    widget_path = os.path.join(os.path.dirname(__file__), 'video_embed_widget.html')
    
    try:
        with open(widget_path, 'r', encoding='utf-8') as f:
            html_template = f.read()
        
        # Replace placeholders with actual values
        html_content = html_template.replace('{{API_URL}}', api_url)
        html_content = html_content.replace('{{DEFAULT_TOP_N}}', str(default_top_n))
        html_content = html_content.replace('{{GITHUB_REPO}}', github_repo)
        html_content = html_content.replace('{{BRANCH}}', branch)
        html_content = html_content.replace('{{SHOW_CONTROLS}}', str(show_controls).lower())
        
        return html_content
        
    except FileNotFoundError:
        return f"<p>Error: HTML widget template not found at {widget_path}</p>", 404

if __name__ == '__main__':
    # Initialize search engine
    if not initialize_search_engine():
        print("Failed to initialize search engine. Exiting.")
        exit(1)
    
    # Run the Flask app
    print("\n🚀 Starting Video Search API...")
    print("📡 API will be available at: http://localhost:5000")
    print("🔍 Search endpoint: http://localhost:5000/search?query=your_query&top_n=5")
    print("📄 Embed snippet: http://localhost:5000/embed")
    print("\n💡 To get an embeddable snippet, visit: http://localhost:5000/embed")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
