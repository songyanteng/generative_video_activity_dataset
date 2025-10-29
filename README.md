# A Generative Video Dataset to Represent Everyday Human Activity

A comprehensive video dataset featuring 1000 AI-generated videos of everyday activites and retrieval functionality.

## Overview

This repository contains a curated video dataset with:

- **1000 AI-generated videos** with diverse scenarios
- **4000 AI-generated audio clips** with ambient sounds and sound effects
- **Semantic search tool** for finding relevant videos using natural language
- **Web embed widget** for integrating video search into surveys and web applications

## Repository Structure

```
generative_video_activity_dataset/
├── audio_files/                     # 4000 audio files (.mp3)
├── search/                          # Video search tools
│   ├── run_search_api.py            # Flask web API for embedding
│   ├── video_embed_widget.html      # HTML widget for surveys
│   ├── video_embeddings.pkl         # Pre-computed video embeddings
│   ├── video_search.py              # Main search script
│   └── video_search_page.html       # Test page for the widget
├── video_files/
│   ├── original/                    # 1000 original video files (.mp4)
│   └── stylised/                    # 1000 stylised video files (.mp4)
├── requirements.txt                 # Python package dependencies
└── video_metadata.json              # Video metadata and prompts
```

## Prerequisites

- **Python**: Python 3.9 - 3.11 installed and available on your PATH (`python --version`).
- **Environment manager**: Either Conda (Miniconda/Anaconda) or `venv` for virtual environments.
- **Package manager**: `pip` is up to date (`python -m pip install --upgrade pip`).
- **Git (optional)**: For cloning the GitHub repository locally.
- **GPU (optional)**: NVIDIA GPU with compatible driver and CUDA/cuDNN if you wish to run a CUDA-enabled PyTorch build. Otherwise, CPU-only works.
- **Operating system**: Windows 10/11, macOS 12+, or a recent Linux distro.
- **Internet access**: To install Python packages the first time and to access the videos.
- **Disk space**: Several GB free for the dataset (videos, audio, embeddings) and dependencies.

## Quick Start

### 1. Environment Setup

**Option A: Using Conda + requirements.txt (Recommended)**

```bash
# Download the repository and navigate to the folder
cd generative_video_activity_dataset

# Create a new conda environment with Python 3.10 (recommended)
conda create -n video_search python=3.10

# Activate the environment
conda activate video_search

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**Option B: Using pip + requirements.txt**

```bash
# Download the repository and navigate to the folder
cd generative_video_activity_dataset

# Create virtual environment
# Windows
py -3.10 -m venv video_search
# macOS/Linux
python3.10 -m venv video_search

# Activate the environment
# Windows
video_search\Scripts\activate
# macOS/Linux
source video_search/bin/activate

# Install all required packages
pip install -r requirements.txt
```

**Option C: Manual Installation (Not Recommended)**

```bash
# Install packages individually
pip install flask==2.3.3 flask-cors==4.0.0 torch>=1.9.0 sentence-transformers>=2.2.0 numpy>=1.21.0 requests==2.28.1
```

Note: If you have a CUDA-capable GPU and a CUDA-enabled PyTorch build installed, it will use the GPU automatically. If you don't have CUDA, performance will be CPU-only (slower but still functional).

### 2. Verify Installation

```bash
cd search
python -c "import torch, sentence_transformers, numpy; print('All dependencies installed successfully!')"
```

---

## Video Search Usage

### Command Line Interface

#### Basic Search Commands

```bash
# Navigate to the search directory
cd search

# Basic search
python video_search.py "your search prompt"

# Search with specific number of results
python video_search.py "your search prompt" --top-n 5

# Search and save results to file
python video_search.py "your search prompt" --top-n 5 --save-results results.json
```

#### More Search Options

```bash
# Search with custom GitHub repository
python video_search.py "your search prompt" --github-repo "user/repo"

# Output in JSON format
python video_search.py "your search prompt" --output-format json

# Hide metadata for cleaner output
python video_search.py "your search prompt" --no-metadata

# Search with custom video path within repository
python video_search.py "your search prompt" --video-path "custom/path"
```

#### Complete Command Reference

| Option            | Description                          | Default                                  | Example                       |
| ----------------- | ------------------------------------ | ---------------------------------------- | ----------------------------- |
| `query`           | Text prompt to search for (required) | -                                        | Examples provided below       |
| `--top-n`         | Number of results to return          | 10                                       | `--top-n 5`                   |
| `--github-repo`   | GitHub repository (username/repo)    | `generative_video_activity_dataset-8280` | `--github-repo "user/repo"`   |
| `--branch`        | Git branch name                      | `main`                                   | `--branch "develop"`          |
| `--video-path`    | Path to videos in repository         | `video_files/original`                   | `--video-path "custom/path"`  |
| `--no-metadata`   | Hide additional metadata             | False                                    | `--no-metadata`               |
| `--output-format` | Output format (text/json)            | `text`                                   | `--output-format json`        |
| `--save-results`  | Save results to JSON file            | None                                     | `--save-results results.json` |

### Programmatic Usage

#### Basic Python Integration

```python
from search.video_search import VideoSearchEngine

# Initialize search engine
search_engine = VideoSearchEngine()

# Search for videos
results = search_engine.search("peaceful indoor activity", top_n=5)

# Generate GitHub URLs
results = search_engine.generate_github_urls(
    results,
    github_repo="generative_video_activity_dataset-8280",
    branch="main",
    video_path="video_files/original"
)

# Process results
for result in results:
    print(f"Video ID: {result['video_id']}")
    print(f"Prompt: {result['video_prompt']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"GitHub URL: {result['github_url']}")
```

## Search Examples

### Example Queries

```bash
python video_search.py "Sunny kitchen morning, person eating breakfast at table, gentle natural light"
python video_search.py "Park trail at dusk, steady jogging, rhythmic breathing, trees passing by"
python video_search.py "Small living room afternoon, person folding laundry neatly on couch"
```

### Output Examples

#### Text Output

```
================================================================================
VIDEO SEARCH RESULTS
================================================================================
Rank 1: d956ff18-2c7b-4ea5-b102-33a14153a9a7
Similarity Score: 0.8234
Video Prompt: I tap out a text on my phone while waiting for the bus, raindrops drip from the shelter roof above.
GitHub URL: https://github.com/songyanteng/generative_video_activity_dataset/blob/main/video_files/original/d956ff18-2c7b-4ea5-b102-33a14153a9a7.mp4
Raw URL: https://raw.githubusercontent.com/songyanteng/generative_video_activity_dataset/main/video_files/original/d956ff18-2c7b-4ea5-b102-33a14153a9a7.mp4
----------------------------------------
```

#### JSON Output

```json
[
  {
    "rank": 1,
    "video_id": "d956ff18-2c7b-4ea5-b102-33a14153a9a7",
    "similarity_score": 0.8234,
    "video_prompt": "I tap out a text on my phone while waiting for the bus, raindrops drip from the shelter roof above.",
    "github_url": "https://github.com/songyanteng/generative_video_activity_dataset/blob/main/video_files/original/d956ff18-2c7b-4ea5-b102-33a14153a9a7.mp4",
    "raw_url": "https://raw.githubusercontent.com/songyanteng/generative_video_activity_dataset/main/video_files/original/d956ff18-2c7b-4ea5-b102-33a14153a9a7.mp4"
  }
]
```

## Configuration

### Automatic Embedding Generation

The system automatically generates text embeddings for the video prompts if they do not exist:

```bash
# First run will generate embeddings automatically
python video_search.py "your search prompt"
# Output: "Embeddings file not found: ../video_embeddings.pkl"
# Output: "Generating embeddings from metadata..."
# Output: "Embeddings saved successfully."
```

### Custom Repository Settings

Edit `search/video_search.py`:

```python
# Update default repository to use your own dataset if needed
DEFAULT_GITHUB_REPO = "user/repo"
DEFAULT_BRANCH = "main"
DEFAULT_VIDEO_PATH = "custom/path"
```

## Integration Examples

### Web Application Integration

```python
from flask import Flask, request, jsonify
from search.video_search import VideoSearchEngine

app = Flask(__name__)
search_engine = VideoSearchEngine()

@app.route('/search', methods=['POST'])
def search_videos():
    query = request.json['query']
    top_n = request.json.get('top_n', 10)
    results = search_engine.search(query, top_n=top_n)
    return jsonify(results)
```

### Jupyter Notebook Integration

```python
# In a Jupyter notebook
from search.video_search import VideoSearchEngine
import pandas as pd

# Initialize and search
search_engine = VideoSearchEngine()
results = search_engine.search("your search prompt", top_n=10)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
df.head()
```

---

## Web Embed Widget Usage

### Overview

The repository includes a web embed widget that allows you to integrate video search functionality directly into surveys, websites, and other web applications. This is useful for user studies, surveys, and content discovery interfaces.

### Quick Start - Web Widget

1. **Start the API server:**

```bash
cd search
python run_search_api.py
```

2. **Test the widget:**

   - Open `search/video_search_page.html` in your browser
   - Try searching for different activities
   - Test the video controls

3. **Embed in your application:**
   - Use the iframe approach: `<iframe src="http://localhost:5000/embed"></iframe>`
   - Or copy the processed HTML from the `/embed` endpoint

### API Endpoints

- `GET /search?query=your_query&top_n=5` - Search for videos
- `GET /embed?default_top_n=3&show_controls=true` - Get embeddable widget
- `GET /health` - Check API status

### Embedding Examples

**For HTML-Enabled Surveys:**

```html
<iframe
  src="http://localhost:5000/embed?default_top_n=5&show_controls=true"
  width="100%"
  height="600"
  frameborder="0"
></iframe>
```

**For Custom Web Applications:**

```html
<div id="video-search-widget">
  <!-- Paste the HTML content from /embed endpoint here -->
</div>
```

---

## Dataset Details

### Video Content

- **Total Videos**: 1000 AI-generated videos
- **Duration**: 10 seconds each
- **Content**: Diverse scenarios including indoor/outdoor activities, technology use, daily routines

### Audio Content

- **Total Audio**: 4000 AI-generated audio (.mp3) clips
- **Types**: Ambient sounds, sound effects, background music
- **Usage**: Paired with the generated videos

### Metadata

- **Video Prompts**: Detailed text descriptions for each video
- **Audio Prompts**: Sound design descriptions
- **Embeddings**: Pre-computed semantic embeddings for fast search

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
