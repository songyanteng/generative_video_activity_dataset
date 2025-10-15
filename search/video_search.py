#!/usr/bin/env python3
"""
Video Search Script for Researchers

This script allows researchers to input text prompts and find the top-n matching videos
from the generated dataset. It uses the existing embedding system to perform semantic
search and returns GitHub URLs for the matching videos.

Usage:
    python video_search.py "your search prompt" --top-n 10
    python video_search.py "peaceful bedroom scene" --top-n 5 --github-repo "username/repo"
"""

import numpy as np
import pickle
import torch
import argparse
import json
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional

# Default GitHub repository configuration
DEFAULT_GITHUB_REPO = "generative_video_activity_dataset-8280"
DEFAULT_BRANCH = "main"
DEFAULT_VIDEO_PATH = "video_files/original"

class VideoSearchEngine:
    """Video search engine using semantic embeddings."""
    
    def __init__(self, embeddings_path: str = "video_embeddings.pkl", 
                 metadata_path: str = "../video_metadata.json"):
        """
        Initialize the video search engine.
        
        Args:
            embeddings_path: Path to the pickled embeddings file
            metadata_path: Path to the dataset metadata JSON file
        """
        # Resolve paths relative to this file's directory to avoid CWD issues in deployments
        base_dir = os.path.dirname(__file__)
        self.embeddings_path = (
            embeddings_path if os.path.isabs(embeddings_path)
            else os.path.normpath(os.path.join(base_dir, embeddings_path))
        )
        self.metadata_path = (
            metadata_path if os.path.isabs(metadata_path)
            else os.path.normpath(os.path.join(base_dir, metadata_path))
        )
        self.video_ids = None
        self.video_texts = None
        self.video_embeddings = None
        self.model = None
        self.metadata = None
        
        # Load all data
        self._load_embeddings()
        self._load_metadata()
        self._load_model()
    
    def _generate_embeddings(self):
        """Generate embeddings from video metadata if they don't exist."""
        print("Generating embeddings from video metadata...")
        
        # Load dataset metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            videos = json.load(f)
        print(f"Loaded {len(videos)} video entries from metadata.")
        
        # Load sentence transformer model
        print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")
        
        # Prepare lists
        video_ids = []
        video_texts = []
        
        print("Building text prompts for each video...")
        for i, video in enumerate(videos):
            combined_text = video['video_prompt']
            video_ids.append(video['video_id'])
            video_texts.append(combined_text)
            
            if i < 3:  # Show first few as a sample
                print(f"  - Sample {i+1} [ID: {video['video_id']}]: {combined_text[:100]}...")
        
        print(f"Prepared combined text for {len(video_texts)} videos.")
        
        # Generate embeddings
        print("Generating embeddings...")
        video_embeddings = model.encode(video_texts, convert_to_tensor=True)
        print(f"Embeddings generated. Shape: {video_embeddings.shape}")
        
        # Save to pickle
        print(f"Saving embeddings to {self.embeddings_path}...")
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump({
                "video_ids": video_ids,
                "embeddings": video_embeddings.cpu().numpy(),
                "texts": video_texts
            }, f)
        
        print("Embeddings saved successfully.")
        return video_ids, video_texts, video_embeddings

    def _load_embeddings(self):
        """Load video embeddings from pickle file, generate if not exists."""
        if not os.path.exists(self.embeddings_path):
            print(f"Embeddings file not found: {self.embeddings_path}")
            print("Generating embeddings from metadata...")
            video_ids, video_texts, video_embeddings = self._generate_embeddings()
            self.video_ids = video_ids
            self.video_texts = video_texts
            self.video_embeddings = video_embeddings
            return
        
        print(f"Loading video embeddings from {self.embeddings_path}...")
        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)
        
        self.video_ids = data["video_ids"]
        self.video_texts = data["texts"]
        self.video_embeddings = torch.tensor(data["embeddings"])
        print(f"Loaded {len(self.video_ids)} video embeddings.")
    
    def _load_metadata(self):
        """Load video metadata from JSON file."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        print(f"Loading video metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # Create a lookup dictionary for quick access
        self.metadata_lookup = {entry["video_id"]: entry for entry in self.metadata}
        print(f"Loaded metadata for {len(self.metadata)} videos.")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully.")
    
    def search(self, query: str, top_n: int = 10) -> List[Dict]:
        """
        Search for videos matching the given query.
        
        Args:
            query: Text prompt to search for
            top_n: Number of top results to return
            
        Returns:
            List of dictionaries containing video information and similarity scores
        """
        if self.model is None or self.video_embeddings is None:
            raise RuntimeError("Search engine not properly initialized")
        
        print(f"Searching for: '{query}'")
        print(f"Returning top {top_n} results...")
        
        # Encode the query
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(device)
        video_embeddings = self.video_embeddings.to(device)
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(query_embedding, video_embeddings)
        
        # Get top results
        top_results = cosine_scores[0].topk(min(top_n, len(self.video_ids)))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
            video_id = self.video_ids[idx.item()]
            similarity_score = float(score.item())
            
            # Get metadata for this video
            video_metadata = self.metadata_lookup.get(video_id, {})
            
            result = {
                "rank": i + 1,
                "video_id": video_id,
                "similarity_score": similarity_score,
                "video_prompt": video_metadata.get("video_prompt", "N/A"),
            }
            results.append(result)
        
        return results
    
    def generate_github_urls(self, results: List[Dict], github_repo: str, 
                           branch: str = "main", video_path: str = None) -> List[Dict]:
        """
        Generate GitHub URLs for the search results.
        
        Args:
            results: List of search results from the search method
            github_repo: GitHub repository in format "username/repo"
            branch: Git branch name (default: "main")
            video_path: Path to videos in the repository (default: auto-detect)
            
        Returns:
            List of results with added GitHub URLs
        """
        if video_path is None:
            video_path = DEFAULT_VIDEO_PATH
        
        # base_url = f"https://github.com/{github_repo}/blob/{branch}/{video_path}"
        base_url = f"https://anonymous.4open.science/r/{github_repo}/blob/{branch}/{video_path}"
        
        for result in results:
            video_id = result["video_id"]
            # result["github_url"] = f"{base_url}/{video_id}.mp4"
            result["github_url"] = f"https://anonymous.4open.science/r/{github_repo}/{video_path}/{video_id}.mp4"
            # result["raw_url"] = f"https://raw.githubusercontent.com/{github_repo}/{branch}/{video_path}/{video_id}.mp4"
            result["raw_url"] = f"https://anonymous.4open.science/r/{github_repo}/{video_path}/{video_id}.mp4"
        
        return results

def format_results(results: List[Dict], show_metadata: bool = True) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search results
        show_metadata: Whether to show additional metadata
        
    Returns:
        Formatted string representation of results
    """
    output = []
    output.append("=" * 80)
    output.append("VIDEO SEARCH RESULTS")
    output.append("=" * 80)
    
    for result in results:
        output.append(f"\nRank {result['rank']}: {result['video_id']}")
        output.append(f"Similarity Score: {result['similarity_score']:.4f}")
        output.append(f"GitHub URL: {result.get('github_url', 'N/A')}")
        output.append(f"Raw URL: {result.get('raw_url', 'N/A')}")
        
        if show_metadata:
            output.append(f"Video Prompt: {result['video_prompt']}")
        
        output.append("-" * 40)
    
    return "\n".join(output)

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Search for videos using semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_search.py "peaceful bedroom scene" --top-n 5
  python video_search.py "person using phone" --top-n 10 --github-repo "username/repo"
  python video_search.py "nighttime activity" --top-n 3 --no-metadata
        """
    )
    
    parser.add_argument("query", help="Text prompt to search for")
    parser.add_argument("--top-n", type=int, default=10, 
                       help="Number of top results to return (default: 10)")
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO,
                       help=f"GitHub repository in format 'username/repo' (default: {DEFAULT_GITHUB_REPO})")
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                       help=f"Git branch name (default: {DEFAULT_BRANCH})")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH,
                       help=f"Path to videos in repository (default: {DEFAULT_VIDEO_PATH})")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Hide additional metadata in output")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--save-results", type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Initialize search engine
        print("Initializing video search engine...")
        search_engine = VideoSearchEngine()
        
        # Perform search
        results = search_engine.search(args.query, args.top_n)
        
        # Generate GitHub URLs
        results = search_engine.generate_github_urls(
            results, 
            args.github_repo, 
            args.branch, 
            args.video_path
        )
        
        # Format and display results
        if args.output_format == "json":
            # Remove the model reference for JSON serialization
            json_results = []
            for result in results:
                json_result = {k: v for k, v in result.items()}
                json_results.append(json_result)
            
            output = json.dumps(json_results, indent=2)
        else:
            output = format_results(results, not args.no_metadata)
        
        print(output)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w', encoding='utf-8') as f:
                if args.output_format == "json":
                    f.write(output)
                else:
                    # Save as JSON regardless of display format
                    json_results = []
                    for result in results:
                        json_result = {k: v for k, v in result.items()}
                        json_results.append(json_result)
                    json.dump(json_results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
