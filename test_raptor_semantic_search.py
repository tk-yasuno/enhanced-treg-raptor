"""
RAPTOR Semantic Search Performance Test
セマンティック検索による精度改善テスト

This script implements:
1. Keyword-based search (baseline)
2. Pure semantic search (embeddings only)
3. Hybrid search (keyword + semantic)

比較項目:
- 検索速度
- 検索精度（スコア）
- 結果の関連性
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sentence-BERT for semantic embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Test Queries (same as keyword test)
# ============================================================================

TEST_QUERIES = [
    "What is the differentiation pathway from hematopoietic stem cells to regulatory T cells?",
    "What is the role of IL-7 receptor in common lymphoid progenitor cells?",
    "Explain the mechanism of thymic selection in CD4 positive T cells",
    "How do CD25 high expression and CD127 low expression function as Treg markers?",
    "What is the difference between thymic-derived Treg and peripherally-induced Treg?",
    "How does Foxp3 transcription factor control Treg cell differentiation?",
    "How does TSDR demethylation contribute to Treg cell stability?",
    "Explain the immunosuppressive mechanisms of regulatory T cells in detail",
    "What role does IL-10 and TGF-beta production by Treg play?",
    "What are the challenges and prospects for clinical applications of regulatory T cells?"
]

# ============================================================================
# Configuration
# ============================================================================

# Semantic model selection
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, general-purpose model (384 dimensions)
# Alternative: "pritamdeka/S-PubMedBert-MS-MARCO" for biomedical domain

# Hybrid search weights
KEYWORD_WEIGHT = 0.4  # 40% keyword score
SEMANTIC_WEIGHT = 0.6  # 60% semantic score

# Cache settings
EMBEDDINGS_CACHE_DIR = Path("data/embeddings_cache")
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Baseline: Keyword Search (from original test)
# ============================================================================

def simple_keyword_search(tree_data: Dict, query: str, top_k: int = 5) -> List[Dict]:
    """
    Original keyword-based search (baseline)
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    # Get tree_nodes from the loaded data
    tree_nodes = tree_data.get("tree_nodes", {})
    
    for node_id, node_info in tree_nodes.items():
        # Get text from summary or content
        summary = node_info.get("summary", "")
        content = node_info.get("content", "")
        text = summary + " " + content
        
        text_lower = text.lower()
        
        # Count keyword matches
        score = sum(1 for word in query_words if word in text_lower)
        
        if score > 0:
            results.append({
                "node_id": node_id,
                "score": score,
                "level": node_info.get("level", -1),
                "is_leaf": node_info.get("is_leaf", False),
                "text": text[:200]
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ============================================================================
# 2. Semantic Search Implementation
# ============================================================================

class SemanticSearchEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"🔧 Loading embedding model: {model_name}")
        # Use CUDA if available, otherwise CPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None
        self.node_ids = None
        self.node_info = None
        
    def build_embeddings(self, tree_data: Dict, cache_file: Path = None) -> None:
        """
        Build embeddings for all nodes (with caching)
        """
        if cache_file and cache_file.exists():
            print(f"📂 Loading cached embeddings from {cache_file.name}")
            cache = np.load(cache_file, allow_pickle=True).item()
            self.embeddings = cache['embeddings']
            self.node_ids = cache['node_ids']
            self.node_info = cache['node_info']
            print(f"  ✓ Loaded {len(self.node_ids)} node embeddings")
            return
        
        print("🔨 Building embeddings for all nodes...")
        texts = []
        node_ids = []
        node_info = []
        
        # Get tree_nodes from the loaded data
        tree_nodes = tree_data.get("tree_nodes", {})
        
        for node_id, info in tree_nodes.items():
            # Get text from summary or content
            summary = info.get("summary", "")
            content = info.get("content", "")
            text = summary + " " + content
            
            if text.strip():
                texts.append(text)
                node_ids.append(node_id)
                node_info.append({
                    "level": info.get("level", -1),
                    "is_leaf": info.get("is_leaf", False),
                    "text": text[:200]
                })
        
        # Generate embeddings (batch processing for efficiency)
        print(f"  Processing {len(texts)} nodes...")
        start_time = time.time()
        # Use larger batch size for GPU, smaller for CPU
        import torch
        batch_size = 32 if torch.cuda.is_available() else 8
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=batch_size, 
            convert_to_numpy=True
        )
        elapsed = time.time() - start_time
        print(f"  ✓ Generated {len(self.embeddings)} embeddings in {elapsed:.2f}s")
        
        self.node_ids = node_ids
        self.node_info = node_info
        
        # Cache the embeddings
        if cache_file:
            print(f"💾 Saving embeddings cache to {cache_file.name}")
            np.save(cache_file, {
                'embeddings': self.embeddings,
                'node_ids': node_ids,
                'node_info': node_info
            })
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Pure semantic search using cosine similarity
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "node_id": self.node_ids[idx],
                "score": float(similarities[idx]),  # 0-1 range
                "level": self.node_info[idx]["level"],
                "is_leaf": self.node_info[idx]["is_leaf"],
                "text": self.node_info[idx]["text"]
            })
        
        return results
    
    def hybrid_search(self, query: str, tree_data: Dict, 
                     keyword_weight: float = KEYWORD_WEIGHT,
                     semantic_weight: float = SEMANTIC_WEIGHT,
                     top_k: int = 5) -> List[Dict]:
        """
        Hybrid search: combines keyword and semantic scores
        """
        # Get keyword results (all nodes)
        keyword_results = simple_keyword_search(tree_data, query, top_k=100)
        keyword_scores = {r["node_id"]: r["score"] for r in keyword_results}
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=100)
        semantic_scores = {r["node_id"]: r["score"] for r in semantic_results}
        
        # Normalize scores to 0-1 range
        if keyword_scores:
            max_keyword = max(keyword_scores.values())
            keyword_scores = {k: v/max_keyword for k, v in keyword_scores.items()}
        
        # Combine scores
        all_node_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
        
        # Get tree_nodes from the loaded data
        tree_nodes = tree_data.get("tree_nodes", {})
        
        combined_results = []
        for node_id in all_node_ids:
            kw_score = keyword_scores.get(node_id, 0)
            sem_score = semantic_scores.get(node_id, 0)
            
            # Weighted combination
            hybrid_score = (keyword_weight * kw_score) + (semantic_weight * sem_score)
            
            # Get node info
            if node_id in tree_nodes:
                node_info = tree_nodes[node_id]
                summary = node_info.get("summary", "")
                content = node_info.get("content", "")
                text = (summary + " " + content)[:200]
                
                combined_results.append({
                    "node_id": node_id,
                    "score": hybrid_score,
                    "keyword_score": kw_score,
                    "semantic_score": sem_score,
                    "level": node_info.get("level", -1),
                    "is_leaf": node_info.get("is_leaf", False),
                    "text": text
                })
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]

# ============================================================================
# 3. Performance Testing
# ============================================================================

def run_comparison_test(tree_file: Path, semantic_engine: SemanticSearchEngine):
    """
    Run all three search methods and compare results
    """
    print("\n" + "="*80)
    print("RAPTOR Semantic Search Comparison Test")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Load tree data
    print(f"📂 Loading RAPTOR tree: {tree_file.name}")
    with open(tree_file, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    # Get number of nodes from tree_nodes
    num_nodes = len(tree_data.get("tree_nodes", {}))
    print(f"  ✓ Loaded {num_nodes} nodes\n")  # -1 for metadata
    
    results_comparison = []
    
    for idx, query in enumerate(TEST_QUERIES, 1):
        print("="*80)
        print(f"Query {idx}/{len(TEST_QUERIES)}")
        print("="*80)
        print(f"❓ Question: {query}\n")
        
        # 1. Keyword Search
        print("🔍 Method 1: Keyword Search (Baseline)")
        start = time.time()
        keyword_results = simple_keyword_search(tree_data, query, top_k=5)
        keyword_time = time.time() - start
        print(f"⏱️  Time: {keyword_time:.4f}s, Results: {len(keyword_results)}")
        if keyword_results:
            print(f"📊 Top result: {keyword_results[0]['node_id']} (score: {keyword_results[0]['score']})")
        
        # 2. Semantic Search
        print("\n🧠 Method 2: Semantic Search (Embeddings)")
        start = time.time()
        semantic_results = semantic_engine.semantic_search(query, top_k=5)
        semantic_time = time.time() - start
        print(f"⏱️  Time: {semantic_time:.4f}s, Results: {len(semantic_results)}")
        if semantic_results:
            print(f"📊 Top result: {semantic_results[0]['node_id']} (score: {semantic_results[0]['score']:.4f})")
        
        # 3. Hybrid Search
        print("\n🔗 Method 3: Hybrid Search (Keyword + Semantic)")
        start = time.time()
        hybrid_results = semantic_engine.hybrid_search(query, tree_data, top_k=5)
        hybrid_time = time.time() - start
        print(f"⏱️  Time: {hybrid_time:.4f}s, Results: {len(hybrid_results)}")
        if hybrid_results:
            print(f"📊 Top result: {hybrid_results[0]['node_id']} (score: {hybrid_results[0]['score']:.4f})")
            print(f"    Breakdown: keyword={hybrid_results[0]['keyword_score']:.4f}, semantic={hybrid_results[0]['semantic_score']:.4f}")
        
        # Store results for comparison
        results_comparison.append({
            "query_id": idx,
            "query": query,
            "keyword": {
                "time": keyword_time,
                "top_results": keyword_results,
                "top_score": keyword_results[0]["score"] if keyword_results else 0
            },
            "semantic": {
                "time": semantic_time,
                "top_results": semantic_results,
                "top_score": semantic_results[0]["score"] if semantic_results else 0
            },
            "hybrid": {
                "time": hybrid_time,
                "top_results": hybrid_results,
                "top_score": hybrid_results[0]["score"] if hybrid_results else 0
            }
        })
        
        print()
    
    # Summary Statistics
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    keyword_times = [r["keyword"]["time"] for r in results_comparison]
    semantic_times = [r["semantic"]["time"] for r in results_comparison]
    hybrid_times = [r["hybrid"]["time"] for r in results_comparison]
    
    print("📊 Average Search Times:")
    print(f"  Keyword:  {np.mean(keyword_times):.4f}s (± {np.std(keyword_times):.4f}s)")
    print(f"  Semantic: {np.mean(semantic_times):.4f}s (± {np.std(semantic_times):.4f}s)")
    print(f"  Hybrid:   {np.mean(hybrid_times):.4f}s (± {np.std(hybrid_times):.4f}s)")
    
    print("\n📈 Score Comparison:")
    for i, result in enumerate(results_comparison, 1):
        print(f"  Q{i}: Keyword={result['keyword']['top_score']:.2f}, "
              f"Semantic={result['semantic']['top_score']:.4f}, "
              f"Hybrid={result['hybrid']['top_score']:.4f}")
    
    # Save detailed results
    output_file = Path("results") / f"semantic_search_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("✅ Comparison test completed!\n")
    
    return results_comparison

# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Find the latest RAPTOR tree file
    tree_files = sorted(Path("results").glob("enhanced_treg_raptor_*.json"))
    if not tree_files:
        print("❌ Error: No RAPTOR tree file found!")
        print("   Expected file pattern: results/enhanced_treg_raptor_*.json")
        return
    
    tree_file = tree_files[-1]  # Use the most recent one
    
    # Load tree data
    with open(tree_file, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    # Initialize semantic search engine
    semantic_engine = SemanticSearchEngine(EMBEDDING_MODEL)
    
    # Build or load embeddings
    cache_file = EMBEDDINGS_CACHE_DIR / f"embeddings_{tree_file.stem}_{EMBEDDING_MODEL.replace('/', '_')}.npy"
    semantic_engine.build_embeddings(tree_data, cache_file=cache_file)
    
    # Run comparison test
    run_comparison_test(tree_file, semantic_engine)

if __name__ == "__main__":
    main()
