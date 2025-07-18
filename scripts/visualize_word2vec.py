#!/usr/bin/env python3
"""
Word2Vec Visualization Tool

This script provides multiple visualization methods for Word2Vec embeddings:
1. 2D Interactive Plot (UMAP/t-SNE)
2. 3D Interactive Plot
3. Similarity Network
4. Similarity Heatmap
5. Word Analogy Visualization
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import warnings
warnings.filterwarnings('ignore')

class Word2VecVisualizer:
    def __init__(self, embeddings_file: str):
        """Load embeddings from JSON file exported by Rust."""
        print(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'r') as f:
            data = json.load(f)

        self.words = data['words']
        self.embeddings = np.array(data['embeddings'])
        self.embedding_dim = data['embedding_dim']
        self.vocab_size = len(self.words)

        print(f"Loaded {self.vocab_size} words with {self.embedding_dim}-dimensional embeddings")

        # Create word-to-index mapping
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}

        # Precompute similarity matrix for faster lookups
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print("Ready for visualization!")

    def reduce_dimensions(self, method='umap', n_components=2, **kwargs):
        """Reduce dimensionality using UMAP, t-SNE, or PCA."""
        print(f"Reducing dimensions using {method.upper()} to {n_components}D...")

        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'cosine')
            )
        elif method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000)
            )
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        reduced_embeddings = reducer.fit_transform(self.embeddings)
        return reduced_embeddings

    def plot_2d_interactive(self, method='umap', max_words=500, **kwargs):
        """Create interactive 2D scatter plot."""
        # Limit words for better performance
        indices = list(range(min(max_words, len(self.words))))
        words_subset = [self.words[i] for i in indices]
        embeddings_subset = self.embeddings[indices]

        # Create temporary visualizer for subset
        temp_viz = Word2VecVisualizer.__new__(Word2VecVisualizer)
        temp_viz.words = words_subset
        temp_viz.embeddings = embeddings_subset
        temp_viz.embedding_dim = self.embedding_dim
        temp_viz.vocab_size = len(words_subset)

        reduced_2d = temp_viz.reduce_dimensions(method=method, n_components=2, **kwargs)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_2d[:, 0],
            'y': reduced_2d[:, 1],
            'word': words_subset,
            'index': range(len(words_subset))
        })

        # Create interactive plot
        fig = px.scatter(
            df, x='x', y='y',
            hover_data=['word'],
            title=f'Word2Vec Embeddings - 2D {method.upper()} Projection',
            labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
        )

        # Add word labels for a subset of points
        label_indices = range(0, len(words_subset), max(1, len(words_subset) // 50))
        for i in label_indices:
            fig.add_annotation(
                x=reduced_2d[i, 0],
                y=reduced_2d[i, 1],
                text=words_subset[i],
                showarrow=False,
                font=dict(size=8),
                bgcolor="rgba(255,255,255,0.8)"
            )

        fig.update_traces(
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         f'{method.upper()} 1: %{{x:.2f}}<br>' +
                         f'{method.upper()} 2: %{{y:.2f}}<extra></extra>'
        )

        return fig

    def plot_3d_interactive(self, method='umap', max_words=500, **kwargs):
        """Create interactive 3D scatter plot."""
        # Limit words for better performance
        indices = list(range(min(max_words, len(self.words))))
        words_subset = [self.words[i] for i in indices]
        embeddings_subset = self.embeddings[indices]

        # Create temporary visualizer for subset
        temp_viz = Word2VecVisualizer.__new__(Word2VecVisualizer)
        temp_viz.words = words_subset
        temp_viz.embeddings = embeddings_subset
        temp_viz.embedding_dim = self.embedding_dim
        temp_viz.vocab_size = len(words_subset)

        reduced_3d = temp_viz.reduce_dimensions(method=method, n_components=3, **kwargs)

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_3d[:, 0],
            y=reduced_3d[:, 1],
            z=reduced_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=4,
                opacity=0.7,
                color=list(range(len(words_subset))),
                colorscale='Viridis',
                showscale=True
            ),
            text=words_subset,
            textposition="middle center",
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>' +
                         f'{method.upper()} 1: %{{x:.2f}}<br>' +
                         f'{method.upper()} 2: %{{y:.2f}}<br>' +
                         f'{method.upper()} 3: %{{z:.2f}}<extra></extra>'
        )])

        fig.update_layout(
            title=f'Word2Vec Embeddings - 3D {method.upper()} Projection',
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3'
            ),
            height=800
        )

        return fig

    def plot_similarity_network(self, threshold=0.7, max_words=100, layout='spring'):
        """Create network graph of word similarities."""
        # Use subset for performance
        indices = list(range(min(max_words, len(self.words))))
        words_subset = [self.words[i] for i in indices]
        similarity_subset = self.similarity_matrix[indices][:, indices]

        # Create network graph
        G = nx.Graph()

        # Add nodes
        for word in words_subset:
            G.add_node(word)

        # Add edges for similarities above threshold
        for i, word1 in enumerate(words_subset):
            for j, word2 in enumerate(words_subset[i+1:], i+1):
                similarity = similarity_subset[i, j]
                if similarity > threshold:
                    G.add_edge(word1, word2, weight=similarity)

        print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        if G.number_of_edges() == 0:
            print(f"No edges found with threshold {threshold}. Try lowering the threshold.")
            return None

        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        # Prepare data for plotly
        edge_x, edge_y = [], []
        edge_info = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} - {edge[1]}: {weight:.3f}")

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())

        # Calculate node degrees for sizing
        node_degrees = [G.degree(node) for node in G.nodes()]

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=[max(10, deg * 3) for deg in node_degrees],
                color=node_degrees,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node Degree")
            ),
            hovertext=[f"{word}<br>Connections: {deg}" for word, deg in zip(node_text, node_degrees)]
        ))

        fig.update_layout(
            title=f'Word Similarity Network (threshold={threshold})',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"Network of {len(words_subset)} words with similarity > {threshold}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )

        return fig

    def plot_similarity_heatmap(self, words_list: list[str]):
        """Create similarity heatmap for specific words."""
        # Find indices of words
        indices = []
        found_words = []
        for word in words_list:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
                found_words.append(word)
            else:
                print(f"Warning: Word '{word}' not found in vocabulary")

        if len(found_words) < 2:
            print("Need at least 2 valid words for heatmap")
            return None

        # Extract similarity submatrix
        similarity_submatrix = self.similarity_matrix[indices][:, indices]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_submatrix,
            x=found_words,
            y=found_words,
            colorscale='RdYlBu_r',
            zmin=0, zmax=1,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='Word Similarity Heatmap',
            xaxis_title='Words',
            yaxis_title='Words'
        )

        return fig

    def find_analogies(self, word1: str, word2: str, word3: str, top_k=5):
        """Find words that complete the analogy: word1 is to word2 as word3 is to ?"""
        if not all(w in self.word_to_idx for w in [word1, word2, word3]):
            missing = [w for w in [word1, word2, word3] if w not in self.word_to_idx]
            print(f"Words not found: {missing}")
            return []

        # Get embeddings
        emb1 = self.embeddings[self.word_to_idx[word1]]
        emb2 = self.embeddings[self.word_to_idx[word2]]
        emb3 = self.embeddings[self.word_to_idx[word3]]

        # Calculate target vector: word2 - word1 + word3
        target_vector = emb2 - emb1 + emb3
        target_vector = target_vector / np.linalg.norm(target_vector)  # Normalize

        # Calculate similarities to all words
        similarities = cosine_similarity([target_vector], self.embeddings)[0]

        # Get top k similar words (excluding input words)
        excluded_indices = {self.word_to_idx[w] for w in [word1, word2, word3]}

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if idx not in excluded_indices:
                results.append((self.words[idx], similarities[idx]))
                if len(results) >= top_k:
                    break

        return results

    def get_most_similar(self, word: str, top_k=10):
        """Get most similar words to a given word."""
        if word not in self.word_to_idx:
            print(f"Word '{word}' not found in vocabulary")
            return []

        word_idx = self.word_to_idx[word]
        similarities = self.similarity_matrix[word_idx]

        # Sort by similarity (excluding the word itself)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if idx != word_idx:  # Exclude the word itself
                results.append((self.words[idx], similarities[idx]))
                if len(results) >= top_k:
                    break

        return results

def main():
    parser = argparse.ArgumentParser(description='Visualize Word2Vec embeddings')
    parser.add_argument('embeddings_file', help='JSON file with embeddings exported from Rust')
    parser.add_argument('--method', choices=['umap', 'tsne', 'pca'], default='umap',
                       help='Dimensionality reduction method')
    parser.add_argument('--max-words', type=int, default=500,
                       help='Maximum number of words to visualize')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for HTML files')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Threshold for similarity network')

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    viz = Word2VecVisualizer(args.embeddings_file)

    print("\n=== Creating Visualizations ===")

    # 1. 2D Interactive Plot
    print("1. Creating 2D interactive plot...")
    fig_2d = viz.plot_2d_interactive(method=args.method, max_words=args.max_words)
    output_2d = os.path.join(args.output_dir, f'word2vec_2d_{args.method}.html')
    fig_2d.write_html(output_2d)
    print(f"   Saved: {output_2d}")

    # 2. 3D Interactive Plot
    print("2. Creating 3D interactive plot...")
    fig_3d = viz.plot_3d_interactive(method=args.method, max_words=args.max_words)
    output_3d = os.path.join(args.output_dir, f'word2vec_3d_{args.method}.html')
    fig_3d.write_html(output_3d)
    print(f"   Saved: {output_3d}")

    # 3. Similarity Network
    print("3. Creating similarity network...")
    fig_network = viz.plot_similarity_network(
        threshold=args.similarity_threshold,
        max_words=min(100, args.max_words)
    )
    if fig_network:
        output_network = os.path.join(args.output_dir, 'word2vec_network.html')
        fig_network.write_html(output_network)
        print(f"   Saved: {output_network}")

    # 4. Example similarity heatmap with common words
    print("4. Creating sample similarity heatmap...")
    common_words = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'it', 'for']
    available_words = [w for w in common_words if w in viz.word_to_idx]

    if len(available_words) >= 2:
        fig_heatmap = viz.plot_similarity_heatmap(available_words[:10])
        if fig_heatmap:
            output_heatmap = os.path.join(args.output_dir, 'word2vec_heatmap.html')
            fig_heatmap.write_html(output_heatmap)
            print(f"   Saved: {output_heatmap}")

    print(f"\n=== All visualizations saved to '{args.output_dir}' ===")
    print("\nTo explore word relationships interactively:")
    print("  python -c \"from visualize_word2vec import Word2VecVisualizer; viz = Word2VecVisualizer('embeddings.json')\"")
    print("  viz.get_most_similar('word', 10)")
    print("  viz.find_analogies('king', 'man', 'woman', 5)")

if __name__ == '__main__':
    main()
