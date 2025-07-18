# Word2Vec Visualization Guide

## Overview

This guide shows you how to visualize and explore your Word2Vec embeddings using the `visualize_word2vec.py` script. You can create interactive 2D/3D plots, similarity networks, and heatmaps to understand how your model represents word relationships.

## 🚀 Quick Start

### 1. Train a Model
```bash
# Train with your corpus
cargo run --release -- train --corpus lee_background.cor --epochs 100 --dimension-embeddings 100

# Quick test with fewer epochs
cargo run --release -- train --epochs 10 --dimension-embeddings 50
```

### 2. Export for Visualization
```bash
# Export embeddings to JSON
cargo run -- visualize --model model.bin --output embeddings.json --max-words 500
```

### 3. Create Interactive Visualizations
```bash
# Generate all visualization types
python visualize_word2vec.py embeddings.json --max-words 300 --method umap

# Use different dimensionality reduction methods
python visualize_word2vec.py embeddings.json --method tsne --max-words 200
python visualize_word2vec.py embeddings.json --method pca --max-words 400
```

## 📊 Visualization Types

### 1. **2D Interactive Scatter Plot**
- **Best for**: Quick overview of word clusters
- **Shows**: How words group together semantically
- **Features**: Hover tooltips, labeled points, zoom/pan
- **Output**: `visualizations/word2vec_2d_umap.html`

### 2. **3D Interactive Plot**
- **Best for**: Immersive exploration of word relationships
- **Shows**: More detailed spatial relationships
- **Features**: 3D rotation, zoom, hover details
- **Output**: `visualizations/word2vec_3d_umap.html`

### 3. **Similarity Network**
- **Best for**: Understanding direct word connections
- **Shows**: Words connected by similarity above threshold
- **Features**: Node size = number of connections, interactive
- **Output**: `visualizations/word2vec_network.html`

### 4. **Similarity Heatmap**
- **Best for**: Comparing specific sets of words
- **Shows**: Pairwise similarity matrix
- **Features**: Color-coded similarity values
- **Output**: `visualizations/word2vec_heatmap.html`

## ⚙️ Script Parameters

### visualize_word2vec.py Options
```bash
python visualize_word2vec.py embeddings.json [OPTIONS]

# Required argument:
  embeddings.json          # JSON file exported from Rust

# Optional parameters:
  --method {umap,tsne,pca} # Dimensionality reduction method (default: umap)
  --max-words N            # Maximum words to visualize (default: 500)
  --output-dir DIR         # Output directory for HTML files (default: visualizations)
  --similarity-threshold T # Threshold for network connections (default: 0.7)
```

### Examples
```bash
# Basic usage with UMAP
python visualize_word2vec.py embeddings.json

# Use t-SNE for different clustering perspective
python visualize_word2vec.py embeddings.json --method tsne --max-words 300

# Lower threshold for denser similarity network
python visualize_word2vec.py embeddings.json --similarity-threshold 0.5

# High-resolution visualization with more words
python visualize_word2vec.py embeddings.json --max-words 1000 --method umap
```

## 🔍 Understanding the Visualizations

### 2D/3D Scatter Plots
- **Points**: Each point represents a word
- **Distance**: Closer points = more similar words
- **Clusters**: Groups of related words
- **Colors**: Distinguish different regions of the space

### Similarity Network
- **Nodes**: Words in your vocabulary
- **Edges**: Similarity connections above threshold
- **Node Size**: Number of connections (degree)
- **Isolated Nodes**: Words with unique meanings or rare usage

### Heatmap
- **Axes**: Word lists (X and Y)
- **Color Intensity**: Similarity strength (red = high, blue = low)
- **Diagonal**: Always 1.0 (words similar to themselves)

## 🎯 What to Look For

### **Good Embeddings Show:**
1. **Semantic Clustering**: Similar words cluster together
   - Animals group with animals
   - Colors form their own cluster
   - Verbs cluster by action type

2. **Analogical Relationships**: Vector arithmetic works
   - king - man + woman ≈ queen
   - Paris - France + Italy ≈ Rome

3. **Smooth Transitions**: Gradual similarity changes in space
4. **Meaningful Neighborhoods**: Words have sensible "neighbors"

### **Common Patterns:**
- **Function words** (the, and, of) often cluster together
- **Content words** form semantic groups (animals, colors, actions)
- **Antonyms** may be close (good/bad) or distant depending on training context

## 🔧 Troubleshooting

### Poor Clustering/Visualization
1. **Increase training epochs**: More training improves embedding quality
2. **Larger embedding dimension**: Try 200-300 dimensions for complex vocabularies
3. **More training data**: Larger corpus provides better word relationships
4. **Adjust learning rate**: Lower rates (0.01-0.001) for more stable learning

### Visualization Performance Issues
1. **Reduce max_words**: Use fewer words for faster rendering
2. **Use PCA**: Fastest dimensionality reduction method
3. **Lower similarity threshold**: Reduces network complexity

### Network Visualization Problems
- **No edges**: Lower the similarity threshold (try 0.3-0.5)
- **Too dense**: Increase threshold or reduce max_words
- **Disconnected**: Normal for diverse vocabularies

## 🚀 Advanced Usage

### Interactive Exploration
After generating visualizations, you can explore them interactively:

1. **Open HTML files** in your browser from the `visualizations/` directory
2. **Hover over points** to see word labels and similarity values
3. **Zoom and pan** to explore different regions
4. **Use browser developer tools** to inspect underlying data

### Custom Word Lists
You can modify the script to focus on specific word categories:

```python
# In visualize_word2vec.py, replace the common_words list
custom_words = ['happy', 'sad', 'angry', 'excited', 'calm', 'worried']
fig_heatmap = viz.plot_similarity_heatmap(custom_words)
```

### Programmatic Access
Use the Word2VecVisualizer class directly:

```python
from visualize_word2vec import Word2VecVisualizer

viz = Word2VecVisualizer('embeddings.json')

# Find similar words
similar = viz.get_most_similar('king', 10)
print(f"Words similar to 'king': {similar}")

# Try analogies
analogies = viz.find_analogies('king', 'man', 'woman', 5)
print(f"king - man + woman = {analogies}")
## 📋 Output Files

After running the visualization script, you'll find these files in the `visualizations/` directory:

- **`word2vec_2d_umap.html`**: Interactive 2D scatter plot
- **`word2vec_3d_umap.html`**: Interactive 3D scatter plot
- **`word2vec_network.html`**: Similarity network graph
- **`word2vec_heatmap.html`**: Similarity heatmap for common words

All files are self-contained HTML that can be opened in any web browser.

## 📈 Interpretation Guide

### **Similarity Scores:**
- **0.8-1.0**: Very similar (synonyms, related forms)
- **0.6-0.8**: Moderately similar (same category)
- **0.4-0.6**: Somewhat related
- **0.0-0.4**: Different meanings
- **Negative**: Potentially opposite meanings

### **Word Analogies:**
Good analogies indicate your model learned relationships:
- **Grammatical**: walk/walked, big/bigger
- **Semantic**: king/queen, Paris/France
- **Functional**: hammer/nail, pen/paper

## 🔗 Integration with Query Tool

Remember you also have the interactive query tool:
```bash
cargo run --release -- query --model model.bin
```

This gives you a command-line interface to:
- Find similar words: `similar word 10`
- Calculate similarities: `similarity word1 word2`
- View embeddings: `embedding word`
- Browse vocabulary: `vocab`

## 🎨 Visualization Tips

### **For Best Results:**
1. **Train longer**: More epochs produce cleaner clusters
2. **Use larger corpus**: More context improves relationships
3. **Experiment with dimensions**: 100-300 works well for most cases
4. **Try different reduction methods**: UMAP (best overall), t-SNE (local structure), PCA (global structure)

### **Performance Optimization:**
- Start with `--max-words 200` for quick exploration
- Use `--method pca` for fastest results
- Increase words gradually as needed

## 🐛 Common Issues & Solutions

### **"No edges found" in network:**
```bash
# Lower the similarity threshold
python visualize_word2vec.py embeddings.json --similarity-threshold 0.3
```

### **Visualization too slow:**
```bash
# Reduce word count
python visualize_word2vec.py embeddings.json --max-words 100
```

### **Poor clustering:**
- Increase training epochs in your Rust model
- Use larger embedding dimensions
- Ensure sufficient training data

## 🏁 Complete Workflow Example

```bash
# 1. Train a model
cargo run --release -- train --corpus text8 --epochs 100 --dimension-embeddings 200

# 2. Export embeddings
cargo run --release -- visualize --model model.bin --output embeddings.json

# 3. Create visualizations
python visualize_word2vec.py embeddings.json --max-words 500 --method umap

# 4. Open visualizations/word2vec_2d_umap.html in your browser

# 5. Explore interactively
cargo run --release -- query --model model.bin
```

Your Word2Vec model is now fully visualized and ready for exploration! 🚀
- **Semantic**: king/queen, Paris/France
- **Functional**: car/drive, pen/write

## 🚨 Troubleshooting

### **No Clear Clusters?**
- Increase training epochs
- Use larger corpus
- Adjust window size in CBOW parameters

### **Python Dependencies Issues?**
```bash
pip install numpy pandas plotly scikit-learn umap-learn networkx matplotlib
```

### **Visualization Too Slow?**
- Reduce `--max-words` parameter
- Use PCA instead of UMAP/t-SNE for faster computation

### **Empty Network Graph?**
- Lower `--similarity-threshold`
- Check if words are actually similar in your corpus

## 🎨 Advanced Usage

### Custom Word Lists for Heatmaps
```python
from visualize_word2vec import Word2VecVisualizer
viz = Word2VecVisualizer('embeddings.json')

# Create heatmap for specific words
words = ['good', 'bad', 'excellent', 'terrible', 'amazing']
fig = viz.plot_similarity_heatmap(words)
fig.show()
```

### Programmatic Analysis
```python
# Load and analyze programmatically
viz = Word2VecVisualizer('embeddings.json')

# Find similar words
similar = viz.get_most_similar('computer', 10)
print(similar)

# Test analogies
analogies = viz.find_analogies('king', 'man', 'woman', 5)
print(analogies)
```

## 🎯 Model Quality Assessment

Use these commands to evaluate your model:

```bash
# In explorer.py interactive mode:
similar king 10          # Should show: queen, royal, prince, etc.
analogy king man woman   # Should show: queen at top
similarity cat dog       # Should be moderate-high (0.4-0.7)
similarity car airplane  # Should be lower than cat/dog
```

## 📁 File Outputs

- `embeddings.json` - Exported word embeddings
- `visualizations/` - HTML interactive plots
- `word2vec_*.png` - Static matplotlib plots
- `model.bin` - Trained model (Rust binary format)

## 🔄 Workflow Example

1. **Train**: `cargo run --release -- train --epochs 100`
2. **Export**: `cargo run --release -- visualize`
3. **Explore**: `python explorer.py embeddings.json`
4. **Visualize**: `python visualize_word2vec.py embeddings.json`
5. **Analyze**: Check clusters, test analogies, examine similarities
6. **Iterate**: Adjust training parameters based on results

Happy exploring! 🎉
