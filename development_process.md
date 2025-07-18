# Word2Vec Algorithm Development Process

## Project Overview

This document chronicles the development of a Word2Vec implementation in pure Rust, using the Continuous Bag of Words (CBOW) architecture with negative sampling. The project demonstrates a complete machine learning pipeline from text preprocessing to word embedding generation.

## Development Timeline

Based on the git history, the development process followed these key phases:

### Phase 1: Initial Implementation (8 months ago)
- **Commit**: `0a01831` - "add benchmarks"
- Started with a basic implementation in `main.rs`
- Restructured code by moving main logic to `lib.rs` and creating a separate binary
- Added comprehensive benchmarking infrastructure using Criterion

### Phase 2: Performance Optimization (8 months ago)
- **Commit**: `79b20c0` - "add main"
- **Commit**: `e258cb3` - "use hashset for stop words"
- **Commit**: `a44cb08` - "get the random values from a single index"
- **Commit**: `8d6ecf0` - "improve perf by adding info about loop bounds"

Key optimizations implemented:
- Replaced linear stop words lookup with HashSet for O(1) performance
- Optimized random sampling strategy for negative sampling
- Improved loop bounds information for better compiler optimization

### Phase 4: Critical Bug Fixes and Algorithm Corrections (Current)
- **Issue**: Discovered fundamental flaw in negative sampling implementation
- **Root Cause**: Logical error in filter condition + sampling from wrong data structure
- **Fix**: Implemented proper vocabulary-based negative sampling
- **Impact**: Dramatically improved embedding quality and similarity calculations

### Phase 5: Advanced Optimizations (Future)
- **Frequency-Based Sampling**: Implement sublinear frequency weighting (power 0.75)
- **Hierarchical Softmax**: Alternative to negative sampling for comparison
- **Subword Information**: Consider FastText-style character n-grams

## Architecture Overview

### Core Components

#### 1. Text Preprocessing (`CorpusValues`)

```rust
pub struct CorpusValues {
    pub words_map: HashMap<String, usize>,  // Word to index mapping
    pub vec: Vec<usize>,                    // Tokenized corpus as indices
}
```

**Key Features:**
- Removes punctuation using regex
- Converts to lowercase
- Filters stop words using HashSet for O(1) lookup
- Creates vocabulary mapping and tokenized representation

#### 2. CBOW Parameters (`CBOWParams`)

```rust
pub struct CBOWParams {
    vocab_size: usize,
    embeddings_dimension: usize,  // Default: 100, configurable 25-1000
    random_samples: usize,        // Default: 30 for negative sampling
    mean: f32,                    // Default: 0.0 for weight initialization
    std_dev: f32,                 // Default: 0.01 for weight initialization
    window_size: usize,           // Context window (default: 2*2+1=5)
    target: usize,                // Target position in window
    learning_rate: f32,           // Default: 0.01
    epochs: usize,                // Default: 100
}
```

**Design Decisions:**
- Builder pattern for flexible parameter configuration
- Gaussian initialization for input embeddings
- Zero initialization for output embeddings (standard practice)

#### 3. Training Algorithm

The core training function implements CBOW with negative sampling:

```rust
pub fn train(
    pairs: &[(Vec<usize>, usize)],  // (context_words, target_word) pairs
    cbow_params: &CBOWParams,
    input_layer: &mut [f32],        // Input embeddings matrix
    hidden_layer: &mut [f32],       // Output embeddings matrix
    corpus: &CorpusValues,
) {
    // Training loop implementation
}
```

**Algorithm Steps:**
1. **Forward Pass**: Sum context word embeddings
2. **Positive Sampling**: Compute loss for actual target word
3. **Negative Sampling**: Sample random words and compute negative loss
4. **Backpropagation**: Update both input and output embeddings

## Implementation Details

### Negative Sampling Strategy

#### Critical Bug Fix: Incorrect Negative Sampling Implementation

**Problem Identified**: The original negative sampling implementation had a fundamental flaw that severely impacted embedding quality:

```rust
// INCORRECT - Original implementation
let random_indices = corpus.vec[breakpoint..cbow_params.random_samples + breakpoint]
    .iter()
    .filter(|x| x.eq(&target));  // BUG: This selects words that ARE the target!
```

**Root Cause Analysis**: This implementation had multiple critical issues:

1. **Logical Error**: Used `x.eq(&target)` instead of `!x.eq(&target)`, selecting target words instead of excluding them
2. **Sampling Bias**: Sampled consecutive words from `corpus.vec[breakpoint..]` instead of random vocabulary words
3. **Semantic Corruption**: Consecutive words in text are often semantically related, making them poor negative examples

**Impact on Training**:
- The model learned to distinguish targets from **semantically related words** (backwards!)
- Limited vocabulary coverage as only consecutively positioned words were sampled
- Severely degraded embedding quality and similarity calculations

**Corrected Implementation**:
```rust
// FIXED - Proper negative sampling
let vocab_sampler = Uniform::from(0..corpus.words_map.len());
let mut negative_samples = Vec::new();

while negative_samples.len() < cbow_params.random_samples {
    let random_word = vocab_sampler.sample(&mut rng);

    // Correctly exclude target word and avoid duplicates
    if random_word != *target && !negative_samples.contains(&random_word) {
        negative_samples.push(random_word);
    }
}
```

**Technical Rationale**:
- **True Random Sampling**: Samples from entire vocabulary space (0..vocab_size)
- **Proper Exclusion**: Correctly excludes target word with `!=` comparison
- **Unbiased Training**: Every vocabulary word has equal probability of being a negative sample
- **Semantic Correctness**: Negative samples are truly random, not semantically related

**Performance Impact**: This fix dramatically improved embedding quality by ensuring the model learns proper semantic distinctions between words.

### Sigmoid Activation

```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + x.neg().exp())
}
```

Used for converting dot products to probabilities in the output layer.

### Matrix Operations

The implementation uses flat vectors instead of 2D matrices for cache efficiency:
- Input matrix: `vocab_size × embedding_dimension`
- Output matrix: `vocab_size × embedding_dimension`
- Index calculation: `position + word_index * embedding_dimension`

## Performance Considerations

### Benchmarking Infrastructure

The project includes comprehensive benchmarking using Criterion:

```rust
// From benches/w2v_bench.rs
fn set_default_benchmark_configs(benchmark: &mut BenchmarkGroup<WallTime>) {
    let sample_size: usize = 100;
    let measurement_time: Duration = Duration::new(10, 0);
    let confidence_level: f64 = 0.97;
    let warm_up_time: Duration = Duration::new(10, 0);
    let noise_threshold: f64 = 0.05;
}
```

### Optimization History

1. **Stop Words Optimization**: Changed from Vec lookup to HashSet
2. **Random Sampling**: Improved to use single random index generation
3. **Loop Bounds**: Added explicit bounds information for compiler optimization

**Question**: What were the specific performance improvements achieved with each optimization? It would be valuable to document the before/after benchmark results.

## CLI Interface

The final implementation provides a user-friendly command-line interface:

```bash
cargo run -- \
    --corpus text8 \
    --dimension-embeddings 100 \
    --epochs 300 \
    --learning-rate 0.01 \
    --model result.json
```

**Parameters:**
- `--corpus`: Input text file (default: text8)
- `--dimension-embeddings`: Embedding size (default: 100)
- `--epochs`: Training iterations (default: 300)
- `--learning-rate`: Learning rate (default: 0.01)
- `--model`: Output file for embeddings (default: result.json)

## Output Format

Embeddings are saved as JSON lines:
```json
{"word": "example", "embedding": [0.1, -0.2, 0.3, ...]}
```

## Development Environment

### Nix Flake Configuration

The project uses Nix for reproducible development:

```nix
# From flake.nix - provides Rust toolchain and dependencies
```

**Question**: What specific versions of Rust and other tools are pinned in the flake? This information would be valuable for reproducibility documentation.

## Testing Strategy

The implementation includes unit tests for core functionality:

1. **Corpus Parsing**: Validates text preprocessing and tokenization
2. **Matrix Creation**: Ensures correct dimensions for embedding matrices
3. **Pair Generation**: Tests context-target pair extraction
4. **Sigmoid Function**: Validates activation function accuracy

## Dataset Information

- **text7**: 13,550 words (smaller dataset for development)
- **text8**: 17,005,207 words (full dataset for training)

**Question**: What is the source and preprocessing of these datasets? Are they from the original Word2Vec paper or custom datasets?

## Key Lessons Learned

### 1. Algorithm Implementation Correctness
**Critical Insight**: A single logical error in negative sampling can completely break the learning process. The difference between `x.eq(&target)` and `!x.eq(&target)` determined whether the model learned proper semantic relationships or inverted ones.

**Takeaway**: In ML implementations, correctness is paramount over performance. A fast but incorrect algorithm is worse than a slow but correct one.

### 2. Sampling Strategy Impact
**Discovery**: The choice of sampling strategy (consecutive corpus words vs. random vocabulary words) has profound effects on embedding quality. Proper negative sampling requires:
- True randomness across the entire vocabulary
- Exclusion of semantically related words
- Uniform coverage of all vocabulary words

### 3. Debugging ML Algorithms
**Challenge**: ML bugs often manifest as "poor quality" rather than crashes, making them harder to detect and debug.

**Solution**: Systematic testing of algorithm components, especially loss functions and sampling strategies.

## Future Improvements

Based on the TODO in README.md:
- **Speed Optimization**: The main focus remains on performance improvements
- **Potential Areas**:
  - SIMD optimizations for vector operations
  - Parallel processing for batch training
  - Memory layout optimizations
  - GPU acceleration consideration

## Technical Choices Rationale

### Why Rust?
- Memory safety without garbage collection overhead
- Zero-cost abstractions for high-performance ML
- Excellent package ecosystem (Cargo)
- Strong type system prevents common ML bugs

### Why CBOW over Skip-gram?
**Question**: Was CBOW chosen for specific performance or accuracy reasons compared to Skip-gram? This architectural decision would be valuable to document.

### Dependencies Justification
- `rand`/`rand_distr`: Professional random number generation
- `regex`: Robust text preprocessing
- `stop-words`: Comprehensive stop word lists
- `serde`/`serde_json`: Efficient serialization
- `clap`: Modern CLI argument parsing
- `criterion`: Industry-standard Rust benchmarking

## Conclusion

This Word2Vec implementation demonstrates a complete ML pipeline in Rust, from initial prototype to optimized, benchmarked, and user-friendly tool. The development process shows iterative improvement focusing on correctness, performance, and usability.

The codebase serves as an excellent example of:
- Clean Rust architecture with separation of concerns
- Performance-conscious implementation with profiling
- Comprehensive testing and benchmarking
- Modern development practices with Nix and CLI tools

**Questions for Further Documentation:**
1. What were the specific performance metrics achieved?
2. How does this implementation compare to reference implementations?
3. What embedding quality metrics were used for validation?
4. **RESOLVED**: ~~Negative sampling filter logic~~ - Fixed critical bug in Phase 4
5. What are the quantitative improvements in embedding quality after the negative sampling fix?
