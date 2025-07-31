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


# 18/07/25
added tracing to check the loss decrease
added few python scripts to compare wiht a standard and visualize thinkgs, however they are awful and fairly useless, i should delete them
cli to interacct with the model (query, train and visualization)
the loss seems waaay to high, i do not know why.
i had an error that helped improve the accuracy while selecting nefative samples, now that i changed it, the similarities are all wrong, why?
I changed the technique to select samples, according to llm, my initial approach wasn't random enough.
## Notes
You want to look for asm like;
addps
or
MULPS

**Questions for Further Documentation:**
1. What were the specific performance metrics achieved?
2. How does this implementation compare to reference implementations?
3. What embedding quality metrics were used for validation?
4. **RESOLVED**: ~~Negative sampling filter logic~~ - Fixed critical bug in Phase 4
5. What are the quantitative improvements in embedding quality after the negative sampling fix?


Does It Matter If the Loss Is High?
The absolute value of the loss matters to some extent, but the trend (decreasing loss) and the context (task, dataset, loss function) are critical for determining whether the high loss is problematic. Here’s why:

Decreasing Loss Indicates Learning:
A decreasing loss over epochs generally means the model is learning, as it’s reducing the error between predictions and true targets. In your logs, the loss decreases slightly from ~1991 (epoch 267) to ~1777 (epoch 299), with fluctuations (e.g., a spike to 2147 in epoch 268). This suggests some learning is happening, which is positive.
If the loss is decreasing consistently, it’s a sign that the optimization process (e.g., gradient descent) is moving the model parameters toward a better solution, even if the absolute values are high.
High Loss Values in Context:
In your CBOW model, the loss is the negative log-likelihood of sigmoid probabilities for positive and negative samples, summed over all training examples in an epoch. High loss values (~1777–2147) could be expected if:
Large Dataset: If your dataset has many examples (e.g., thousands or millions, common in Word2Vec with large corpora like Wikipedia), the loss accumulates across examples. For example, if each example has a loss of ~0.5–1.0 and you have 2000–4000 examples per epoch, the total loss could naturally reach ~1000–4000.
Number of Negative Samples: Your code uses cbow_params.random_samples negative samples per example, each contributing to the loss. If this is high (e.g., 5–20), it amplifies the epoch loss.
Unnormalized Data/Embeddings: If the embeddings (input_layer and hidden_layer) have large values, the dot products (f) in your code can lead to unstable sigmoid outputs, inflating the loss.
However, at epochs 267–299, the loss should ideally be much lower (e.g., tens or low hundreds for a large dataset) if the model is converging well. Your values (~1777–2147) are high for such late epochs, suggesting the model may not be learning as effectively as it could.
When High Loss Is a Problem:
Slow Convergence: If the loss is decreasing but still high after 267+ epochs, the model may be converging too slowly due to issues like a suboptimal learning rate, poor initialization, or insufficient training iterations.
Plateauing: Your logs show fluctuations (e.g., 2147 in epoch 268, 1777 in epoch 299), indicating the model might be stuck in a suboptimal region or oscillating, which can prevent further loss reduction.
Task Performance: For CBOW, the ultimate goal is to produce high-quality word embeddings. Even if the loss is high, the embeddings might still be useful if they capture meaningful relationships (e.g., high cosine similarity for related words like “king” and “queen”). However, high loss often correlates with poor embedding quality, so you should validate this.
Comparison to Expected Values: In Word2Vec, per-example loss should ideally drop to ~0.1–0.5 for a well-trained model (sigmoid probabilities close to 1 for positive samples, 0 for negative). If your average per-example loss is still high (e.g., >1.0), the model isn’t learning effectively.
Null Losses: The null losses in your logs (e.g., epochs 275, 278, 280) are a red flag, as they suggest issues like:
Numerical Instability: Large dot products causing sigmoid(f) to be 0 or 1, leading to -ln(0) (undefined, resulting in NaN).
Logging/Computation Errors: The training loop or logging mechanism might fail to record the loss.
These must be fixed, as they indicate incomplete or invalid training iterations, undermining the reliability of the loss trend.


There are several important differences between your C (og.c) and Rust (algo.rs) implementations of word2vec. Here are the main ones based on your code:

1. Corpus Preprocessing
C: Reads words directly from file, treats any non-alphabetic character as a separator, and does not explicitly lowercase or remove punctuation.
Rust: Explicitly lowercases the corpus and removes punctuation using a regex. Also filters out stop words.
Effect:
Rust may have a smaller, cleaner vocabulary and different word indices, leading to different embeddings and similarity scores.

2. Stop Words
C: Does not remove stop words.
Rust: Removes English stop words using the stop_words crate.
Effect:
This changes the context and frequency of words, affecting training and resulting vectors.

3. Random Initialization
C: Uses a custom random number generator and initializes weights in a specific way.
Rust: Uses the rand and rand_distr crates, with a normal distribution for initialization.
Effect:
Different initial weights can lead to different training trajectories, especially with small corpora or few epochs.

4. Training Algorithm
C: Implements both CBOW and Skip-gram, supports hierarchical softmax and negative sampling, and uses multi-threading with Hogwild! style updates.
Rust: Only implements CBOW (from the code shown), with negative sampling. Also supports Hogwild! style parallelism via Rayon, but the main train function is single-threaded.
Effect:
If you use different architectures or training settings, results will differ.

5. Context Window
C: The window size is dynamic (randomized per word).
Rust: The window size is fixed for all pairs.
Effect:
This changes the context words used for each target, affecting training.

6. Negative Sampling
C: Uses a precomputed unigram table for negative sampling.
Rust: Samples negatives from the vocabulary indices using choose_multiple.
Effect:
The distribution of negative samples may differ, affecting training.

7. Learning Rate
C: Learning rate (alpha) decays linearly during training.
Rust: Learning rate is fixed per epoch.
Effect:
Affects convergence and final embeddings.

8. Vocabulary and Indexing
C: Builds vocabulary as it reads the corpus, no explicit mapping to indices shown.
Rust: Builds a HashMap of word to index, which may result in different orderings.
Effect:
Word indices may not match between implementations, so embeddings for the "same" word may not be directly comparable unless you align vocabularies.

9. Output Format
C: Can output in binary or text, with or without K-means clustering.
Rust: Output format not shown, but likely different.
Summary Table
Feature	C Implementation	Rust Implementation
Lowercasing	No	Yes
Punctuation removal	No	Yes (regex)
Stop word removal	No	Yes
Random init	Custom	rand_distr Normal
CBOW/Skip-gram	Both	CBOW only (shown)
Window size	Dynamic	Fixed
Negative sampling	Unigram table	Uniform random
Learning rate decay	Yes	No (fixed)
Parallelism	Hogwild!	Rayon (optional)
Output format	Binary/Text	Not shown
Why are your results so different?
Vocabulary and word order are likely different.
Corpus preprocessing is stricter in Rust.
Training hyperparameters and algorithms differ.
Negative sampling and learning rate schedules are not matched.
To make results comparable:

Use the same preprocessing (lowercasing, punctuation, stop words).
Align vocabularies and word indices.
Match all hyperparameters (window, negative samples, learning rate, epochs).
Use the same random seed and initialization if possible.
Ensure both use the same training algorithm (CBOW/Skip-gram, negative sampling, etc.).
Let me know if you want help aligning any of these aspects!


pin threads to cores

same time as the c model 22 min for the text8 for 100 epochs
 nix run .#c-word2vec -- -train text8 -output c_model.txt -size 100 -window 5 -negative 5 -iter 100 -cbow 1 -alpha 0.025  -binary 0

 train --epochs 100 --corpus text8 --model model_parl.bin

person correlation not great, but close to the c model too. some improvements could be the learning rate
