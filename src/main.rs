use clap::{Parser, Subcommand};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use word2vec::{
    algo::{parse_corpus, train, CBOWParams},
    model::Word2VecModel,
};

fn get_corpus(file_path: &PathBuf) -> String {
    let mut f = match OpenOptions::new()
        .read(true)
        .write(false)
        .create(false)
        .truncate(false)
        .open(file_path)
    {
        Ok(value) => value,
        Err(e) => panic!("Problem creating the file: {:?}", e),
    };
    let mut corpus = String::new();
    f.read_to_string(&mut corpus).unwrap();
    corpus
}

#[derive(Debug, Parser)]
#[command(name = "Word2Vec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Train a new Word2Vec model
    Train {
        #[arg(short, long, default_value = "lee_background.cor")]
        corpus: PathBuf,
        #[arg(short, long, default_value = "100")]
        dimension_embeddings: usize,
        #[arg(short, long, default_value = "300")]
        epochs: usize,
        #[arg(short, long, default_value = "0.025")]
        learning_rate: f32,
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
    },
    /// Query a trained model for word similarities
    Query {
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
    },
    /// Export model data for visualization
    Visualize {
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
        #[arg(short, long, default_value = "embeddings.json")]
        output: PathBuf,
        #[arg(long, default_value = "500")]
        max_words: usize,
    },
    /// Evaluate model on SimLex-999 dataset
    Evaluate {
        /// Path to trained model
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
        /// Path to SimLex-999.txt file
        #[arg(short, long, default_value = "SimLex-999/SimLex-999.txt")]
        input: PathBuf,
        /// Output file for detailed results
        #[arg(short, long, default_value = "evaluation_results.csv")]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            corpus,
            dimension_embeddings,
            epochs,
            learning_rate,
            model,
        } => {
            train_model(corpus, dimension_embeddings, epochs, learning_rate, model);
        }
        Commands::Query { model } => {
            query_model(model);
        }
        Commands::Visualize {
            model,
            output,
            max_words,
        } => {
            visualize_model(model, output, max_words);
        }
        Commands::Evaluate {
            model,
            input,
            output,
        } => {
            evaluate_model(&model, &input, &output);
        }
    }
}

fn train_model(
    corpus_path: PathBuf,
    dimension_embeddings: usize,
    epochs: usize,
    learning_rate: f32,
    model_path: PathBuf,
) {
    let log_name = format!(
        "word2vec_train_{}_{}_{}_{}.log",
        corpus_path.to_str().unwrap(), dimension_embeddings, epochs, learning_rate
    );
    let file_appender = tracing_appender::rolling::never("logs", log_name);
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(non_blocking)
                .log_internal_errors(true)
                .with_target(false)
                .flatten_event(true)
                .with_span_list(false),
        )
        .init();

    println!("Training model with parameters:");
    println!("  Corpus: {:?}", corpus_path);
    println!("  Embedding dimension: {}", dimension_embeddings);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Model output: {:?}", model_path);

    let start = Instant::now();
    let corpus = parse_corpus(get_corpus(&corpus_path));
    let duration = start.elapsed();
    println!("Time elapsed in parse_corpus() is: {:?}", duration);

    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_embeddings_dimension(dimension_embeddings)
        .set_epochs(epochs)
        .set_learning_rate(learning_rate);
    let pairs = cbow_params.generate_pairs(&corpus.vec);
    let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
    let duration = start.elapsed();
    println!("Time elapsed in create_matrices() is: {:?}", duration);

    train(
        &pairs,
        &cbow_params,
        &mut input_layer,
        &mut hidden_layer,
        &corpus,
    );
    let duration = start.elapsed();
    println!("Time elapsed in train() is: {:?}", duration);

    // Create and save the model
    let model = Word2VecModel::new(corpus.words_map, input_layer, dimension_embeddings);
    model.save(&model_path);
    println!("Model saved to {:?}", model_path);
}

fn query_model(model_path: PathBuf) {
    println!("Loading model from {:?}...", model_path);
    let model = Word2VecModel::load(&model_path);

    println!("Model loaded successfully!");
    println!("Vocabulary size: {}", model.vocab_size());
    println!("Embedding dimension: {}", model.embedding_dim);
    println!("\nAvailable commands:");
    println!("  similarity <word1> <word2>  - Calculate cosine similarity between two words");
    println!("  similar <word> [k]          - Find k most similar words (default k=10)");
    println!("  embedding <word>            - Show word embedding vector");
    println!("  vocab                       - Show first 20 words in vocabulary");
    println!("  quit                        - Exit the program");
    println!();

    loop {
        print!("word2vec> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                let parts: Vec<&str> = input.split_whitespace().collect();
                match parts.as_slice() {
                    ["quit"] | ["exit"] => {
                        println!("Goodbye!");
                        break;
                    }
                    ["similarity", word1, word2] => match model.cosine_similarity(word1, word2) {
                        Some(similarity) => {
                            println!(
                                "Cosine similarity between '{}' and '{}': {:.4}",
                                word1, word2, similarity
                            );
                        }
                        None => {
                            if !model.contains_word(word1) {
                                println!("Word '{}' not found in vocabulary", word1);
                            }
                            if !model.contains_word(word2) {
                                println!("Word '{}' not found in vocabulary", word2);
                            }
                        }
                    },
                    ["similar", word] => {
                        find_similar_words(&model, word, 10);
                    }
                    ["similar", word, k_str] => match k_str.parse::<usize>() {
                        Ok(k) => find_similar_words(&model, word, k),
                        Err(_) => println!("Invalid number: {}", k_str),
                    },
                    ["embedding", word] => match model.get_embedding(word) {
                        Some(embedding) => {
                            println!("Embedding for '{}': {:?}", word, embedding);
                        }
                        None => {
                            println!("Word '{}' not found in vocabulary", word);
                        }
                    },
                    ["vocab"] => {
                        let mut words: Vec<_> = model.vocab.keys().collect();
                        words.sort();
                        println!("First 20 words in vocabulary:");
                        for (i, word) in words.iter().take(20).enumerate() {
                            println!("  {}: {}", i + 1, word);
                        }
                        if words.len() > 20 {
                            println!("  ... and {} more words", words.len() - 20);
                        }
                    }
                    _ => {
                        println!("Unknown command. Available commands:");
                        println!("  similarity <word1> <word2>");
                        println!("  similar <word> [k]");
                        println!("  embedding <word>");
                        println!("  vocab");
                        println!("  quit");
                    }
                }
            }
            Err(error) => {
                println!("Error reading input: {}", error);
                break;
            }
        }
        println!();
    }
}

fn find_similar_words(model: &Word2VecModel, word: &str, k: usize) {
    match model.most_similar(word, k) {
        Some(similar_words) => {
            println!("Top {} words most similar to '{}':", k, word);
            for (i, (similar_word, similarity)) in similar_words.iter().enumerate() {
                println!(
                    "  {}: {} (similarity: {:.4})",
                    i + 1,
                    similar_word,
                    similarity
                );
            }
        }
        None => {
            println!("Word '{}' not found in vocabulary", word);
        }
    }
}

fn evaluate_model(model_path: &PathBuf, simlex_path: &PathBuf, output_path: &PathBuf) {
    println!("Loading model from {:?}...", model_path);

    let model = Word2VecModel::load(model_path);

    println!("Model loaded with {} words", model.vocab_size());

    println!("Evaluating on SimLex-999...");
    match model.evaluate_simlex(simlex_path, output_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error evaluating model: {}", e);
            return;
        }
    };
}

fn visualize_model(model_path: PathBuf, output_path: PathBuf, max_words: usize) {
    println!("Loading model from {:?}...", model_path);
    let model = Word2VecModel::load(&model_path);

    println!("Model loaded successfully!");
    println!(
        "Exporting embeddings to {:?} (max words: {})...",
        output_path, max_words
    );

    // Get most frequent words (first in sorted order, which typically correlates with frequency)
    let mut words: Vec<_> = model.vocab.keys().collect();
    words.sort();

    let mut export_data = serde_json::Map::new();
    let mut embeddings_array = Vec::new();
    let mut words_array = Vec::new();

    for word in words.iter().take(max_words) {
        if let Some(embedding) = model.get_embedding(word) {
            words_array.push(serde_json::Value::String(word.to_string()));
            embeddings_array.push(serde_json::Value::Array(
                embedding
                    .iter()
                    .map(|&x| {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(x as f64)
                                .unwrap_or(serde_json::Number::from(0)),
                        )
                    })
                    .collect(),
            ));
        }
    }

    export_data.insert("words".to_string(), serde_json::Value::Array(words_array));
    export_data.insert(
        "embeddings".to_string(),
        serde_json::Value::Array(embeddings_array),
    );
    export_data.insert(
        "embedding_dim".to_string(),
        serde_json::Value::Number(serde_json::Number::from(model.embedding_dim)),
    );
    export_data.insert(
        "vocab_size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(model.vocab_size())),
    );

    let json =
        serde_json::to_string_pretty(&export_data).expect("Failed to serialize embeddings to JSON");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_path)
        .expect("Failed to create output file");
    file.write_all(json.as_bytes())
        .expect("Failed to write embeddings to file");

    println!("Embeddings exported successfully!");
    println!(
        "Run 'python visualize_word2vec.py {}' to create visualizations",
        output_path.display()
    );
}
