use clap::{Parser, Subcommand};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, IoSlice, Read, Write},
    path::PathBuf,
    sync::{mpsc::channel, Arc},
    thread,
    time::Instant,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use word2vec::{
    algo::{parse_corpus, train, train_parallel, CBOWParams},
    model::{cosine_similarity, Word2VecModel},
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
        #[arg(short, long, default_value = "5")]
        window: usize,
        #[arg(long, default_value = "5")]
        negative: usize,
        #[arg(long, default_value = "5")]
        min_count: usize,
        #[arg(long, default_value = "1")]
        cbow: usize, // 1 for CBOW, 0 for skip-gram (not implemented)
        #[arg(long, default_value = "1e-3")]
        sample: f32,
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
        #[arg(short, long, default_value = "true")]
        parallel: bool,
    },
    /// Query a trained model for word similarities
    Query {
        #[arg(short, long, default_value = "model.bin")]
        model: PathBuf,
    },
    /// Compare two models
    Compare {
        #[arg(short, long, default_value = "model.bin")]
        rust_model: PathBuf,
        #[arg(short, long, default_value = "c_model.txt")]
        c_model: PathBuf,
        #[arg(long)]
        max_words: Option<usize>,
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
            window,
            negative,
            min_count,
            cbow,
            sample,
            model,
            parallel,
        } => {
            train_model(
                corpus,
                dimension_embeddings,
                epochs,
                learning_rate,
                window,
                negative,
                min_count,
                cbow,
                sample,
                model,
                parallel,
            );
        }
        Commands::Query { model } => {
            query_model(model);
        }
        Commands::Compare {
            rust_model,
            c_model,
            max_words,
        } => {
            compare_models(rust_model, c_model, max_words);
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
    window: usize,
    negative: usize,
    min_count: usize,
    cbow: usize,
    sample: f32,
    model_path: PathBuf,
    parallel: bool,
) {
    let log_name = format!(
        "word2vec_train_{}_{}_{}_{}.log",
        corpus_path.to_str().unwrap(),
        dimension_embeddings,
        epochs,
        learning_rate
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
    println!("  Window size: {}", window);
    println!("  Negative samples: {}", negative);
    println!("  Min count: {}", min_count);
    println!("  CBOW: {}", cbow);
    println!("  Sample: {}", sample);
    println!("  Model output: {:?}", model_path);

    let start = Instant::now();
    let corpus = parse_corpus(get_corpus(&corpus_path));
    let duration = start.elapsed();
    println!("Time elapsed in parse_corpus() is: {:?}", duration);

    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_learning_rate(learning_rate)
        .set_embeddings_dimension(dimension_embeddings)
        .set_epochs(epochs)
        .set_window_size(window)
        .set_random_samples(negative);
    let pairs = cbow_params.generate_pairs(&corpus.vec);
    let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
    let duration = start.elapsed();
    println!("Time elapsed in create_matrices() is: {:?}", duration);

    if parallel {
        train_parallel(
            &pairs,
            &cbow_params,
            &mut input_layer,
            &mut hidden_layer,
            &corpus,
        );
    } else {
        train(
            &pairs,
            &cbow_params,
            &mut input_layer,
            &mut hidden_layer,
            &corpus,
        );
    }

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

enum Msg {
    Data(String),
    End,
}

fn compare_models(rust_model_path: PathBuf, c_model_path: PathBuf, max_words: Option<usize>) {
    println!("Loading model from {:?}...", rust_model_path);
    let rust_model = Arc::new(Word2VecModel::load(&rust_model_path));

    println!("Model loaded successfully!");

    println!("Loading model from {:?}...", c_model_path);
    let c_model = Arc::new(Word2VecModel::load(&c_model_path));

    println!("Model loaded successfully!");

    // let mut export_data = serde_json::Map::new();

    let mut output_file = File::create("comparaison.csv").unwrap();

    writeln!(output_file, "word,word2,c_score,rust_score").unwrap();

    let mut words: Vec<&String> = rust_model.vocab.keys().collect();
    if let Some(max) = max_words {
        words.truncate(max);
    }
    // Use BufWriter for efficient file writing
    let output_file = File::create("comparaison.csv").unwrap();
    let mut writer = BufWriter::new(output_file);
    writeln!(writer, "word,word2,c_score,rust_score").unwrap();

    let (tx, rx) = channel();

    thread::spawn(move || {
        let mut results = Vec::with_capacity(1080);

        loop {
            let Ok(msg) = rx.try_recv() else { continue };
            match msg {
                Msg::End => break,
                Msg::Data(data) => {
                    results.push(data);
                }
            };
            if results.len() >= 1080 {
                let bufs: Vec<_> = results
                    .iter()
                    .map(|s| s.as_bytes())
                    .map(IoSlice::new)
                    .collect();
                writer.write_vectored(&bufs).expect("Failed to write");
                results.clear();
            }
        }
        if !results.is_empty() {
            let bufs: Vec<_> = results
                .iter()
                .map(|s| s.as_bytes())
                .map(IoSlice::new)
                .collect();
            writer.write_vectored(&bufs).expect("Failed to write");
            writer.flush().expect("Failed to flush");
        }
    });

    // Use rayon for parallelism
    use rayon::prelude::*;
    words.par_iter().enumerate().for_each(|(i, &word1)| {
        for word2 in words.iter().skip(i) {
            let Some(rust_embedding1) = rust_model.get_embedding(word1) else {
                continue;
            };
            let Some(c_embedding1) = c_model.get_embedding(word1) else {
                continue;
            };

            let Some(rust_embedding2) = rust_model.get_embedding(word2) else {
                continue;
            };

            let Some(c_embedding2) = c_model.get_embedding(word2) else {
                continue;
            };

            let c_score = cosine_similarity(c_embedding1, c_embedding2);
            let rust_score = cosine_similarity(rust_embedding1, rust_embedding2);

            let data = format!("{word1},{word2},{c_score},{rust_score}\n");
            tx.send(Msg::Data(data)).unwrap();
        }
    });
    tx.send(Msg::End).unwrap();

    // let json =
    //     serde_json::to_string_pretty(&export_data).expect("Failed to serialize embeddings to JSON");
    // let mut file = OpenOptions::new()
    //     .write(true)
    //     .create(true)
    //     .truncate(true)
    //     .open("comparaison.json")
    //     .expect("Failed to create output file");
    // file.write_all(json.as_bytes())
    //     .expect("Failed to write embeddings to file");

    println!("Embeddings exported successfully!");
}
