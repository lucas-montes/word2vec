use clap::Parser;
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions}, io::{Read, Write}, path::{Path, PathBuf}, time::Instant
};
use word2vec::{parse_corpus, train, CBOWParams};

fn generate_result(
    word: &str,
    index: &usize,
    embeddings: &[f32],
    embeddings_dimension: usize,
) -> Value {
    let embedding: Vec<f32> = (0..embeddings_dimension)
        .map(|position| embeddings[position + index * embeddings_dimension])
        .collect();
    json!({"word": word, "embedding":embedding })
}

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
#[command(name = "Word2Vev")]
struct Cli {
    #[arg(short, long, default_value = "text8")]
    corpus: PathBuf,
    #[arg(short, long, default_value = "100")]
    dimension_embeddings: usize,
    #[arg(short, long, default_value = "300")]
    epochs: usize,
    #[arg(short, long, default_value = "0.01")]
    learning_rate: f32,
    #[arg(short, long, default_value = "result.json")]
    model: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    println!("Running with parameters: {:?}", cli);
    let start = Instant::now();
    let corpus = parse_corpus(get_corpus(&cli.corpus));
    let duration = start.elapsed();
    println!("Time elapsed in parse_corpus() is: {:?}", duration);

    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_embeddings_dimension(cli.dimension_embeddings)
        .set_epochs(cli.epochs)
        .set_learning_rate(cli.learning_rate);
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

    let values = corpus
        .words_map
        .into_iter()
        .map(|(k, v)| generate_result(&k, &v, &input_layer, cbow_params.embeddings_dimension()));

    let mut file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&cli.model)
    {
        Ok(value) => value,
        Err(e) => panic!("Problem creating the file: {:?}", e),
    };
    file.set_len(0).unwrap();

    for value in values {
        file.write_all(serde_json::to_string(&value).unwrap().as_bytes())
            .expect("something");
        file.write(b"\n").expect("something");
    }
}
