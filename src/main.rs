use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::Write,
};
use word2vec::{parse_corpus, train, CBOWParams};

fn save_changes(file_path: &str, values: Vec<Value>) {
    let mut file = open_or_create_file(file_path);
    file.set_len(0).unwrap();
    file.write_all(serde_json::to_string_pretty(&values).unwrap().as_bytes())
        .expect("something");
}

fn open_or_create_file(file_path: &str) -> File {
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
    {
        Ok(value) => value,
        Err(e) => panic!("Problem creating the file: {:?}", e),
    }
}

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

fn main() {
    let raw_corpus = "Today we will be learning about the fundamentals of data science and statistics. Data Science and statistics are hot and growing fields with alternative names of machine learning, artificial intelligence, big data, etc. I'm really excited to talk to you about data science and statistics because data science and statistics have long been a passions of mine. I didn't used to be very good at data science and statistics but after studying data science and statistics for a long time, I got better and better at it until I became a data science and statistics expert. I'm really excited to talk to you about data science and statistics, thanks for listening to me talk about data science and statistics.".to_string();

    let corpus = parse_corpus(raw_corpus);
    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_embeddings_dimension(100)
        .set_epochs(300)
        .set_learning_rate(0.01);
    let pairs = cbow_params.generate_pairs(&corpus.vec);
    let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
    train(
        &pairs,
        &cbow_params,
        &mut input_layer,
        &mut hidden_layer,
        &corpus,
    );

    let values = corpus
        .words_map
        .into_iter()
        .map(|(k, v)| generate_result(&k, &v, &input_layer, cbow_params.embeddings_dimension()))
        .collect();

    save_changes("result.json", values)
}
