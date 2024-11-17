use word2vec::{parse_corpus, train, CBOWParams};

#[test]
fn test_train() {
    let text = "Today we will be learning about the fundamentals of data science and statistics. Data Science and statistics are hot and growing fields with alternative names of machine learning, artificial intelligence, big data, etc. I'm really excited to talk to you about data science and statistics because data science and statistics have long been a passions of mine. I didn't used to be very good at data science and statistics but after studying data science and statistics for a long time, I got better and better at it until I became a data science and statistics expert. I'm really excited to talk to you about data science and statistics, thanks for listening to me talk about data science and statistics.";

    let corpus = parse_corpus(text.into());

    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_embeddings_dimension(3)
        .set_epochs(10)
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
}
