use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use pprof::criterion::{Output, PProfProfiler};
use std::{fs::OpenOptions, io::Read, time::Duration};
use word2vec::{parse_corpus, train, CBOWParams, CorpusValues};

fn set_default_benchmark_configs(benchmark: &mut BenchmarkGroup<WallTime>) {
    let sample_size: usize = 100;
    let measurement_time: Duration = Duration::new(10, 0);
    let confidence_level: f64 = 0.97;
    let warm_up_time: Duration = Duration::new(10, 0);
    let noise_threshold: f64 = 0.05;

    benchmark
        .sample_size(sample_size)
        .measurement_time(measurement_time)
        .confidence_level(confidence_level)
        .warm_up_time(warm_up_time)
        .noise_threshold(noise_threshold);
}

fn get_corpus(file_path: &str) -> String {
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

fn bench_text_processing() {
    let raw_corpus = get_corpus("text7");
    parse_corpus(raw_corpus);
}

fn bench_training(
    corpus: &CorpusValues,
    random_samples: usize,
    embeddings_dimension: usize,
    epochs: usize,
) {
    let cbow_params = CBOWParams::new(corpus.words_map.len())
        .set_random_samples(random_samples)
        .set_embeddings_dimension(embeddings_dimension)
        .set_epochs(epochs)
        .set_learning_rate(0.01);
    let pairs = cbow_params.generate_pairs(&corpus.vec);
    let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
    train(
        &pairs,
        &cbow_params,
        &mut input_layer,
        &mut hidden_layer,
        corpus,
    );
}

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("w2v");
    set_default_benchmark_configs(&mut benchmark);

    let raw_corpus = get_corpus("text7");
    let corpus = parse_corpus(raw_corpus);

    let params: [(usize, usize, usize); 3] = [(5, 25, 5), (10, 100, 50), (15, 200, 100)];

    for (random_samples, embeddings_dimension, epochs) in params {
        benchmark.bench_function(
            BenchmarkId::new(
                "training",
                format!(
                    "random_samples-{}-embeddings_dimension-{}-epochs-{}",
                    random_samples, embeddings_dimension, epochs
                ),
            ),
            |bencher| {
                bencher.iter(|| {
                    bench_training(
                        black_box(&corpus),
                        random_samples,
                        embeddings_dimension,
                        epochs,
                    )
                });
            },
        );
    }

    benchmark.bench_function(BenchmarkId::new("text-processing", "text7"), |bencher| {
        bencher.iter(|| bench_text_processing());
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(500, Output::Flamegraph(None)));
    targets = bench
}

criterion_main!(benches);
