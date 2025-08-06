use criterion::{
     criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use pprof::criterion::{Output, PProfProfiler};
use std::{fs::OpenOptions, io::Read, time::Duration};
use word2vec::algo::{parse_corpus, train, train_parallel, train_parallel_pinned, CBOWParams};

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

const CORPUS: &str = "lee_background.cor";

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

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("w2v-long");
    set_default_benchmark_configs(&mut benchmark);

    let raw_corpus = get_corpus(CORPUS);
    let corpus = parse_corpus(raw_corpus);

    let random_samples = 10;
    let embeddings_dimension = 100;
    let epochs = 100;

    let params_name = format!(
            "random_samples-{random_samples}-embeddings_dimension-{embeddings_dimension}-epochs-{epochs}"
        );

    benchmark.bench_function(
        BenchmarkId::new("parallel-single-thread", &params_name),
        |bencher| {
            let cbow_params = CBOWParams::new(corpus.words_map.len())
                .set_random_samples(random_samples)
                .set_embeddings_dimension(embeddings_dimension)
                .set_epochs(epochs)
                .set_learning_rate(0.01);
            let pairs = cbow_params.generate_pairs(&corpus.vec);
            let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
            bencher.iter(|| {
                train(
                    &pairs,
                    &cbow_params,
                    &mut input_layer,
                    &mut hidden_layer,
                    &corpus,
                )
            });
        },
    );

    benchmark.bench_function(
        BenchmarkId::new("parallel-training", &params_name),
        |bencher| {
            let cbow_params = CBOWParams::new(corpus.words_map.len())
                .set_random_samples(random_samples)
                .set_embeddings_dimension(embeddings_dimension)
                .set_epochs(epochs)
                .set_learning_rate(0.01);
            let pairs = cbow_params.generate_pairs(&corpus.vec);
            let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
            bencher.iter(|| {
                train_parallel(
                    &pairs,
                    &cbow_params,
                    &mut input_layer,
                    &mut hidden_layer,
                    &corpus,
                )
            });
        },
    );

    benchmark.bench_function(
        BenchmarkId::new("parallel-training-pinned", &params_name),
        |bencher| {
            let cbow_params = CBOWParams::new(corpus.words_map.len())
                .set_random_samples(random_samples)
                .set_embeddings_dimension(embeddings_dimension)
                .set_epochs(epochs)
                .set_learning_rate(0.01);
            let pairs = cbow_params.generate_pairs(&corpus.vec);
            let (mut input_layer, mut hidden_layer) = cbow_params.create_matrices();
            bencher.iter(|| {
                train_parallel_pinned(
                    &pairs,
                    &cbow_params,
                    &mut input_layer,
                    &mut hidden_layer,
                    &corpus,
                )
            });
        },
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(500, Output::Flamegraph(None)));
    targets = bench
}

criterion_main!(benches);
