# Word2Vec

The word2vec model in pure rust.


To save a baseline, use `cargo bench --bench w2v_bench -- --save-baseline <name>`. To compare against an existing baseline, use `cargo bench --bench w2v_bench -- --baseline <name>`. For more on baselines, see below.

To benchmark
```bash
cargo bench
```

To profile
```bash
cargo bench --bench w2v_bench -- --profile-time 30
```

text7 has 13550 words
text8 has 17005207 words

```bash
wc -w text
```

## TODO
Improve speed.