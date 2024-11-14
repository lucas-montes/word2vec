# Word2Vec

The word2vec model in pure rust.

To benchmark
```bash
cargo bench
```

To profile
```bash
cargo bench --bench w2v_bench -- --profile-time 30
```

text7 has 1350 words
text8 has 17005207 words

```bash
wc -w text
```

## TODO
Improve speed.