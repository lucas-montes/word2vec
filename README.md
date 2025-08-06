# Word2Vec

The word2vec model in pure rust.


To save a baseline, use `cargo bench --bench w2v_bench -- --save-baseline <name>`. To compare against an existing baseline, use `cargo bench --bench w2v_bench -- --baseline <name>`.

To benchmark
```bash
cargo bench
```

To profile
```bash
cargo bench --bench w2v_bench -- --profile-time 30
```

text8 has 17005207 words

```bash
wc -w text
```

get the asm output
```bash
cargo asm --lib word2vec::algo::train --rust --simplify
```

To run the c model
```bash
 nix run .#c-word2vec -- -train lee_background.cor -output c_model.txt -size 100 -window 5 -negative 5 -iter 3 -cbow 1 -alpha 0.025  -binary 0
```

To run the rust model
```bash
cargo run --release -- train
```

Run the binary and check the help menu to see what optios are available from the CLI
```bash
cargo run
```
