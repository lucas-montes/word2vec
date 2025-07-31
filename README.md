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

get the asm output
```bash
cargo asm --lib word2vec::algo::train --rust --simplify
```

## TODO
Improve speed.


```bash
 nix run .#c-word2vec -- -train lee_background.cor -output c_model.txt -size 100 -window 5 -negative 5 -iter 3 -cbow 1 -alpha 0.025  -binary 0
```

 cargo bench --bench w2v_bench -- --profile-time=30


command for c create vectors
```bash
#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/text8
VECTOR_DATA=$DATA_DIR/text8-vector.bin

if [ ! -e $VECTOR_DATA ]; then
  if [ ! -e $TEXT_DATA ]; then
		sh ./create-text8-data.sh
	fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
fi
```
