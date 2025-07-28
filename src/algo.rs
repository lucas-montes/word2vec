use rand::{seq::SliceRandom, thread_rng};
use rand_distr::{Distribution, Normal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ops::Neg,
};

#[derive(Debug)]
pub struct CorpusValues {
    pub words_map: HashMap<String, usize>,
    pub vec: Vec<usize>,
}

impl CorpusValues {
    pub fn new() -> Self {
        Self {
            words_map: HashMap::new(),
            vec: Vec::new(),
        }
    }
    pub fn populate<'a>(mut self, clean_corpus: Cow<'a, str>) -> Self {
        let stop_words = stop_words::get(stop_words::LANGUAGE::English);
        let mut index = 0;
        let sw: HashSet<&str> = HashSet::from_iter(stop_words.iter().map(|s| s.as_str()));
        for word in clean_corpus
            .split_whitespace()
            .filter(|w| w.chars().all(|c| c.is_alphabetic()))
        {
            if !sw.contains(word) {
                if self.words_map.contains_key(word) {
                    let i = self.words_map.get(word).unwrap();
                    self.vec.push(*i);
                } else {
                    //TODO: create the initial embeddings here
                    self.words_map.insert(word.to_string(), index);
                    self.vec.push(index);
                    index += 1;
                };
            }
        }
        self
    }
}

pub struct CBOWParams {
    vocab_size: usize,
    embeddings_dimension: usize, // Test with size 25 to 1000. More corpus more dimension
    random_samples: usize,
    mean: f32,
    std_dev: f32,
    window_size: usize,
    target: usize,
    learning_rate: f32,
    epochs: usize,
}
impl CBOWParams {
    pub fn set_random_samples(mut self, random_samples: usize) -> Self {
        self.random_samples = random_samples;
        self
    }
    pub fn set_embeddings_dimension(mut self, embeddings_dimension: usize) -> Self {
        self.embeddings_dimension = embeddings_dimension;
        self
    }
    pub fn embeddings_dimension(&self) -> usize {
        self.embeddings_dimension
    }
    pub fn set_mean(mut self, mean: f32) -> Self {
        self.mean = mean;
        self
    }
    pub fn set_std_dev(mut self, std_dev: f32) -> Self {
        self.std_dev = std_dev;
        self
    }
    pub fn set_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size * 2 + 1;
        self.target = window_size;
        self
    }

    pub fn set_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    pub fn set_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
    pub fn default() -> Self {
        let window_size = 2;
        Self {
            vocab_size: 0,
            random_samples: 30,
            embeddings_dimension: 100,
            mean: 0.0,
            std_dev: 0.01,
            window_size: window_size * 2 + 1,
            target: window_size,
            learning_rate: 0.01,
            epochs: 100,
        }
    }
    pub fn new(vocab_size: usize) -> Self {
        let mut result = Self::default();
        result.vocab_size = vocab_size;
        result
    }

    pub fn create_matrices(&self) -> (Vec<f32>, Vec<f32>) {
        // set the embeddings_dimension from and type
        let normal = Normal::new(self.mean, self.std_dev).unwrap();
        let mut rng = thread_rng();
        let input_matrix: Vec<f32> = (0..self.vocab_size)
            .flat_map(|_| {
                (0..self.embeddings_dimension)
                    .map(|_| normal.sample(&mut rng))
                    .collect::<Vec<f32>>()
            })
            .collect();
        let output_matrix = vec![0.0; self.vocab_size * self.embeddings_dimension];

        (input_matrix, output_matrix)
    }

    pub fn generate_pairs(&self, corpus: &[usize]) -> Vec<(Vec<usize>, usize)> {
        //TODO: use an iterator instead of collecting
        corpus
            .windows(self.window_size)
            .map(|w| {
                let mut window = w.to_vec();
                let target = window.remove(self.target);
                (window, target)
            })
            .collect()
    }
}

pub fn parse_corpus(mut corpus: String) -> CorpusValues {
    let re = Regex::new(r"[[:punct:]]").unwrap();
    corpus.make_ascii_lowercase();
    let clean_corpus = re.replace_all(&corpus, "");
    CorpusValues::new().populate(clean_corpus)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + x.neg().exp())
}

fn pass(
    context: &[usize],
    target: &usize,
    cbow_params: &CBOWParams,
    input_layer: &mut [f32],
    hidden_layer: &mut [f32],
    neu1: &mut [f32],
    neu1e: &mut [f32],
    epoch_loss: &mut f32,
    rng: &mut rand::rngs::ThreadRng,
    vocab_indices: &[usize],
) {
    // === FORWARD PASS ===
    // pass the input layer to the hidden layer
    for position in 0..neu1.len() {
        let mut f = 0.0;
        for context_index in context {
            //TODO: panic bound
            let i = position + *context_index * cbow_params.embeddings_dimension;
            f += &input_layer[i];
        }
        // neu1[position] = f;
        neu1[position] = f / context.len() as f32;
    }

    // positive sampling
    let target_l2 = target * cbow_params.embeddings_dimension;
    let f = neu1
        .iter()
        .enumerate()
        .map(|(i, v)| v * hidden_layer[i + target_l2])
        .sum::<f32>();

    let sig = sigmoid(f);
    *epoch_loss += -sig.ln();
    let g = (1.0 - sig) * cbow_params.learning_rate;

    for c in 0..neu1e.len() {
        neu1e[c] = g * hidden_layer[c + target_l2];
        hidden_layer[c + target_l2] += g * neu1[c];
    }

    // let breakpoint = between.sample(&mut rng);

    let negative_samples = vocab_indices
        .choose_multiple(rng, cbow_params.random_samples + 1)
        .filter(|&&word_idx| word_idx != *target)
        .take(cbow_params.random_samples);

    // let random_indices = corpus.vec[breakpoint..cbow_params.random_samples + breakpoint]
    //     .iter()
    //     .filter(|x| !x.eq(&target));

    for negative_target in negative_samples {
        let l2 = negative_target * cbow_params.embeddings_dimension;
        let f: f32 = neu1
            .iter()
            .enumerate()
            .map(|(i, v)| v * hidden_layer[i + l2])
            .sum();

        let sig = sigmoid(f);
        *epoch_loss += -(1.0 - sig).ln();
        let g = (0.0 - sig) * cbow_params.learning_rate;

        for c in 0..neu1e.len() {
            neu1e[c] += g * hidden_layer[c + l2];
            hidden_layer[c + l2] += g * neu1[c];
        }
    }

    // === BACKPROPAGATION ===
    // backpropagation, pass the hidden layer to the input layer
    context.iter().for_each(|context_index| {
        neu1e.iter().enumerate().for_each(|(k, v)| {
            input_layer[k + context_index * cbow_params.embeddings_dimension] += v;
        })
    });
}

struct UnsafePtr<T>(*mut T);
unsafe impl<T> Sync for UnsafePtr<T> {}

pub fn train_hogwild(
    pairs: &[(Vec<usize>, usize)],
    cbow_params: &CBOWParams,
    input_layer: &mut [f32],
    hidden_layer: &mut [f32],
    corpus: &CorpusValues,
) {
    let vocab_indices: Vec<usize> = (0..corpus.words_map.len()).collect();
    let emb_dim = cbow_params.embeddings_dimension;

     let input_ptr = UnsafePtr(input_layer.as_mut_ptr());
    let hidden_ptr = UnsafePtr(hidden_layer.as_mut_ptr());
    let input_len = input_layer.len();
    let hidden_len = hidden_layer.len();
    unsafe {
        pairs.par_iter().for_each(|(context, target)| {
            let mut neu1 = vec![0.0; emb_dim];
            let mut neu1e = vec![0.0; emb_dim];
            let mut rng = rand::thread_rng();

            // inside of the closure to avoid the smarter new fine-grained closure capturing.
            let _ = &input_ptr;
            let _ = &hidden_ptr;

            // SAFETY: This is "Hogwild!" style, so races may occur, but it's fast.
            pass(
                context,
                target,
                cbow_params,
                std::slice::from_raw_parts_mut(input_ptr.0, input_len),
                std::slice::from_raw_parts_mut(hidden_ptr.0, hidden_len),
                &mut neu1,
                &mut neu1e,
                &mut 0.0,
                &mut rng,
                &vocab_indices,
            );
        });
    }
}

pub fn train(
    pairs: &[(Vec<usize>, usize)],
    cbow_params: &CBOWParams,
    input_layer: &mut [f32],
    hidden_layer: &mut [f32],
    corpus: &CorpusValues,
) {
    let mut neu1: Vec<f32> = vec![0.0; cbow_params.embeddings_dimension];
    let mut neu1e: Vec<f32> = vec![0.0; cbow_params.embeddings_dimension];

    let vocab_indices: Vec<usize> = (0..corpus.words_map.len()).collect();

    // let between = Uniform::from(0..corpus.vec.len() - cbow_params.random_samples);
    let mut rng = thread_rng();

    for epoch in 0..cbow_params.epochs {
        let mut epoch_loss = 0.0;
        for (context, target) in pairs {
            pass(
                context,
                target,
                cbow_params,
                input_layer,
                hidden_layer,
                &mut neu1,
                &mut neu1e,
                &mut epoch_loss,
                &mut rng,
                &vocab_indices,
            );
        }
        tracing::info!(epoch = epoch, epoch_loss = epoch_loss, "Training epoch");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_basic() {
        let cbow_params = CBOWParams::new(2)
            .set_embeddings_dimension(2)
            .set_learning_rate(0.1)
            .set_random_samples(1);

        // Two words in vocab, 2-dim embeddings
        let mut input_layer = vec![0.5, -0.5, 0.3, 0.7]; // shape: [2, 2]
        let mut hidden_layer = vec![0.1, 0.2, -0.1, 0.4]; // shape: [2, 2]
        let mut neu1 = vec![0.0; 2];
        let mut neu1e = vec![0.0; 2];
        let mut epoch_loss = 0.0;
        let mut rng = thread_rng();
        let vocab_indices = vec![0, 1];

        // Context is word 0, target is word 1
        let context = &[0];
        let target = &1;

        pass(
            context,
            target,
            &cbow_params,
            &mut input_layer,
            &mut hidden_layer,
            &mut neu1,
            &mut neu1e,
            &mut epoch_loss,
            &mut rng,
            &vocab_indices,
        );

        assert_eq!(epoch_loss, 1.4943991);
        assert_eq!(input_layer, vec![0.4895032, -0.487263, 0.3, 0.7]);
        assert_eq!(
            hidden_layer,
            vec![0.07562487, 0.22437513, -0.07189117, 0.37189117]
        );

        assert_eq!(neu1, vec![0.5, -0.5]);
        assert_eq!(neu1e, vec![-0.010496791, 0.012737007]);
    }

    #[test]
    fn test_pass_rich_params() {
        // Use a larger vocab and higher-dimensional embeddings
        let cbow_params = CBOWParams::new(5)
            .set_embeddings_dimension(4)
            .set_learning_rate(0.2)
            .set_random_samples(2);

        // 5 words in vocab, 4-dim embeddings
        let mut input_layer = vec![
            0.1, 0.2, 0.3, 0.4, // word 0
            0.5, 0.6, 0.7, 0.8, // word 1
            0.9, 1.0, 1.1, 1.2, // word 2
            1.3, 1.4, 1.5, 1.6, // word 3
            1.7, 1.8, 1.9, 2.0, // word 4
        ];
        let mut hidden_layer = vec![
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,
            0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
        ];
        let mut neu1 = vec![0.0; 4];
        let mut neu1e = vec![0.0; 4];
        let mut epoch_loss = 0.0;
        let mut rng = thread_rng();
        let vocab_indices = vec![0, 1, 2, 3, 4];

        // Context is words 0, 1, 3; target is word 2
        let context = &[0, 1, 3];
        let target = &2;

        let orig_input = input_layer.clone();
        let orig_hidden = hidden_layer.clone();

        pass(
            context,
            target,
            &cbow_params,
            &mut input_layer,
            &mut hidden_layer,
            &mut neu1,
            &mut neu1e,
            &mut epoch_loss,
            &mut rng,
            &vocab_indices,
        );

        // Check that loss is positive and finite
        assert!(epoch_loss.is_finite() && epoch_loss > 0.0);

        // Check that weights have changed
        assert_ne!(input_layer, orig_input, "input_layer should be updated");
        assert_ne!(hidden_layer, orig_hidden, "hidden_layer should be updated");

        // Check that neu1 and neu1e are not all zeros
        assert!(neu1.iter().any(|&x| x != 0.0), "neu1 should be updated");
        assert!(neu1e.iter().any(|&x| x != 0.0), "neu1e should be updated");
    }

    fn default_params() -> CBOWParams {
        CBOWParams::new(4)
    }

    #[test]
    fn test_parse_corpus() {
        let words_map: HashMap<String, usize> = HashMap::from([
            ("uno".into(), 0),
            ("dos".into(), 1),
            ("tres".into(), 2),
            ("cinco".into(), 3),
        ]);
        let corpus = "uno and dos tres uno cinco".to_string();
        let clean = parse_corpus(corpus);
        assert_eq!(clean.vec, vec![0, 1, 2, 0, 3]);
        assert_eq!(clean.words_map, words_map);
    }

    #[test]
    fn test_create_matrices() {
        let cbow_params = default_params();
        let (input_layer, hidden_layer) = cbow_params.create_matrices();
        assert_eq!(input_layer.len(), hidden_layer.len());
        assert_eq!(input_layer.len(), 400);
    }

    #[test]
    fn test_generate_pairs() {
        let cbow_params = default_params();
        let (context, target) = cbow_params
            .generate_pairs(&[0, 1, 2, 0, 3])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(context, &[0, 1, 0, 3]);
        assert_eq!(target, 2);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.23894692834), 0.5594541);
    }
}
