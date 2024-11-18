use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
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
        for word in clean_corpus.split_whitespace() {
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
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
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
        if vocab_size <= result.random_samples {
            result.random_samples = vocab_size - 1;
        }
        result.vocab_size = vocab_size;
        result
    }

    pub fn create_matrices(&self) -> (Array2<f32>, Array2<f32>) {
        let normal = Normal::new(self.mean, self.std_dev).unwrap();
        let input_matrix = Array2::random((self.vocab_size, self.embeddings_dimension), normal);
        let output_matrix = Array2::zeros((self.vocab_size, self.embeddings_dimension));
        (input_matrix, output_matrix)
    }

    pub fn generate_pairs(&self, corpus: &[usize]) -> Vec<(Vec<usize>, usize)> {
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

pub fn train(
    pairs: &[(Vec<usize>, usize)],
    cbow_params: &CBOWParams,
    input_layer: &mut Array2<f32>,
    hidden_layer: &mut Array2<f32>,
    corpus: &CorpusValues,
) {
    let between = Uniform::from(0..corpus.vec.len() - cbow_params.random_samples);
    let mut rng = thread_rng();

    for _ in 0..cbow_params.epochs {
        for (context, target) in pairs {
            // pass the input layer to the hidden layer
            let neu1 = get_neurone(input_layer, context, cbow_params.embeddings_dimension());

            // negative sampling
            // update_hidden_layer_and_neurone
            let mut target_slice = hidden_layer.row_mut(*target);
            let f = neu1.dot(&target_slice);
            let g = (1.0 - sigmoid(f)) * cbow_params.learning_rate;

            let mut neu1e = g * &target_slice;
            target_slice += &(g * &neu1);

            let breakpoint = between.sample(&mut rng);

            let random_indices = corpus.vec[breakpoint..cbow_params.random_samples + breakpoint]
                .iter()
                .filter(|x| x.eq(&target));

            for &negative_target in random_indices {
                let mut target_slice = hidden_layer.row_mut(negative_target);
                let f = neu1.dot(&target_slice);
                let g = (0.0 - sigmoid(f)) * cbow_params.learning_rate;

                neu1e += &(g * &target_slice);
                target_slice += &(g * &neu1);
            }

            // backpropagation, pass the hidden layer to the input layer
            // update_input_layer
            for &context_index in context {
                let mut row_slice = input_layer.row_mut(context_index);
                row_slice += &neu1e;
            }
        }
    }
}

fn update_input_layer(input_layer: &mut Array2<f32>, neu1e: &Array1<f32>, context: &[usize]) {
    for &context_index in context {
        let mut row_slice = input_layer.row_mut(context_index);
        row_slice += neu1e;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + x.neg().exp())
}

fn get_neurone(
    input_layer: &Array2<f32>,
    context: &[usize],
    embeddings_dimension: usize,
) -> Array1<f32> {
    let mut s = Array1::zeros(embeddings_dimension);

    for &context_index in context {
        let row_slice = input_layer.row(context_index);
        s += &row_slice;
    }
    s
}

fn update_hidden_layer_and_neurone(
    hidden_layer: &mut Array2<f32>,
    neu1: &Array1<f32>,
    neu1e: &mut Array1<f32>,
    target: usize,
    learning_rate: f32,
    sign: f32,
) {
    let mut target_slice = hidden_layer.row_mut(target);
    let f = neu1.dot(&target_slice);
    let g = (sign - sigmoid(f)) * learning_rate;

    neu1e.assign(&(g * &target_slice));
    target_slice += &(g * neu1);
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

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

    #[test]
    fn test_get_neurone() {
        let input_layer = array![
            [0.1, 1.0, 0.0, 3.0,],
            [0.2, 1.0, 0.0, 3.0,],
            [0.3, 1.0, 0.0, 3.0,]
        ];
        let context = [1, 2];
        let result = get_neurone(&input_layer, &context, 4);
        assert_eq!(result, array![0.5, 2.0, 0.0, 6.0]);
    }

    #[test]
    fn test_update_input_layer() {
        let mut input_layer = array![
            [0.1, 1.0, 0.0, 3.0,],
            [0.2, 1.0, 0.0, 3.0,],
            [0.3, 1.0, 0.0, 3.0,]
        ];

        let neu1e = array![0.1, 1.0, 0.0, 3.0];

        let context = [1, 2];
        update_input_layer(&mut input_layer, &neu1e, &context);
        assert_eq!(
            input_layer,
            array![
                [0.1, 1.0, 0.0, 3.0,],
                [0.3, 2.0, 0.0, 6.0,],
                [0.4, 2.0, 0.0, 6.0,]
            ]
        );
    }
}
