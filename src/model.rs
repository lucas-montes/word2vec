use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::Path,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct Word2VecModel {
    pub vocab: HashMap<String, usize>,
    pub embeddings: Vec<f32>,
    pub embedding_dim: usize,
}

impl Word2VecModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        if path.as_ref().extension().is_some_and(|e| e.eq("txt")) {
            return Self::load_c_file(path).expect("Failed to load model from C file");
        }
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .expect("Failed to open model file");
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .expect("Failed to read model file");
        bincode::deserialize(&buffer).expect("Failed to deserialize model")
    }

    fn load_c_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        let mut header = first_line.split_whitespace();
        let vocab_size: usize = header.next().unwrap().parse()?;
        let embedding_dim: usize = header.next().unwrap().parse()?;

        let mut vocab = HashMap::with_capacity(vocab_size);
        let mut embeddings = Vec::with_capacity(vocab_size * embedding_dim);

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let word = parts.next().unwrap().to_string();
            vocab.insert(word, idx);

            for val in parts {
                embeddings.push(val.parse::<f32>()?);
            }
        }

        Ok(Self {
            vocab,
            embeddings,
            embedding_dim,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let serialized = bincode::serialize(self).expect("Failed to serialize model");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .expect("Failed to create model file");
        file.write_all(&serialized)
            .expect("Failed to write model to file");
    }

    pub fn new(vocab: HashMap<String, usize>, embeddings: Vec<f32>, embedding_dim: usize) -> Self {
        Self {
            vocab,
            embeddings,
            embedding_dim,
        }
    }

    pub fn get_embedding(&self, word: &str) -> Option<&[f32]> {
        self.vocab.get(word).map(|&index| {
            let start = index * self.embedding_dim;
            let end = start + self.embedding_dim;
            &self.embeddings[start..end]
        })
    }

    pub fn cosine_similarity(&self, word1: &str, word2: &str) -> Option<f32> {
        let embedding1 = self.get_embedding(word1)?;
        let embedding2 = self.get_embedding(word2)?;

        Some(cosine_similarity(embedding1, embedding2))
    }

    pub fn most_similar(&self, word: &str, top_k: usize) -> Option<Vec<(String, f32)>> {
        let target_embedding = self.get_embedding(word)?;

        let mut similarities: Vec<(String, f32)> = self
            .vocab
            .iter()
            .filter(|(w, _)| *w != word) // Exclude the word itself
            .map(|(w, &index)| {
                let start = index * self.embedding_dim;
                let end = start + self.embedding_dim;
                let embedding = &self.embeddings[start..end];
                let similarity = cosine_similarity(target_embedding, embedding);
                (w.clone(), similarity)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);
        Some(similarities)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn contains_word(&self, word: &str) -> bool {
        self.vocab.contains_key(word)
    }

    pub fn evaluate_simlex<P: AsRef<Path> + std::fmt::Debug>(
        &self,
        simlex_path: P,
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(simlex_path)?;
        let reader = BufReader::new(file);

        let mut output_file = File::create(&output_path)?;

        writeln!(output_file, "wordd,word2,human_score,model_score")?;

        for line in reader.lines().skip(1) {
            let line = line?;
            let mut parts = line.split('\t');

            let word1 = parts.next().unwrap();
            let word2 = parts.next().unwrap();
            let human_score = parts.nth(1).unwrap_or("N/A").parse::<f32>().unwrap_or(0.0) / 10.0;

            let model_score = match self.cosine_similarity(word1, word2) {
                Some(score) => format!("{:.4}", score),
                None => "N/A".to_string(),
            };

            writeln!(
                output_file,
                "{},{},{},{}",
                word1, word2, human_score, model_score
            )?;
        }

        println!("Evaluation complete. Results saved to {:?}.", output_path);
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
