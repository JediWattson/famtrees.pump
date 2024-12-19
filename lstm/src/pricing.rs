use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct PriceEmbedding {
    weights: Array2<f32>,
    bias: Array1<f32>,
    embedding_size: usize,
}

impl PriceEmbedding {
    pub fn new(input_size: usize, embedding_size: usize) -> Self {
        let weights = Array2::random((embedding_size, input_size), Uniform::new(-0.1, 0.1));
        let bias = Array1::random(embedding_size, Uniform::new(-0.1, 0.1));
        PriceEmbedding { weights, bias, embedding_size }
    }

    pub fn forward(&self, price: f32) -> Array1<f32> {
        let price_vec = Array1::from(vec![price]);
        self.weights.dot(&price_vec) + &self.bias
    }    
}
