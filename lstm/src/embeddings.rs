use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let m = a.len();
    let n = b.len();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

pub type EmbeddingGrads = (Array2<f32>, Array1<f32>, Array1<f32>); 

#[derive(Clone)]
pub struct EmbeddingLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

pub struct Embedding {
   layer: EmbeddingLayer,
   embedding_size: usize,
   input_size: usize
}

impl Embedding {
    pub fn new(input_size: usize, embedding_size: usize) -> Self {
        let weights = Array2::random((embedding_size, input_size), Uniform::new(-0.1, 0.1));
        let bias = Array1::random(embedding_size, Uniform::new(-0.1, 0.1));
        Embedding { embedding_size, input_size, layer: EmbeddingLayer { weights, bias } }
    }

   pub fn one_hot(&self, token: usize) -> Array1<f32> {
        let mut one_hot = Array1::zeros(self.input_size);
        one_hot[token] = 1.0;
        one_hot
    }

    pub fn forward(&self, value: &Array1<f32>) -> Array1<f32> {
        self.layer.weights.dot(value) + &self.layer.bias
    } 
    
    pub fn backward(&self, input: &Array1<f32>, grad_output: &Array1<f32>) -> EmbeddingGrads {
        let grad_output_2d = input.clone().insert_axis(Axis(0));
        let grad_weights = grad_output.clone().insert_axis(Axis(1)).dot(&grad_output_2d);
        
        //println!("grad weights: {:?}", grad_weights.t().shape());

        let grad_bias = grad_output.clone();
        let grad_input = self.layer.weights.t().dot(grad_output);


        (grad_weights, grad_bias, grad_input)
    }

    pub fn update(&mut self, grad_weights: &Array2<f32>, grad_bias: &Array1<f32>, learning_rate: f32) {
        //println!("grads: {:?}, weights: {:?}", grad_weights.shape(), self.layer.weights.shape());
        self.layer.weights = &self.layer.weights - learning_rate * grad_weights;
        self.layer.bias = &self.layer.bias - learning_rate * grad_bias;
    }
}


