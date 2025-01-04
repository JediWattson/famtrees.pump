use arrayfire::{randu, transpose, matmul, dim4, Array, MatProp};

pub type EmbeddingGrads = (Array<f64>, Array<f64>, Array<f64>); 

#[derive(Clone)]
pub struct EmbeddingLayer {
    weights: Array<f64>,
    bias: Array<f64>,
}

pub struct Embedding {
   layer: EmbeddingLayer,
   input_size: u64
}

impl Embedding {
    pub fn new(input_size: u64, embedding_size: u64) -> Self {
        let weights = randu::<f64>(dim4!(embedding_size, input_size));
        let bias =  randu::<f64>(dim4!(embedding_size));
        Embedding { input_size, layer: EmbeddingLayer { weights, bias } }
    }

   pub fn one_hot(&self, token: usize) -> Array<f64> {
        let mut one_vec = vec![0.0; self.input_size as usize];
        one_vec[token] = 1.0;
        Array::new(&one_vec, dim4!(self.input_size))
    }

    pub fn forward(&self, value: &Array<f64>) -> Array<f64> {
        matmul(&self.layer.weights, &value, MatProp::NONE, MatProp::NONE) + &self.layer.bias
    } 
    
    pub fn backward(&self, input: &Array<f64>, grad_output: &Array<f64>) -> EmbeddingGrads {
        let grad_output_t = transpose(&grad_output, false);
        let grad_weights = matmul(&input, &grad_output_t, MatProp::NONE, MatProp::NONE);

        let grad_bias = grad_output.clone();
        let grad_input = matmul(&transpose(&self.layer.weights, false), grad_output, MatProp::NONE, MatProp::NONE);


        (grad_weights, grad_bias, grad_input)
    }

    pub fn update(&mut self, grad_weights: &Array<f64>, grad_bias: &Array<f64>, learning_rate: f64) {
        self.layer.weights = &self.layer.weights - learning_rate * transpose(grad_weights, false);
        self.layer.bias = &self.layer.bias - learning_rate * grad_bias;
    }
}


