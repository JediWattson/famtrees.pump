use crate::layer::{LSTMLayer, LSTMLayerGradients};

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }

    pub fn update(&self, layer: &mut LSTMLayer, gradients: &LSTMLayerGradients) {
        // Update weights and biases for each LSTM cell
        for (cell, grad) in layer.cells.iter_mut().zip(gradients.cells.iter()) {
       }
    }
}
