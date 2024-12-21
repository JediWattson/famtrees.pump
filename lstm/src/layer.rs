use crate::cell::{Neurons, LSTMCell};
use ndarray::{s, Array1, Array2};
use bincode::{serialize, deserialize};
use std::fs::{File, read};
use std::io::Write;

pub struct LSTMLayer {
    hidden_size: usize,
    input_size: usize,
    learning_rate: f32,
    cells: Vec<LSTMCell>,
}

fn mse_loss(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
    ((predictions - targets).mapv(|x| x * x)).sum() / (predictions.len() as f32)
}

fn mse_derivative(prediction: &Array1<f32>, target: &Array1<f32>) -> Array1<f32> {
    assert_eq!(prediction.len(), target.len(), "Single prediction and target must have the same length");
    let n = prediction.len() as f32;
    (prediction - target).mapv(|x| 2.0 * x / n)
}

impl LSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f32) -> Self {
        let cells = std::iter::repeat_with(|| LSTMCell::new(hidden_size, hidden_size))
            .take(1) // Single layer, but could be expanded for stacked layers
            .collect();
        
        LSTMLayer {
            learning_rate,
            cells, 
            hidden_size,
            input_size,
        }
    }
 
    pub fn train(&mut self, batch: &Vec<Array2<f32>>) -> f32 {
        println!("Forward Started!");
        let mut cached_hs = Vec::new();
        let mut cached_cs = Vec::new();
        let mut h = Array1::<f32>::zeros(self.hidden_size);
        let mut c = Array1::<f32>::zeros(self.hidden_size);
        for sequence in batch {
            let (hs, cs, next_h, next_c) = self.forward(&sequence, &h, &c);
            h = next_h;
            c = next_c;
            cached_hs.push(hs);
            cached_cs.push(cs);
        }
        
        println!("Backward Prop Started!");
        let mut total_loss = 0.0;
        for t in (1..batch.len()).rev() {
            let target = &batch[t];
            let grads_seq = self.backward(
                target,
                &cached_hs,
                &cached_cs,
                t
            );        

            self.update(&grads_seq);
            let mut prediction = Array2::<f32>::zeros((self.hidden_size, self.input_size));
            for (i, predictor) in cached_hs[t].iter().enumerate() {
                prediction.slice_mut(s![.., i]).assign(predictor);
            }

            total_loss += mse_loss(target, &prediction);
        }
        let _ = self.save_weights("weights.bin");
        total_loss
    } 

    fn forward(
        &self, 
        sequence: &Array2<f32>, 
        last_h: &Array1<f32>, 
        last_c: &Array1<f32>
    ) -> (
        Vec<Array1<f32>>, 
        Vec<Array1<f32>>, 
        Array1<f32>, 
        Array1<f32>
    ) {
        let mut hs: Vec<Array1<f32>> = Vec::new();
        let mut cs: Vec<Array1<f32>> = Vec::new();
        let mut h = last_h.clone();
        let mut c = last_c.clone();
        for t in 0..self.input_size {
            let x = &sequence.column(t).to_owned();
            for cell in &self.cells {
                let (new_h, new_c) = cell.forward(x, &h, &c);
                hs.push(new_h.clone());
                cs.push(new_c.clone());
                h = new_h;
                c = new_c;

            }

        }

        (hs, cs, c, h)
    }


    fn backward(
        &mut self, 
        target: &Array2<f32>,
        cached_hs: &Vec<Vec<Array1<f32>>>,
        cached_cs: &Vec<Vec<Array1<f32>>>,
        step: usize
    ) -> Vec<Neurons> {
       
        let mut grads_seq = Vec::new(); 
        let mut dh = Array1::zeros(self.hidden_size);
        let mut dc = Array1::zeros(self.hidden_size); 
        
        let prev_h = &cached_hs[step-1];
        let prev_c = &cached_cs[step-1]; 
        let h = &cached_hs[step];
        let c = &cached_cs[step];
    
        for t in 1..self.input_size {
            let x_target = target.column(t).to_owned();

            let seq_h = h[t].to_owned();
            let seq_c = c[t].to_owned();
            let prev_seq_h = prev_h[t].to_owned();
            let prev_seq_c = prev_c[t].to_owned();
            let loss_x = mse_derivative(&seq_h, &x_target);
            
            for cell in &self.cells {
                let (grad_h, _, new_dc, new_dh) = cell.backward(
                    &seq_h, 
                    &seq_c, 
                    &prev_seq_h, 
                    &prev_seq_c, 
                    &loss_x, 
                    &dh, 
                    &dc,
                );

                dh = new_dh;
                dc = new_dc;
                grads_seq.push(grad_h);
            }
        }
        grads_seq
    }

    fn update(&mut self, gradients: &Vec<Neurons>) {
        let max_norm: f32 = 2.0;   
        let min_norm: f32 = 0.1;
        for g_h in gradients { 
            let total_norm = g_h.get_sum().sqrt();
            assert!(!(total_norm.is_nan() || total_norm.is_infinite()), "NOT A NUMBER");
           
            let clip_h = if total_norm > max_norm {
                &g_h.clip(max_norm / total_norm)
            } else if total_norm < min_norm {
                &g_h.clip(min_norm / total_norm)
            } else { g_h };

            self.cells[0].update(&clip_h, self.learning_rate);
        }
    }

    fn save_weights(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;
        let weights: Vec<Vec<Array2<f32>>> = self.cells.iter()
            .map(|cell| { cell.save() }).collect();
        
        let serialized = serialize(&weights)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load_weights(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = read(filename)?;
        let weights: Vec<Vec<Array2<f32>>> = deserialize(&buffer)?;

        if self.cells.len() != weights.len() {
            return Err("Mismatch in number of layers between saved and current model".into());
        }

        for (cell, layer_weights) in self.cells.iter_mut().zip(weights.into_iter()) {
            if layer_weights.len() != 4 {
                return Err("Incorrect number of weight matrices for LSTM cell".into());
            }
            cell.load(&layer_weights); 
        }
        Ok(())
    }
}
