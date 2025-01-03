use std::fs::{File, read};
use std::io::Write;
use ndarray::{Axis, Array1, Array2};
use bincode::{serialize, deserialize};
use crate::cell::{average_gradients_over_time, Neurons, LSTMCell};
use crate::embeddings::{Embedding, EmbeddingGrads};

pub struct LSTMLayer {
    token_embedding: Embedding,
    dense_embedding: Embedding,
    hidden_size: usize,
    vocab_size: usize,
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
    pub fn new() -> Self {
        let vocab_size = 128;
        let hidden_size = 48;
        let learning_rate = 0.00001;
        
        let token_embedding = Embedding::new(vocab_size, hidden_size);
        let dense_embedding = Embedding::new(hidden_size, vocab_size);
        let cells = std::iter::repeat_with(|| LSTMCell::new(hidden_size, hidden_size))
            .take(3) 
            .collect();
        
        LSTMLayer {
            cells, 
            token_embedding,
            dense_embedding,
            learning_rate,
            hidden_size,
            vocab_size,
        }
    }

    pub fn train(&mut self, batch: &[Vec<f32>]) {
        println!("Forward Started!");
        let mut cached_hs = Vec::new();
        let mut cached_cs = Vec::new();
        let mut os = Vec::new();
        let mut h = Array1::<f32>::zeros(self.hidden_size);
        let mut c = Array1::<f32>::zeros(self.hidden_size);
        for sequence in batch {
            let (o, hs, cs, next_h, next_c) = self.forward(sequence, &h, &c);
            h = next_h;
            c = next_c;
            cached_hs.push(hs);
            cached_cs.push(cs);
            os.push(o);
        }

        println!("Backward Prop Started!");
        let mut total_loss = 0.0;
        for t in 1..batch.len() {
            let (grads_seq, loss) = self.backward(
                &batch[t],
                &os[t],
                &cached_hs,
                &cached_cs,
                t
            );
            self.update(&grads_seq);
            total_loss += loss;
        }
       
        println!("saving");
        let _ = self.save_weights("weights.bin");
        println!("Loss: {}", total_loss/(batch.len() as f32));
    } 

    fn forward(
        &self, 
        sequence: &Vec<f32>,
        last_h: &Array1<f32>, 
        last_c: &Array1<f32>
    ) -> ( 
        Vec<Array1<f32>>,
        Vec<Vec<Array1<f32>>>,
        Vec<Vec<Array1<f32>>>,
        Array1<f32>, 
        Array1<f32>, 
    ) {
        let mut hs = Vec::new();
        let mut cs = Vec::new();
        let mut h = last_h.clone();
        let mut c = last_c.clone();
        let mut os = Vec::new();
        for token in sequence {
            let mut hs_x = Vec::new();
            let mut cs_x = Vec::new();

            let one_hot = self.token_embedding.one_hot(*token as usize);
            let emb = self.token_embedding.forward(&one_hot);
 
            for cell in &self.cells {
                let (new_h, new_c) = cell.forward(&emb, &h, &c);
                h = new_h;
                c = new_c;
                hs_x.push(h.clone());
                cs_x.push(c.clone());
            }

            let o = self.dense_embedding.forward(&h);
            hs.push(hs_x);
            cs.push(cs_x);
            os.push(o);
        }

        (os, hs, cs, h, c)
    }

    fn backward(
        &mut self,
        target: &Vec<f32>,
        output: &Vec<Array1<f32>>,
        cached_hs: &Vec<Vec<Vec<Array1<f32>>>>,
        cached_cs: &Vec<Vec<Vec<Array1<f32>>>>,
        step: usize
    ) -> (Vec<(EmbeddingGrads, Neurons, EmbeddingGrads)>, f32) {       
        let mut grads_seq = Vec::new(); 
      
        let mut dh = Array1::zeros(self.hidden_size);
        let mut dc = Array1::zeros(self.hidden_size); 
 
        let prev_hs = &cached_hs[step-1];
        let prev_cs = &cached_cs[step-1]; 
        let hs = &cached_hs[step];
        let cs = &cached_cs[step];
        let mut loss = 0.0;

        for (i, token) in target.iter().enumerate().rev() {
            let mut grads_dx = Array2::zeros((self.cells.len(), self.hidden_size));
            let mut grads_h = Vec::new();
           
            let one_hot = self.token_embedding.one_hot(*token as usize);
            loss += (&output[i] - &one_hot).mapv(|x| x.powi(2)).sum() / 2.0;
            let loss_x = &output[i] - &one_hot;
            
            let grads_d = self.dense_embedding.backward(&hs[i][0], &loss_x);
        
            for (j, cell) in self.cells.iter().enumerate() {
                let (grad_h, dx, new_dc, new_dh) = cell.backward(
                    &hs[i][j], 
                    &cs[i][j], 
                    &prev_hs[i][j], 
                    &prev_cs[i][j], 
                    &grads_d.2, 
                    &dh,
                    &dc
                );
                
                dh = new_dh;
                dc = new_dc;
                grads_dx.row_mut(j).assign(&dx);
                grads_h.push(grad_h);
            }
            


            let avg_h = average_gradients_over_time(&grads_h);
            let avg_dx = grads_dx.mean_axis(Axis(0)).unwrap();
            let grads_i = self.token_embedding.backward(&one_hot, &avg_dx);
        
            grads_seq.push((grads_d, avg_h, grads_i));
        }

        (grads_seq, loss / (target.len() as f32))
    }

    fn update(&mut self, gradients: &Vec<(EmbeddingGrads, Neurons, EmbeddingGrads)> ) {
        let max_norm: f32 = 2.0;   
        let min_norm: f32 = 0.1;
        for (g_d, g_h, g_e) in gradients.iter() {
            self.dense_embedding.update(&g_d.0, &g_d.1, self.learning_rate);
            
            let total_norm = g_h.get_sum().sqrt();
            assert!(!(total_norm.is_nan() || total_norm.is_infinite()), "NOT A NUMBER");
            let clip_h = if total_norm > max_norm {
                &g_h.clip(max_norm / total_norm)
            } else if total_norm < min_norm {
                &g_h.clip(min_norm / total_norm)
            } else { g_h };
            
            for cell in &mut self.cells {
                cell.update(&clip_h, self.learning_rate);
            }

            self.token_embedding.update(&g_e.0, &g_e.1, self.learning_rate);
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
