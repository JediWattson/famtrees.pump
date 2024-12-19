use crate::cell::LSTMCell;
use ndarray::{s, Array1, Array2};
use bincode::{serialize, deserialize};
use std::fs::{File, read};
use std::io::Write;

pub struct LSTMLayer {
    hidden_size: usize,
    cells: Vec<LSTMCell>,
    cached_hs: Vec<Array2<f32>>,
    cached_cs: Vec<Array2<f32>>,
    last_h: Array1<f32>,
    last_c: Array1<f32>,
}


fn mse_derivative(prediction: &Array1<f32>, target: &Array1<f32>) -> Array1<f32> {
    assert_eq!(prediction.len(), target.len(), "Single prediction and target must have the same length");
    let n = prediction.len() as f32;
    (prediction - target).mapv(|x| 2.0 * x / n)
}

impl LSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let cells = std::iter::repeat_with(|| LSTMCell::new(input_size, hidden_size))
            .take(1) // Single layer, but could be expanded for stacked layers
            .collect();
        
        LSTMLayer { 
            cells, 
            hidden_size,
            cached_hs: Vec::new(),
            cached_cs: Vec::new(),
            last_h: Array1::zeros(hidden_size),
            last_c: Array1::zeros(hidden_size),
        }
    }
 
    pub fn train(&mut self, batch: &Vec<Array2<f32>>) -> f32 {
        println!("Forward Started!");
        for sequence in batch {
            let (hs, cs) = self.forward(&sequence);
            self.cached_hs.push(hs);
            self.cached_cs.push(cs); 
        }
        
        println!("Backward Prop Started!");
        let mut total_loss = 0.0;
        for t in (1..batch.len()).rev() {
            total_loss += self.backward(batch[t].to_owned(), t);        
        }
        
        let _ = self.save_weights("weights.bin");
        total_loss
    } 

    fn forward(&mut self, sequence: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = sequence.nrows();
        let mut h = self.last_h.clone();
        let mut c = self.last_c.clone();
        let mut hs = Array2::<f32>::zeros((self.hidden_size, seq_len));
        let mut cs = Array2::<f32>::zeros((self.hidden_size, seq_len));

        for t in 0..seq_len {
            let x = sequence.row(t).to_owned();
            let (new_h, new_c) = self.cells[0].forward(&x, &h, &c);
            h = new_h;
            c = new_c;
            hs.slice_mut(s![.., t]).assign(&h);
            cs.slice_mut(s![.., t]).assign(&c);
        }

        self.last_h = h;
        self.last_c = c;
        (hs, cs)
    }


    fn backward(&mut self, seq: Array2<f32>, step: usize) -> f32 {
        let mut dh = Array1::zeros(self.hidden_size);
        let mut dc = Array1::zeros(self.hidden_size); 
        let prev_h = &self.cached_hs[step-1];
        let prev_c = &self.cached_cs[step-1]; 
        let h = &self.cached_hs[step];
        let c = &self.cached_cs[step];
        
        let mut total_loss = 0.0;
        for t in 1..seq.nrows() {
            let seq_h = h.row(t).to_owned();
            let seq_c = c.row(t).to_owned();
            let prev_seq_h = prev_h.row(t).to_owned();
            let prev_seq_c = prev_c.row(t).to_owned();
           
            let x_out = seq.column(t).to_owned();
            let loss = mse_derivative(&seq_h, &x_out);

            let x_emb = seq.row(t).to_owned();
            let (_, new_dc, new_dh) = self.cells[0].backward(
                &x_emb, 
                &seq_c, 
                &prev_seq_h, 
                &prev_seq_c, 
                &dh, 
                &dc,
                &loss, 
            );


            dh = new_dh;
            dc = new_dc; 

            total_loss += loss.sum();
        }

        total_loss
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
