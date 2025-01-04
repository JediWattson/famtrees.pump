use arrayfire::{constant, exp, tile, log, mul, sum_all, sum, dim4, Dim4, Array};
//use bincode::{serialize, deserialize};
use crate::cell::{default, Neurons, LSTMCell};
use crate::embeddings::{Embedding, EmbeddingGrads};
use std::io::Write;

fn softmax(x: &Array<f64>) -> Array<f64> {
    let exp_x = exp(x);
    let sum_exp_x = sum(&exp_x, 0); // Assuming 0 is the dimension for classes
    exp_x / tile(&sum_exp_x, dim4!(x.dims()[0]))
}

fn cross_entropy_loss(probs: &Array<f64>, label: &Array<f64>) -> f64 {
    let log_probs = log(&probs);
    let loss_per_example = -mul(label, &log_probs, false); 
    let total_loss = sum(&sum(&loss_per_example, 0), 1); 
    let batch_size = total_loss.dims()[1]; 
    let average_loss = sum(&total_loss, 0) / constant(batch_size as f32, Dim4::new(&[1, 1, 1, 1]));
    
    sum_all(&average_loss).0
}


pub struct LSTMLayer {
    token_embedding: Embedding,
    dense_embedding: Embedding,
    hidden_size: u64,
    learning_rate: f64,
    cells: Vec<LSTMCell>,
}

impl LSTMLayer {
    pub fn new() -> Self {
        let vocab_size = 128;
        let hidden_size = 32;
        let learning_rate = 0.00001;
        
        let token_embedding = Embedding::new(vocab_size, hidden_size);
        let dense_embedding = Embedding::new(hidden_size, vocab_size);
        let cells = std::iter::repeat_with(|| LSTMCell::new(hidden_size, hidden_size))
            .take(12) 
            .collect();
        
        LSTMLayer {
            cells, 
            token_embedding,
            dense_embedding,
            learning_rate,
            hidden_size,
        }
    }

    pub fn train(&mut self, batch: &[Vec<f32>]) {
        println!("Forward Started!");
        let mut cached_hs = Vec::new();
        let mut cached_cs = Vec::new();
        let mut os = Vec::new();
        let mut h = constant(0.0, dim4!(self.hidden_size));
        let mut c = constant(0.0, dim4!(self.hidden_size));
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
            let percentage = (t as f64 + 1.0) / batch.len() as f64 * 100.0;
            
            print!("\r{:.2}% done", percentage);
            std::io::stdout().flush().unwrap();

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
       
        //println!("saving");
        //let _ = self.save_weights("weights.bin");
        println!("\nLoss: {}", total_loss/(batch.len() as f64));
    } 

    fn forward(
        &self, 
        sequence: &Vec<f32>,
        last_h: &Array<f64>, 
        last_c: &Array<f64>
    ) -> ( 
        Vec<Array<f64>>,
        Vec<Vec<Array<f64>>>,
        Vec<Vec<Array<f64>>>,
        Array<f64>, 
        Array<f64>, 
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
        output: &Vec<Array<f64>>,
        cached_hs: &Vec<Vec<Vec<Array<f64>>>>,
        cached_cs: &Vec<Vec<Vec<Array<f64>>>>,
        step: usize
    ) -> (Vec<(EmbeddingGrads, Neurons, EmbeddingGrads)>, f64) {       
        let mut grads_seq = Vec::new(); 
        let mut dc = constant(0.0, dim4!(self.hidden_size));
        let mut dh = constant(0.0, dim4!(self.hidden_size));
 
        let prev_hs = &cached_hs[step-1];
        let prev_cs = &cached_cs[step-1]; 
        let hs = &cached_hs[step];
        let cs = &cached_cs[step];
        let mut loss = 0.0;

        for (i, token) in target.iter().enumerate().rev() {
            let mut sum_grads_dx = constant(0.0, dim4!(self.hidden_size));
            let mut avg_h = default(self.hidden_size, self.hidden_size);
             
            let one_hot = self.token_embedding.one_hot(*token as usize);

            let probs = softmax(&output[i]);
            loss += cross_entropy_loss(&probs, &one_hot);
            let loss_x = probs - &one_hot;
     
            let cell_len = self.cells.len() as f64;
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
                sum_grads_dx += dx;
                avg_h += grad_h / cell_len;
            }
   
            let dx_dims = sum_grads_dx.dims();
            let avg_dx = sum_grads_dx / constant(cell_len, dx_dims);
            let grads_i = self.token_embedding.backward(&one_hot, &avg_dx);

            grads_seq.push((grads_d, avg_h, grads_i));

            //let sqr_diff = pow2(&loss_x);
            //loss += sum_all(&sqr_diff).0 / 2.0;
        }

        (grads_seq, loss / (target.len() as f64))
    }

    fn update(&mut self, gradients: &Vec<(EmbeddingGrads, Neurons, EmbeddingGrads)> ) {
        //let max_norm: f64 = 2.0;   
        //let min_norm: f64 = 0.1;
        for (g_d, g_h, g_e) in gradients.iter() {
            self.dense_embedding.update(&g_d.0, &g_d.1, self.learning_rate);

            let total_norm = g_h.get_sum().sqrt();
            //println!("{}", total_norm);
            assert!(!(total_norm.is_nan() || total_norm.is_infinite()), "NOT A NUMBER");


            for cell in &mut self.cells {
                cell.update(&g_h, self.learning_rate);
            }

            self.token_embedding.update(&g_e.0, &g_e.1, self.learning_rate);
       }
    }

}
