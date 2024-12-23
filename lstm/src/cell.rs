use ndarray::{s, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::ops;

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


pub fn average_gradients_over_time(gradients: &[Neurons]) -> Neurons {
    let num_steps = gradients.len() as f32;

    let zero_grad = Neurons {
        w_f: Array2::zeros(gradients[0].w_f.dim()),
        w_i: Array2::zeros(gradients[0].w_i.dim()),
        w_c: Array2::zeros(gradients[0].w_c.dim()),
        w_o: Array2::zeros(gradients[0].w_o.dim()),
        b_f: Array1::zeros(gradients[0].b_f.len()),
        b_i: Array1::zeros(gradients[0].b_i.len()),
        b_c: Array1::zeros(gradients[0].b_c.len()),
        b_o: Array1::zeros(gradients[0].b_o.len()),
    };

    let summed_grads = gradients.iter().fold(zero_grad, |mut acc, grad| {
        acc.w_f += &grad.w_f;
        acc.w_i += &grad.w_i;
        acc.w_c += &grad.w_c;
        acc.w_o += &grad.w_o;
        acc.b_f += &grad.b_f;
        acc.b_i += &grad.b_i;
        acc.b_c += &grad.b_c;
        acc.b_o += &grad.b_o;
        acc
    });
    
    summed_grads / num_steps
}


fn gate_outputs(cell: &Neurons, concat_input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
    let f = (cell.w_f.dot(concat_input) + &cell.b_f).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let i = (cell.w_i.dot(concat_input) + &cell.b_i).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let c_tilde = (cell.w_c.dot(concat_input) + &cell.b_c).mapv(|x| x.tanh());
    let o = (cell.w_o.dot(concat_input) + &cell.b_o).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    (f, i, c_tilde, o)
}

#[derive(Debug, Clone)]
pub struct Neurons {
    w_f: Array2<f32>,
    w_i: Array2<f32>,
    w_c: Array2<f32>,
    w_o: Array2<f32>,
    b_f: Array1<f32>,
    b_i: Array1<f32>,
    b_c: Array1<f32>,
    b_o: Array1<f32>,
}

impl ops::Div<f32> for Neurons {
    type Output = Neurons;
    fn div(self, rhs: f32) -> Self::Output { 
        Neurons {
            w_f: self.w_f / rhs,
            w_i: self.w_i / rhs,
            w_c: self.w_c / rhs,
            w_o: self.w_o / rhs, 
            b_f: self.b_f / rhs,
            b_i: self.b_i / rhs,
            b_c: self.b_c / rhs,
            b_o: self.b_o / rhs,
        }  

    }

}

impl Iterator for Neurons {
    type Item = Neurons;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.clone())}    
}

impl Neurons {
    pub fn get_sum(&self) -> f32 {
        self.w_f.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.w_i.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.w_c.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.w_o.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.b_f.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.b_i.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.b_c.iter().map(|x| x.powi(2)).sum::<f32>()
        + self.b_o.iter().map(|x| x.powi(2)).sum::<f32>()
    }

    pub fn clip(&self, clip_coef: f32) -> Neurons {
        Neurons { 
            w_f: &self.w_f * clip_coef,
            w_i: &self.w_i * clip_coef,
            w_c: &self.w_c * clip_coef,
            w_o: &self.w_o * clip_coef,

            b_f: &self.b_f * clip_coef,
            b_i: &self.b_i * clip_coef,
            b_c: &self.b_c * clip_coef,
            b_o: &self.b_o * clip_coef,
        }
    }
  }

pub struct LSTMCell {
    neurons: Neurons,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_init = || Array2::random((hidden_size, input_size + hidden_size), Uniform::new(-0.1, 0.1));
        let bias_init = || Array1::random(hidden_size, Uniform::new(-0.1, 0.1));

        LSTMCell {
            neurons: Neurons {
                w_f: weight_init(),
                w_i: weight_init(),
                w_c: weight_init(),
                w_o: weight_init(),
                b_f: bias_init(),
                b_i: bias_init(),
                b_c: bias_init(),
                b_o: bias_init(),
            },
        }
    }

    pub fn forward(
        &self, 
        x: &Array1<f32>, 
        prev_h: &Array1<f32>, 
        prev_c: &Array1<f32>
    ) -> (Array1<f32>, Array1<f32>) {
        let neurons = &self.neurons;
        let concat_input = x.iter().chain(prev_h.iter()).cloned().collect::<Array1<f32>>();
        let f = (neurons.w_f.dot(&concat_input) + &neurons.b_f).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let i = (neurons.w_i.dot(&concat_input) + &neurons.b_i).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let o = (neurons.w_o.dot(&concat_input) + &neurons.b_o).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let c_tilde = (neurons.w_c.dot(&concat_input) + &neurons.b_c).mapv(|x| x.tanh());
       
        let c = f * prev_c + i * c_tilde;
        let h = o * c.mapv(|x| x.tanh());

        (h, c)
    }

    pub fn backward(
        &self,
        x: &Array1<f32>,
        c: &Array1<f32>,
        prev_h: &Array1<f32>,
        prev_c: &Array1<f32>,
        loss: &Array1<f32>,
        next_dh: &Array1<f32>,
        next_dc: &Array1<f32>,
    ) -> (Neurons, Array1<f32>, Array1<f32>, Array1<f32>) {
        let concat_input = x.iter().chain(prev_h.iter()).cloned().collect::<Array1<f32>>();
        let (f, i, c_tilde, o) = gate_outputs(&self.neurons, &concat_input);
    
        let dh_dc = o * (1.0 - c.mapv(|x| x.tanh() * x.tanh()));
        let dc = (next_dc * &f) + (loss + next_dh) * dh_dc;

        // Gradient of loss with respect to gates
        let df = &dc * prev_c;
        let di = &dc * c_tilde;
        let dc_tilde = &dc * i;
        let do_ = (next_dh + loss) * c.mapv(|x| 1.0 - x.tanh() * x.tanh()); 

        // Compute gradients for weights and biases
        let dw_f = outer(&df, &concat_input);
        let dw_i = outer(&di, &concat_input);
        let dw_c = outer(&dc_tilde, &concat_input);
        let dw_o = outer(&do_, &concat_input);

        // Gradient with respect to input (for backpropagation)
        let dx = self.neurons.w_f.t().dot(&df) 
            + self.neurons.w_i.t().dot(&di) 
            + self.neurons.w_c.t().dot(&dc_tilde) 
            + self.neurons.w_o.t().dot(&do_);
        
        let input_size = self.neurons.w_f.ncols() - self.neurons.w_f.nrows();

        (
            Neurons {
                w_f: dw_f,
                w_i: dw_i,
                w_c: dw_c,
                w_o: dw_o,
                b_f: df,
                b_i: di,
                b_c: dc_tilde,
                b_o: do_,
            },
            dx.slice(s![..input_size]).to_owned(), 
            f * dc,
            dx.slice(s![input_size..]).to_owned()
        )
    }

    pub fn update(&mut self, grad: &Neurons, learning_rate: f32) {
        let neurons = &mut self.neurons;
        // Update weights
        neurons.w_f = &neurons.w_f - learning_rate * &grad.w_f;
        neurons.w_i = &neurons.w_i -learning_rate * &grad.w_i;
        neurons.w_c = &neurons.w_c - learning_rate * &grad.w_c;
        neurons.w_o = &neurons.w_o - learning_rate * &grad.w_o;

        // Update biases
        neurons.b_f = &neurons.b_f - learning_rate * &grad.b_f;
        neurons.b_i = &neurons.b_i - learning_rate * &grad.b_i;
        neurons.b_c = &neurons.b_c - learning_rate * &grad.b_c;
        neurons.b_o = &neurons.b_o - learning_rate * &grad.b_o;
    }
    
    pub fn save(&self) -> Vec<Array2<f32>>  {
        let cell = &self.neurons;
        vec![cell.w_f.clone(), cell.w_i.clone(), cell.w_c.clone(), cell.w_o.clone()]
    }

    pub fn load(&mut self, layer_weights: &Vec<Array2<f32>>) {
        let cell = &mut self.neurons;
        cell.w_f = layer_weights[0].clone();
        cell.w_i = layer_weights[1].clone();
        cell.w_c = layer_weights[2].clone();
        cell.w_o = layer_weights[3].clone();
    }
}


