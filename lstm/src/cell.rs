use ndarray::{s, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

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

fn gate_outputs(cell: &Neurons, concat_input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
    let f = (cell.w_f.dot(concat_input) + &cell.b_f).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let i = (cell.w_i.dot(concat_input) + &cell.b_i).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let c_tilde = (cell.w_c.dot(concat_input) + &cell.b_c).mapv(|x| x.tanh());
    let o = (cell.w_o.dot(concat_input) + &cell.b_o).mapv(|x| 1.0 / (1.0 + (-x).exp()));
    (f, i, c_tilde, o)
}

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

pub struct LSTMCell {
    neurons: Neurons,
    hidden_size: usize,
    learning_rate: f32
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_init = || Array2::random((hidden_size, input_size + hidden_size), Uniform::new(-0.1, 0.1));
        let bias_init = || Array1::random(hidden_size, Uniform::new(-0.1, 0.1));

        LSTMCell {
            hidden_size,
            learning_rate: 0.01,
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
        &mut self,
        x: &Array1<f32>,
        c: &Array1<f32>,
        prev_h: &Array1<f32>,
        prev_c: &Array1<f32>,
        next_dh: &Array1<f32>,
        next_dc: &Array1<f32>,
        loss: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
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
        
        // Gradient with respect to previous h (for backpropagation through time)
        let dh_prev = dx.slice(s![self.neurons.w_f.ncols() - self.hidden_size..]);

        let gradients = Neurons {
            w_f: dw_f,
            w_i: dw_i,
            w_c: dw_c,
            w_o: dw_o,
            b_f: df,
            b_i: di,
            b_c: dc_tilde,
            b_o: do_,
        };
       
        self.update(&gradients);
        
        return (
            dx.slice(s![..self.neurons.w_f.ncols() - self.hidden_size]).to_owned(), 
            f * dc,
            dh_prev.to_owned()
        );
    }

    fn update(&mut self, grad: &Neurons) {
        let neurons = &mut self.neurons;
        // Update weights
        neurons.w_f = &neurons.w_f - self.learning_rate * &grad.w_f;
        neurons.w_i = &neurons.w_i - self.learning_rate * &grad.w_i;
        neurons.w_c = &neurons.w_c - self.learning_rate * &grad.w_c;
        neurons.w_o = &neurons.w_o - self.learning_rate * &grad.w_o;

        // Update biases
        neurons.b_f = &neurons.b_f - self.learning_rate * &grad.b_f;
        neurons.b_i = &neurons.b_i - self.learning_rate * &grad.b_i;
        neurons.b_c = &neurons.b_c - self.learning_rate * &grad.b_c;
        neurons.b_o = &neurons.b_o - self.learning_rate * &grad.b_o;
    }
    
    pub fn save(&self) -> Vec<Array2<f32>>  {
        let cell = &self.neurons;
        vec![cell.w_f.clone(), cell.w_i.clone(), cell.w_c.clone(), cell.w_o.clone()]
    }

    pub fn load(&mut self, layer_weights: &Vec<Array2<f32>>) {
        let cell = &mut self.neurons;
        println!("{:?}", layer_weights);
        cell.w_f = layer_weights[0].clone();
        cell.w_i = layer_weights[1].clone();
        cell.w_c = layer_weights[2].clone();
        cell.w_o = layer_weights[3].clone();
    }
}


