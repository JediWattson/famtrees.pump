use std::ops;
use arrayfire::{sum_all, sum, print, pow2, randu, constant, index, join, moddims, transpose, matmul, tanh, sigmoid, MatProp, Array, Seq, dim4};

fn outer(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    let b_t = transpose(&b, false);
    matmul(&a, &b_t, MatProp::NONE, MatProp::NONE)
}

fn t_dot(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    let a_t = transpose(&a, false);
    matmul(&a_t, &b, MatProp::NONE, MatProp::NONE)
}

pub fn default(hidden_size: u64, input_size: u64) -> Neurons {
    let weight_init = || constant(0.0, dim4!(hidden_size, hidden_size + input_size));
    let bias_init = || constant(0.0, dim4!(hidden_size));

    Neurons {
        w_f: weight_init(),
        w_i: weight_init(),
        w_c: weight_init(),
        w_o: weight_init(),
        b_f: bias_init(),
        b_i: bias_init(),
        b_c: bias_init(),
        b_o: bias_init(),
    }
}

type GateOutputs = (Array<f64>, Array<f64>, Array<f64>, Array<f64>);

#[derive(Clone)]
pub struct Neurons {
    w_f: Array<f64>,
    w_i: Array<f64>,
    w_c: Array<f64>,
    w_o: Array<f64>,
    b_f: Array<f64>,
    b_i: Array<f64>,
    b_c: Array<f64>,
    b_o: Array<f64>,
}

impl ops::Div<f64> for Neurons {
    type Output = Neurons;
    fn div(self, rhs: f64) -> Self::Output { 
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

impl ops::AddAssign<Neurons> for Neurons { 
    fn add_assign(&mut self, rhs: Neurons)  {
        self.w_f += rhs.w_f;
        self.w_i += rhs.w_i;
        self.w_c += rhs.w_c;
        self.w_o += rhs.w_o;
        self.b_f += rhs.b_f;
        self.b_i += rhs.b_i;
        self.b_c += rhs.b_c;
        self.b_o += rhs.b_o;
    }
}


impl Iterator for Neurons {
    type Item = Neurons;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.clone())}    
}

impl Neurons {
    pub fn get_sum(&self) -> f64 {
        sum_all(&pow2(&self.w_f)).0
        + sum_all(&pow2(&self.w_i)).0
        + sum_all(&pow2(&self.w_c)).0
        //+ sum_all(&pow2(&self.w_o)).0

        + sum_all(&pow2(&self.b_f)).0
        + sum_all(&pow2(&self.b_i)).0
        + sum_all(&pow2(&self.b_c)).0
        //+ sum_all(&pow2(&self.b_o)).0
   }

    pub fn clip(&self, clip_coef: f64) -> Neurons {
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
    pub fn new(input_size: u64, hidden_size: u64) -> Self {
        let weight_init = || randu::<f64>(dim4!(hidden_size, hidden_size + input_size));
        let bias_init = || randu::<f64>(dim4!(hidden_size));

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
        x: &Array<f64>, 
        prev_h: &Array<f64>, 
        prev_c: &Array<f64>
    ) -> (Array<f64>, Array<f64>) {
        let concat_input = join(0, x, prev_h);
        let (f, i, c_tilde, o) = self.gate_outputs(&concat_input);
        let f = sigmoid(&f);
        let i = sigmoid(&i);
        let o = sigmoid(&o);
        let c_tilde = tanh(&c_tilde);


        let c = (f * prev_c) + (i * c_tilde);
        let h = o * tanh(&c);

        (h, c)
    }

    pub fn backward(
        &self,
        x: &Array<f64>,
        c: &Array<f64>,
        prev_h: &Array<f64>,
        prev_c: &Array<f64>,
        loss: &Array<f64>,
        next_dh: &Array<f64>,
        next_dc: &Array<f64>,
    ) -> (Neurons, Array<f64>, Array<f64>, Array<f64>) {
        let concat_input = join(0, x, prev_h);
        let (f, i, c_tilde, o) = self.gate_outputs(&concat_input);
        let dc_new = next_dc + next_dh * &o * (1.0 - tanh(&c) * tanh(&c));

        let dh_dc = &o * (1.0 - tanh(&c));
        let dc = (next_dc * &f) + (loss + next_dh) * dh_dc;

        let df = &dc_new * prev_c * &f * (1.0 - &f); // Derivative of sigmoid and multiplication by c_prev
        let di = &dc_new * &c_tilde * &i * (1.0 - &i);
        let dc_tilde = dc_new * i * (1.0 - &c_tilde * &c_tilde); // Derivative of tanh
        let do_ = next_dh * tanh(&c) * &o * (1.0 - &o);

        // Compute gradients for weights and biases
        let dw_f = outer(&df, &concat_input);
        let dw_i = outer(&di, &concat_input);
        let dw_c = outer(&dc_tilde, &concat_input);
        let dw_o = outer(&do_, &concat_input);

        // Gradient with respect to input (for backpropagation)
        let dx = t_dot(&self.neurons.w_f, &df)
            + t_dot(&self.neurons.w_i, &di)
            + t_dot(&self.neurons.w_c, &dc_tilde)
            + t_dot(&self.neurons.w_o, &do_);
       
        let input_size = self.neurons.w_f.dims()[0]; // - self.neurons.w_f.dims()[1];

        (
            Neurons {
                w_f: dw_f,
                w_i: dw_i,
                w_c: dw_c,
                w_o: dw_o,
                b_f: sum(&df, 1),
                b_i: sum(&di, 1),
                b_c: sum(&dc_tilde, 1),
                b_o: sum(&do_, 1),
            },
            index(&dx, &[Seq::new(0.0, (input_size - 1) as f64, 1.0), Seq::default()]),
            f * dc,
            index(&dx, &[Seq::new(input_size as f64, -1.0, 1.0), Seq::default()])
        )
    }

    pub fn update(&mut self, grad: &Neurons, learning_rate: f64) {
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
    
    fn gate_outputs(&self, concat_input: &Array<f64>) -> GateOutputs {
        let neurons = &self.neurons;

        let f =  matmul(&neurons.w_f, &concat_input, MatProp::NONE, MatProp::NONE) + &neurons.b_f;
        let i = matmul(&neurons.w_i, &concat_input, MatProp::NONE, MatProp::NONE) + &neurons.b_i;
        let o = matmul(&neurons.w_o, &concat_input, MatProp::NONE, MatProp::NONE) + &neurons.b_o;
        let c_tilde = matmul(&neurons.w_c, &concat_input, MatProp::NONE, MatProp::NONE) + &neurons.b_c; 
        
        (f, i, c_tilde, o)
    }
}


