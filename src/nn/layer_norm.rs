use burn::{
    config::Config,
    module::{Module, Param},
    nn::Initializer,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    pub d_model: usize,

    #[config(default = true)]
    pub bias: bool,

    #[config(default = 1e-5)]
    pub epsilon: f64,

    #[config(default = "Initializer::Ones")]
    pub gamma_initializer: Initializer,

    #[config(default = "Initializer::Zeros")]
    pub beta_initializer: Initializer,
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm {
            gamma: self.gamma_initializer.init([self.d_model], device),
            beta: match self.bias {
                true => Some(self.beta_initializer.init([self.d_model], device)),
                false => None,
            },
            epsilon: self.epsilon,
        }
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Option<Param<Tensor<B, 1>>>,
    epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let (sigma, mu) = x.clone().var_mean_bias(D - 1);
        let norm = (x - mu) / (sigma.sqrt() + self.epsilon);

        let mut output = norm * self.gamma.val().unsqueeze();
        if let Some(beta) = &self.beta {
            output = output + beta.val().unsqueeze();
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestBackend = burn::backend::NdArray;
    type TestAudodiffBackend = burn::backend::Autodiff<TestBackend>;

    #[test]
    fn forward() {
        let device = Default::default();
        let module = LayerNormConfig::new(10).init::<TestBackend>(&device);
        let x = Tensor::<TestBackend, 2>::from_floats(
            [[
                -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
            ]],
            &device,
        );

        let y = module.forward(x);
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[
                -0.4990, -1.9680, 1.6178, -0.7486, -0.6470, 0.8576, 0.0461, 1.1111, -0.2614, 0.4915,
            ]],
            &device,
        );

        y.to_data().assert_approx_eq(&expected.to_data(), 3);
    }

    #[test]
    fn backward() {
        let device = Default::default();
        let module = LayerNormConfig::new(2).init::<TestAudodiffBackend>(&device);
        let x1 = Tensor::<TestAudodiffBackend, 2>::from_floats([[0., 1.], [3., 4.]], &device)
            .require_grad();
        let x2 = Tensor::<TestAudodiffBackend, 2>::from_floats([[6., 7.], [9., 10.]], &device)
            .require_grad();
        let x = x1.clone().matmul(x2.clone());

        let y = module.forward(x);
        let grads = y.backward();

        let x1_grad = x1.grad(&grads).unwrap();
        let x2_grad = x2.grad(&grads).unwrap();
        let gamma_grad = module.gamma.grad(&grads).unwrap();
        let beta_grad = module.beta.unwrap().grad(&grads).unwrap();

        let x1_grad_expected = Tensor::zeros_like(&x1_grad);
        x1_grad
            .to_data()
            .assert_approx_eq(&x1_grad_expected.to_data(), 3);

        let x2_grad_expected = Tensor::zeros_like(&x2_grad);
        x2_grad
            .to_data()
            .assert_approx_eq(&x2_grad_expected.to_data(), 3);

        let gamma_grad_expected = Tensor::<TestAudodiffBackend, 1>::from_floats([-2., 2.], &device);
        gamma_grad
            .to_data()
            .assert_approx_eq(&gamma_grad_expected.to_data(), 3);

        let beta_grad_expected = Tensor::<TestAudodiffBackend, 1>::from_floats([2., 2.], &device);
        beta_grad
            .to_data()
            .assert_approx_eq(&beta_grad_expected.to_data(), 3);
    }
}
