use burn::{
    config::Config,
    module::Module,
    nn::{Gelu, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub d_model: usize,
    pub d_ff: usize,

    #[config(default = true)]
    pub bias: bool,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub fc1_initializer: Initializer,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub fc2_initializer: Initializer,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            fc1: LinearConfig::new(self.d_model, self.d_ff)
                .with_bias(self.bias)
                .with_initializer(self.fc1_initializer.clone())
                .init(device),
            fc2: LinearConfig::new(self.d_ff, self.d_model)
                .with_bias(self.bias)
                .with_initializer(self.fc2_initializer.clone())
                .init(device),
            gelu: Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    gelu: Gelu,
}

impl<B: Backend> Mlp<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.fc1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.fc2.forward(x);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    type TestBackend = burn::backend::NdArray;

    #[test]
    fn forward() {
        let device = Default::default();
        let module = MlpConfig::new(2, 3).init::<TestBackend>(&device);
        let x = Tensor::random([8, 2], Distribution::Normal(0., 1.), &device);

        let y = module.forward(x);
        assert_eq!(y.dims(), [8, 2]);
    }
}
