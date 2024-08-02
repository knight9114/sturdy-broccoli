use crate::nn::{
    CausalSelfAttention, CausalSelfAttentionConfig, LayerNorm, LayerNormConfig, Mlp, MlpConfig,
};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer},
    tensor::{backend::Backend, Bool, Tensor},
};

#[derive(Config, Debug)]
pub struct GptBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,

    #[config(default = true)]
    pub bias: bool,

    #[config(default = 0.0)]
    pub mha_dropout: f64,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub mha_qkv_initializer: Initializer,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub mha_out_initializer: Initializer,

    #[config(default = 0.0)]
    pub mlp_dropout: f64,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub mlp_fc1_initializer: Initializer,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub mlp_fc2_initializer: Initializer,

    #[config(default = -1e5)]
    pub mask_value: f64,

    #[config(default = false)]
    pub quiet_softmax: bool,

    #[config(default = 1e-5)]
    pub epsilon: f64,

    #[config(default = "Initializer::Ones")]
    pub norm_gamma_initializer: Initializer,

    #[config(default = "Initializer::Zeros")]
    pub norm_beta_initializer: Initializer,
}

impl GptBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GptBlock<B> {
        GptBlock {
            mha: CausalSelfAttentionConfig::from(self).init(device),
            mha_dropout: DropoutConfig::new(self.mha_dropout).init(),
            mha_norm: LayerNormConfig::from(self).init(device),
            mlp: MlpConfig::from(self).init(device),
            mlp_dropout: DropoutConfig::new(self.mlp_dropout).init(),
            mlp_norm: LayerNormConfig::from(self).init(device),
        }
    }
}

impl From<&GptBlockConfig> for LayerNormConfig {
    fn from(config: &GptBlockConfig) -> LayerNormConfig {
        LayerNormConfig::new(config.d_model)
            .with_bias(config.bias)
            .with_epsilon(config.epsilon)
            .with_gamma_initializer(config.norm_gamma_initializer.clone())
            .with_beta_initializer(config.norm_beta_initializer.clone())
    }
}

impl From<&GptBlockConfig> for CausalSelfAttentionConfig {
    fn from(config: &GptBlockConfig) -> CausalSelfAttentionConfig {
        CausalSelfAttentionConfig::new(config.d_model, config.n_heads)
            .with_bias(config.bias)
            .with_dropout(config.mha_dropout)
            .with_mask_value(config.mask_value)
            .with_quiet_softmax(config.quiet_softmax)
            .with_qkv_initializer(config.mha_qkv_initializer.clone())
            .with_out_initializer(config.mha_out_initializer.clone())
    }
}

impl From<&GptBlockConfig> for MlpConfig {
    fn from(config: &GptBlockConfig) -> MlpConfig {
        MlpConfig::new(config.d_model, config.d_ff)
            .with_bias(config.bias)
            .with_fc1_initializer(config.mlp_fc1_initializer.clone())
            .with_fc2_initializer(config.mlp_fc2_initializer.clone())
    }
}

#[derive(Module, Debug)]
pub struct GptBlock<B: Backend> {
    mha_norm: LayerNorm<B>,
    mha_dropout: Dropout,
    mha: CausalSelfAttention<B>,
    mlp_norm: LayerNorm<B>,
    mlp_dropout: Dropout,
    mlp: Mlp<B>,
}

impl<B: Backend> GptBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (y, attn) = self.mha.forward(self.mha_norm.forward(x.clone()), mask);
        let x = x + self.mha_dropout.forward(y);
        let y = self.mlp.forward(self.mlp_norm.forward(x.clone()));
        (x + self.mlp_dropout.forward(y), attn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    type TestBackend = burn::backend::NdArray;

    #[test]
    fn forward_without_mask() {
        let device = Default::default();
        let module = GptBlockConfig::new(64, 4, 128).init::<TestBackend>(&device);
        let x = Tensor::random([8, 13, 64], Distribution::Normal(0., 1.), &device);
        let (y, attn) = module.forward(x, None);

        assert_eq!(y.dims(), [8, 13, 64]);
        assert_eq!(attn.dims(), [8, 4, 13, 13]);
    }

    #[test]
    fn forward_with_mask() {
        let device = Default::default();
        let module = GptBlockConfig::new(64, 4, 128).init::<TestBackend>(&device);
        let x = Tensor::random([8, 13, 64], Distribution::Normal(0., 1.), &device);
        let mask = Tensor::triu_mask([13, 13], 1, &device);
        let (y, attn) = module.forward(x, Some(mask));

        assert_eq!(y.dims(), [8, 13, 64]);
        assert_eq!(attn.dims(), [8, 4, 13, 13]);
    }
}
