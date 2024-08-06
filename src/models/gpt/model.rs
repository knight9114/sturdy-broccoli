use crate::nn::{GptBlock, GptBlockConfig, LayerNorm, LayerNormConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::generate_autoregressive_mask, Dropout, DropoutConfig, Embedding,
        EmbeddingConfig, Initializer, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Bool, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct GptConfig {
    pub vocab_size: usize,

    #[config(default = 12)]
    pub n_layers: usize,

    #[config(default = 768)]
    pub d_model: usize,

    #[config(default = 12)]
    pub n_heads: usize,

    #[config(default = 3072)]
    pub d_ff: usize,

    #[config(default = 1024)]
    pub max_sequence_length: usize,

    #[config(default = true)]
    pub bias: bool,

    #[config(default = 0.02)]
    pub projection_scale: f64,

    #[config(default = 0.0)]
    pub embedding_dropout: f64,

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

impl GptConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt<B> {
        let scale = self.projection_scale / (2. * self.n_layers as f64).sqrt();
        let init = Initializer::Normal {
            mean: 0.,
            std: scale,
        };
        let config = self
            .clone()
            .with_mha_out_initializer(init.clone())
            .with_mlp_fc2_initializer(init.clone());

        Gpt {
            core: GptCore {
                token_embedding: EmbeddingConfig::new(config.vocab_size, config.d_model)
                    .init(device),
                position_embedding: EmbeddingConfig::new(
                    config.max_sequence_length,
                    config.d_model,
                )
                .init(device),
                dropout: DropoutConfig::new(config.embedding_dropout).init(),
                blocks: (0..config.n_layers)
                    .map(|_| GptBlockConfig::from(&config).init(device))
                    .collect(),
                norm: LayerNormConfig::from(&config).init(device),
            },
            lm_head: LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn from_name(name: &str) -> GptConfig {
        match name {
            "nano" => Self::default()
                .with_n_layers(3)
                .with_n_heads(3)
                .with_d_model(48)
                .with_d_ff(48 * 4),
            "micro" => Self::default()
                .with_n_layers(4)
                .with_n_heads(4)
                .with_d_model(128)
                .with_d_ff(128 * 4),
            "mini" => Self::default()
                .with_n_layers(6)
                .with_n_heads(6)
                .with_d_model(192)
                .with_d_ff(192 * 4),
            "default" => Self::default(),
            "medium" => Self::default()
                .with_n_layers(24)
                .with_n_heads(16)
                .with_d_model(1024)
                .with_d_ff(1024 * 4),
            "large" => Self::default()
                .with_n_layers(36)
                .with_n_heads(20)
                .with_d_model(1280)
                .with_d_ff(1280 * 4),
            "xlarge" => Self::default()
                .with_n_layers(48)
                .with_n_heads(25)
                .with_d_model(1600)
                .with_d_ff(1600 * 4),
            "124M" => Self::from_name("default"),
            "350M" => Self::from_name("medium"),
            "774M" => Self::from_name("large"),
            "1558M" => Self::from_name("xlarge"),
            _ => {
                eprintln!("unknown `name={}` provided, using default", name);
                Self::default()
            }
        }
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        let config = Self::new(50304);
        let scale = config.projection_scale / (2. * config.n_layers as f64).sqrt();
        let init = Initializer::Normal {
            mean: 0.,
            std: scale,
        };

        config
            .with_mha_out_initializer(init.clone())
            .with_mlp_fc2_initializer(init.clone())
    }
}

impl From<&GptConfig> for GptBlockConfig {
    fn from(config: &GptConfig) -> GptBlockConfig {
        GptBlockConfig::new(config.d_model, config.n_heads, config.d_ff)
            .with_bias(config.bias)
            .with_mask_value(config.mask_value)
            .with_quiet_softmax(config.quiet_softmax)
            .with_mha_dropout(config.mha_dropout)
            .with_mha_qkv_initializer(config.mha_qkv_initializer.clone())
            .with_mha_out_initializer(config.mha_out_initializer.clone())
            .with_mlp_dropout(config.mlp_dropout)
            .with_mlp_fc1_initializer(config.mlp_fc1_initializer.clone())
            .with_mlp_fc2_initializer(config.mlp_fc2_initializer.clone())
            .with_epsilon(config.epsilon)
            .with_norm_gamma_initializer(config.norm_beta_initializer.clone())
            .with_norm_beta_initializer(config.norm_gamma_initializer.clone())
    }
}

impl From<&GptConfig> for LayerNormConfig {
    fn from(config: &GptConfig) -> LayerNormConfig {
        LayerNormConfig::new(config.d_model)
            .with_bias(config.bias)
            .with_epsilon(config.epsilon)
            .with_gamma_initializer(config.norm_gamma_initializer.clone())
            .with_beta_initializer(config.norm_beta_initializer.clone())
    }
}

#[derive(Module, Debug)]
pub struct GptCore<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    dropout: Dropout,
    blocks: Vec<GptBlock<B>>,
    norm: LayerNorm<B>,
}

impl<B: Backend> GptCore<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 3, Bool>>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let device = self.devices()[0].clone();
        let [b, t] = input.dims();
        let mut attns = Vec::with_capacity(self.blocks.len());

        let pos = Tensor::arange(0..t as i64, &device).unsqueeze::<2>();
        let t_emb = self.position_embedding.forward(pos);
        let s_emb = self.token_embedding.forward(input);

        let attention_mask =
            attention_mask.or_else(|| Some(generate_autoregressive_mask(b, t, &device)));
        let mut x = self.dropout.forward(s_emb + t_emb);
        for block in self.blocks.iter() {
            let (y, attn) = block.forward(x, attention_mask.clone(), padding_mask.clone());
            x = y;
            attns.push(attn)
        }

        let x = self.norm.forward(x);
        (x, attns)
    }
}

#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    core: GptCore<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> Gpt<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 3, Bool>>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let (x, attns) = self.core.forward(input, attention_mask, padding_mask);
        (self.lm_head.forward(x), attns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Float};
    type TestBackend = burn::backend::NdArray;

    #[test]
    fn forward() {
        let device = Default::default();
        let module = GptConfig::from_name("nano").init::<TestBackend>(&device);
        let x = Tensor::<TestBackend, 2, Float>::random(
            [8, 13],
            Distribution::Uniform(0., 50304.),
            &device,
        );
        let (y, attns) = module.forward(x.int(), None, None);

        assert_eq!(y.dims(), [8, 13, 50304]);
        assert_eq!(module.core.blocks.len(), attns.len());
    }
}
