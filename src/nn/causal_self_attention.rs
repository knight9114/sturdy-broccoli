use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Bool, Tensor},
};

#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,

    #[config(default = true)]
    pub bias: bool,

    #[config(default = 0.0)]
    pub dropout: f64,

    #[config(default = -1e5)]
    pub mask_value: f64,

    #[config(default = false)]
    pub quiet_softmax: bool,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub qkv_initializer: Initializer,

    #[config(
        default = "Initializer::KaimingUniform{gain:3f64.recip().sqrt(), fan_out_only:false}"
    )]
    pub out_initializer: Initializer,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "`n_heads={}` must divide `d_model={}`",
            self.n_heads,
            self.d_model,
        );

        CausalSelfAttention {
            qkv: LinearConfig::new(self.d_model, 3 * self.d_model)
                .with_bias(self.bias)
                .with_initializer(self.qkv_initializer.clone())
                .init(device),
            out: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(self.bias)
                .with_initializer(self.out_initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_model / self.n_heads,
            mask_value: self.mask_value,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    qkv: Linear<B>,
    out: Linear<B>,
    dropout: Dropout,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    mask_value: f64,
    quiet_softmax: bool,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 3, Bool>>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let qkv = self.qkv.forward(x);
        let [q, k, v] = qkv
            .chunk(3, 2)
            .try_into()
            .unwrap_or_else(|_| panic!("failed to split `qkv`"));

        let q = self.split_heads(q);
        let k = self.split_heads(k);
        let v = self.split_heads(v);

        let mut attn = q.matmul(k.transpose()) / (self.d_head as f64).sqrt();
        if let Some(mask) = padding_mask {
            attn = attn.mask_fill(mask.unsqueeze_dims(&[1, 2]), self.mask_value);
        }
        if let Some(mask) = attention_mask {
            attn = attn.mask_fill(mask.unsqueeze_dim(1), self.mask_value);
        }
        let attn = match self.quiet_softmax {
            true => activation::quiet_softmax(attn, 3),
            false => activation::softmax(attn, 3),
        };
        let attn = self.dropout.forward(attn);

        let y = attn.clone().matmul(v);
        let y = self.merge_heads(y);
        let y = self.out.forward(y);

        (y, attn)
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, t, _] = x.dims();
        x.reshape([b, t, self.n_heads, self.d_head]).swap_dims(1, 2)
    }

    fn merge_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        x.swap_dims(1, 2).reshape([0, 0, -1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::attention::generate_autoregressive_mask;
    use burn::tensor::Distribution;
    type TestBackend = burn::backend::NdArray;

    #[test]
    fn forward_without_mask() {
        let device = Default::default();
        let module = CausalSelfAttentionConfig::new(64, 4).init::<TestBackend>(&device);
        let x = Tensor::random([8, 13, 64], Distribution::Normal(0., 1.), &device);
        let (y, attn) = module.forward(x, None, None);

        assert_eq!(y.dims(), [8, 13, 64]);
        assert_eq!(attn.dims(), [8, 4, 13, 13]);
    }

    #[test]
    fn forward_with_mask() {
        let device = Default::default();
        let module = CausalSelfAttentionConfig::new(64, 4).init::<TestBackend>(&device);
        let x = Tensor::random([8, 13, 64], Distribution::Normal(0., 1.), &device);
        let attention_mask = Some(generate_autoregressive_mask(8, 13, &device));
        let (y, attn) = module.forward(x, attention_mask, None);

        assert_eq!(y.dims(), [8, 13, 64]);
        assert_eq!(attn.dims(), [8, 4, 13, 13]);
    }

    #[test]
    #[should_panic]
    fn invalid_heads() {
        let device = Default::default();
        let _module = CausalSelfAttentionConfig::new(64, 5).init::<TestBackend>(&device);
    }
}
