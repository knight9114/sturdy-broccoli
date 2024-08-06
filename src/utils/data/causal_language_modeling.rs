use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, SqliteDataset, SqliteDatasetError};
use burn::nn::attention::{
    generate_autoregressive_mask, generate_padding_mask, GeneratePaddingMask,
};
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(derive_new::new, Clone, Debug, Deserialize, Serialize)]
pub struct ItemForCausalLM {
    pub text: String,
}

pub struct DatasetForCausalLM {
    dataset: SqliteDataset<ItemForCausalLM>,
}

impl DatasetForCausalLM {
    pub fn from_db_file<P: AsRef<Path>>(
        db_file: P,
        split: &str,
    ) -> Result<Self, SqliteDatasetError> {
        let dataset = SqliteDataset::from_db_file(db_file, split)?;
        Ok(Self { dataset })
    }
}

impl Dataset<ItemForCausalLM> for DatasetForCausalLM {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<ItemForCausalLM> {
        self.dataset.get(index)
    }
}

#[derive(derive_new::new, Clone, Debug)]
pub struct InferenceBatchForCausalLM<B: Backend> {
    pub prefixes: Tensor<B, 2, Int>,
    pub padding_mask: Tensor<B, 2, Bool>,
    pub attention_mask: Tensor<B, 3, Bool>,
}

#[derive(derive_new::new, Clone, Debug)]
pub struct TrainBatchForCausalLM<B: Backend> {
    pub prefixes: Tensor<B, 2, Int>,
    pub suffixes: Tensor<B, 2, Int>,
    pub padding_mask: Tensor<B, 2, Bool>,
    pub attention_mask: Tensor<B, 3, Bool>,
}

#[derive(derive_new::new, Clone, Debug)]
pub struct BatcherForCausalLM {
    tokenizer: Tokenizer,
    max_sequence_length: usize,
}

impl<B: Backend> Batcher<ItemForCausalLM, InferenceBatchForCausalLM<B>> for BatcherForCausalLM {
    fn batch(&self, items: Vec<ItemForCausalLM>) -> InferenceBatchForCausalLM<B> {
        let mut tokens: Vec<Vec<usize>> = Vec::with_capacity(items.len());
        for item in items {
            let encoded = self
                .tokenizer
                .encode(item.text, true)
                .expect("failed to tokenize input")
                .get_ids()
                .iter()
                .map(|t| *t as usize)
                .collect();
            tokens.push(encoded);
        }

        let GeneratePaddingMask { tensor, mask } = generate_padding_mask::<B>(
            self.tokenizer
                .token_to_id("<|endoftext|>")
                .expect("failed to get the token, `<|endoftext|>`") as usize,
            tokens,
            Some(self.max_sequence_length),
            &B::Device::default(),
        );
        let prefixes = tensor;
        let padding_mask = mask;
        let [batch_size, seq_length] = prefixes.dims();

        let attention_mask =
            generate_autoregressive_mask(batch_size, seq_length, &B::Device::default());

        InferenceBatchForCausalLM {
            prefixes,
            padding_mask,
            attention_mask,
        }
    }
}

impl<B: Backend> Batcher<ItemForCausalLM, TrainBatchForCausalLM<B>> for BatcherForCausalLM {
    fn batch(&self, items: Vec<ItemForCausalLM>) -> TrainBatchForCausalLM<B> {
        let batch: InferenceBatchForCausalLM<B> = self.batch(items);
        let [batch_size, seq_length] = batch.prefixes.dims();

        let prefixes = batch
            .prefixes
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let suffixes = batch.prefixes.slice([0..batch_size, 1..seq_length]);
        let attention_mask =
            batch
                .attention_mask
                .slice([0..batch_size, 0..seq_length - 1, 0..seq_length - 1]);
        let padding_mask = batch.padding_mask.slice([0..batch_size, 0..seq_length - 1]);

        TrainBatchForCausalLM {
            prefixes,
            suffixes,
            attention_mask,
            padding_mask,
        }
    }
}
