use burn::config::Config;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{MapperDataset, SelectionDataset};
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;

use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};

use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

use crate::data::{NormalizeOHLCItem, OHLCBatch, OHLCBatcher, OHLCDataset, ToInnerBackend};
use crate::model::RooneyConfig;

/// Utility struct for configuring the train/test data split and providing data loaders for training
#[derive(Config, Debug)]
pub struct DataConfig {
    #[config(default = 1.0)]
    pub use_only: f32,
    #[config(default = 0.8)]
    pub train_split: f32,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 64)]
    pub window_size: usize,
    #[config(default = 42)]
    pub seed: u64,
}

impl DataConfig {
    /// Method for coalescing the builder pattern into the train and test data (respectively)
    /// User must provide a readable source formatted in McZielinski CSV style
    pub fn build<R: Read, B: AutodiffBackend>(
        &self,
        source: R,
        device: &B::Device,
    ) -> Result<
        (
            Arc<dyn DataLoader<B, OHLCBatch<B>>>,
            Arc<dyn DataLoader<B::InnerBackend, OHLCBatch<B::InnerBackend>>>,
        ),
        String,
    > {
        let base_dataset = match OHLCDataset::<B>::new(self.window_size.clone(), source, &device) {
            Ok(ds) => ds,
            Err(message) => return Err(message),
        };

        let visible_portion = (base_dataset.len() as f32 * self.use_only) as usize;
        let block_size = 4 * self.window_size + self.batch_size.clone();
        let total_visible_blocks = visible_portion / block_size.clone();
        let number_test_blocks =
            (total_visible_blocks.clone() as f32 * (1.0 - self.train_split)) as usize;

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.clone());
        let test_blocks: HashSet<usize> = (0..number_test_blocks)
            .map(|_| rng.random_range(0..total_visible_blocks.clone()))
            .collect();

        let train_blocks: Vec<usize> = (0..total_visible_blocks)
            .filter(|i_block| !test_blocks.contains(i_block))
            .collect();
        let test_blocks = Vec::from_iter(test_blocks.into_iter());

        let unroll = |set: Vec<usize>| {
            let mut unrolled = Vec::with_capacity(set.len() * block_size.clone());
            let origin = base_dataset.len();
            for i_block in set.into_iter() {
                let offset = block_size.clone() * i_block;
                for j_inner in 0..block_size.clone() {
                    unrolled.push(origin - offset - j_inner - 1)
                }
            }
            unrolled
        };

        let batcher = OHLCBatcher {};

        let train_set = DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size.clone())
            .shuffle(self.seed.clone())
            .num_workers(self.num_workers.clone())
            .build(SelectionDataset::from_indices_unchecked(
                MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem),
                unroll(train_blocks),
            ));

        let test_set = DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size.clone())
            .shuffle(self.seed.clone())
            .num_workers(self.num_workers.clone())
            .build(SelectionDataset::from_indices_unchecked(
                MapperDataset::new(
                    MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem),
                    ToInnerBackend,
                ),
                unroll(test_blocks),
            ));

        Ok((train_set, test_set))
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub data: DataConfig,
    pub rooney: RooneyConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}
