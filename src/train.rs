use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::MapperDataset;

use std::io::stdin;

mod data;
use data::{NormalizeOHLCItem, OHLCDataset, OHLCItem};

type Backend = Wgpu<f32, i32>;

#[derive(Config, Debug)]
struct DataConfig {
    #[config(default = 1.0)]
    pub use_only: f32,
    #[config(default = 0.8)]
    pub train_split: f32,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
}

impl DataConfig {}

fn main() -> Result<(), String> {
    let device = WgpuDevice::default();
    let ohlc_dataset = OHLCDataset::<Backend>::new(64, stdin(), &device).unwrap();

    let mapped_dataset: MapperDataset<_, _, OHLCItem<Backend>> =
        MapperDataset::new(ohlc_dataset, NormalizeOHLCItem);

    println!("{:?}", mapped_dataset.get(0));

    Ok(())
}
