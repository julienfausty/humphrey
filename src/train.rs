use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::MapperDataset;

use std::io::stdin;

mod data;
use data::{NormalizeOHLCItem, OHLCDataset, OHLCItem};

type Backend = Wgpu<f32, i32>;

fn main() -> Result<(), String> {
    let device = WgpuDevice::default();
    let ohlc_dataset = OHLCDataset::<Backend>::new(64, stdin(), &device).unwrap();

    let mapped_dataset: MapperDataset<_, _, OHLCItem<Backend>> =
        MapperDataset::new(ohlc_dataset, NormalizeOHLCItem);

    println!("{:?}", mapped_dataset.get(0));

    Ok(())
}
