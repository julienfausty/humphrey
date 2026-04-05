use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use csv;
use std::io::{Read, stdin};

#[derive(Debug, serde::Deserialize)]
struct McZielinskiOHLC {
    pub Timestamp: f32,
    pub Low: f32,
    pub Open: f32,
    pub Close: f32,
    pub High: f32,
    pub Volume: f32,
}

fn parse<T: Read>(source: T) -> Result<Vec<f32>, String> {
    let mut ohlc_data = Vec::new();
    let mut reader = csv::Reader::from_reader(source);
    for parsed in reader.deserialize() {
        let ohlc: McZielinskiOHLC = match parsed {
            Ok(ohlc) => ohlc,
            Err(err) => return Err(format!("Error parsing csv data:\n{:?}", err)),
        };

        ohlc_data.push(ohlc.Timestamp);
        ohlc_data.push(ohlc.Low);
        ohlc_data.push(ohlc.Open);
        ohlc_data.push(ohlc.Close);
        ohlc_data.push(ohlc.High);
        ohlc_data.push(ohlc.Volume);
    }
    Ok(ohlc_data)
}

#[derive(Debug, Clone)]
pub struct OHLCItem<B: Backend> {
    pub block: Tensor<B, 2>,
    pub next: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct OHLCDataset<B: Backend> {
    block_size: usize,
    loaded: Tensor<B, 2>,
}

impl<B: Backend> OHLCDataset<B> {
    pub fn new<Src: Read>(
        block_size: usize,
        source: Src,
        device: &<B as Backend>::Device,
    ) -> Result<OHLCDataset<B>, String> {
        let raw = match parse(source) {
            Ok(raw) => raw,
            Err(message) => return Err(message),
        };

        let n_time_steps = raw.len() / 6;

        if n_time_steps < 2 * block_size {
            return Err(format!(
                "Cannot create dataset with block_size {block_size} and total size {n_time_steps} because toal size is too small."
            ));
        }

        Ok(OHLCDataset {
            block_size,
            loaded: Tensor::<B, 2>::from_data(TensorData::new(raw, [n_time_steps, 6]), device),
        })
    }
}

impl<B: Backend> Dataset<OHLCItem<B>> for OHLCDataset<B> {
    fn get(&self, index: usize) -> Option<OHLCItem<B>> {
        if index >= self.len() {
            return None;
        }

        Some(OHLCItem {
            block: self
                .loaded
                .clone()
                .slice([index..(index + self.block_size)]),
            next: self
                .loaded
                .clone()
                .slice([(index + self.block_size)..(index + 2 * self.block_size)]),
        })
    }

    fn len(&self) -> usize {
        self.loaded.shape().dims[0] - 2 * self.block_size
    }
}

fn main() -> Result<(), String> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let ohlc_dataset = OHLCDataset::<burn::backend::Wgpu<f32, i32>>::new(60, stdin(), &device);

    println!("{:?}", ohlc_dataset);

    Ok(())
}
