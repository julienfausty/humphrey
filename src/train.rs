use burn::backend::{Wgpu, wgpu::WgpuDevice};

use std::io::stdin;

mod data;
use data::OHLCDataset;

fn main() -> Result<(), String> {
    let device = WgpuDevice::default();
    let ohlc_dataset = OHLCDataset::<Wgpu<f32, i32>>::new(60, stdin(), &device);

    println!("{:?}", ohlc_dataset);

    Ok(())
}
