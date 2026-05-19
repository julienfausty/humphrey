use burn::backend::{Wgpu, wgpu::WgpuDevice};

use std::io::{Stdin, stdin};

mod data;
use data::DataConfig;

type BackendInUse = Wgpu<f32, i32>;

fn main() -> Result<(), String> {
    let device = WgpuDevice::default();

    let split_sets = DataConfig::new()
        .build::<Stdin, BackendInUse>(stdin(), &device)
        .unwrap();

    println!(
        "split: train {}, test {}",
        split_sets.0.num_items(),
        split_sets.1.num_items()
    );
    println!("{:?}", split_sets.0.iter().next());

    Ok(())
}
