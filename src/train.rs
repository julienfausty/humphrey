use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::nn::loss::{KLDivLoss, KLDivLossConfig, Reduction};
use burn::prelude::s;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep};

use std::io::{Stdin, stdin};

mod data;
use data::{DataConfig, OHLC2Distribution, OHLCBatch};

mod model;
use model::{Rooney, RooneyConfig};

impl<B: AutodiffBackend> TrainStep for Rooney<B> {
    type Input = OHLCBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: OHLCBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let pass = self.forward(batch.blocks).reshape([0, -1]);

        let grid_size = pass.shape()[1];

        let projector = OHLC2Distribution;

        let targets = projector.project(
            batch.nexts.clone().slice_assign(
                s![0.., 0.., 0],
                batch.nexts.clone().slice(s![0.., 0.., 0]) - 1.0,
            ),
            grid_size,
        );

        let loss = MseLoss::new().forward(pass.clone(), targets.clone(), Reduction::Auto);

        let output = RegressionOutput::new(loss, pass, targets);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

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
