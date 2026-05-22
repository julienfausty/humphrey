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

        let prices = batch.nexts.clone().slice(s![0.., 0.., 1..5]);

        // gaussian kernel density estimation of next distribution over regular grid defined between [0, 2] with grid_size steps to generate targets
        // use OHLC data to define standard deviation for each of the contributions
        // weigh each of the contributions by decreasing exponential in time and volume of activity
        // normalize the sum at the end
        // KLDivLoss to generate loss

        let convolution = Tensor::<B, 3>::zeros(
            [prices.shape()[0], prices.shape()[1], grid_size],
            &prices.device(),
        );

        let targets = Tensor::<B, 2>::zeros(pass.shape(), &pass.device());

        let loss =
            KLDivLossConfig::new()
                .init()
                .forward(pass.clone(), targets.clone(), Reduction::Auto);

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
