use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::prelude::s;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};

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

impl<B: Backend> InferenceStep for Rooney<B> {
    type Input = OHLCBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: OHLCBatch<B>) -> RegressionOutput<B> {
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

        RegressionOutput::new(loss, pass, targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub data: DataConfig,
    pub rooney: RooneyConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend<InnerBackend = B>>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.data.seed);

    let (dataloader_train, dataloader_test) =
        config.data.build::<Stdin, B>(stdin(), &device).unwrap();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.rooney.build::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
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
