use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{KLDivLossConfig, Reduction};
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
            (0.95, 1.05),
        );

        let loss = KLDivLossConfig::new().init().forward(
            pass.clone(),
            targets.clone(),
            Reduction::BatchMean,
        );

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
            (0.95, 1.05),
        );

        let loss = KLDivLossConfig::new().init().forward(
            pass.clone(),
            targets.clone(),
            Reduction::BatchMean,
        );

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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
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
type AutodiffBackendInUse = Autodiff<BackendInUse>;

fn main() -> Result<(), String> {
    let device = WgpuDevice::default();

    let artifact_dir = "/tmp/rooney";

    let window_size = 64;

    let data_config = DataConfig::new()
        .with_use_only(0.05)
        .with_window_size(window_size);

    let model_config = RooneyConfig::new(window_size, 6 /*ohlc features*/, 1, 32)
        .with_latent_size(128)
        .with_n_expansion_stacks(4)
        .with_n_thinking_stacks(1)
        .with_n_attention_heads(4)
        .with_n_reasoning_layers(2)
        .with_n_distillation_layers(2);

    train::<AutodiffBackendInUse>(
        artifact_dir,
        TrainingConfig::new(data_config, model_config, AdamConfig::new())
            .with_num_epochs(5)
            .with_learning_rate(1e-3),
        device.clone(),
    );

    Ok(())
}
