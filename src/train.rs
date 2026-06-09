use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::config::Config;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

use std::io::{Stdin, stdin};

use rooney::configs::{DataConfig, TrainingConfig};
use rooney::model::RooneyConfig;

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

    let window_size = 256;

    let data_config = DataConfig::new()
        .with_use_only(0.01)
        .with_batch_size(32)
        .with_window_size(window_size);

    let model_config = RooneyConfig::new(window_size, 6 /*ohlc features*/, 1, 32)
        .with_latent_size(128)
        .with_n_expansion_stacks(4)
        .with_n_thinking_stacks(1)
        .with_n_attention_heads(1)
        .with_n_reasoning_layers(2)
        .with_n_distillation_layers(2);

    train::<AutodiffBackendInUse>(
        artifact_dir,
        TrainingConfig::new(data_config, model_config, AdamConfig::new())
            .with_num_epochs(3)
            .with_learning_rate(1e-5),
        device.clone(),
    );

    Ok(())
}
