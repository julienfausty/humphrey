use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::modules::conv::{Conv1d, Conv1dConfig};
use burn::nn::modules::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{Dropout, DropoutConfig, PaddingConfig1d, Relu};
use burn::prelude::s;
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep};

use crate::data::{OHLC2Distribution, OHLCBatch};

#[derive(Module, Debug)]
pub struct ExpansionLayer<B: Backend> {
    stacks: Vec<(Conv1d<B>, Relu)>,
    dropout: Dropout,
}

impl<B: Backend> ExpansionLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.stacks.iter().fold(input, |acc, stack| {
            self.dropout.forward(stack.1.forward(stack.0.forward(acc)))
        })
    }
}

#[derive(Config, Debug)]
pub struct ExpansionLayerConfig {
    input_n_channels: usize,
    output_n_channels: usize,
    #[config(default = 5)]
    kernel_size: usize,
    #[config(default = 4)]
    n_stacks: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl ExpansionLayerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> ExpansionLayer<B> {
        let projection_chain: Vec<usize> = (0..(self.n_stacks + 1))
            .map(|i_chain| {
                let weight = i_chain as f64 / self.n_stacks as f64;

                ((1.0 - weight) * self.input_n_channels as f64
                    + weight * self.output_n_channels as f64) as usize
            })
            .collect();

        ExpansionLayer {
            stacks: (0..self.n_stacks)
                .map(|i_chain| {
                    (
                        Conv1dConfig::new(
                            projection_chain[i_chain],
                            projection_chain[i_chain + 1],
                            self.kernel_size.clone(),
                        )
                        .with_padding(PaddingConfig1d::Same)
                        .init(device),
                        Relu::new(),
                    )
                })
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct DistillationLayer<B: Backend> {
    stacks: Vec<(AdaptiveAvgPool1d, Relu, Conv1d<B>, Relu)>,
    dropout: Dropout,
}

impl<B: Backend> DistillationLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.stacks.iter().fold(input, |acc, stack| {
            self.dropout.forward(
                stack.3.forward(
                    stack
                        .2
                        .forward(self.dropout.forward(stack.1.forward(stack.0.forward(acc)))),
                ),
            )
        })
    }
}

#[derive(Config, Debug)]
pub struct DistillationLayerConfig {
    input_size: usize,
    input_n_channels: usize,
    output_size: usize,
    output_n_channels: usize,
    #[config(default = 5)]
    kernel_size: usize,
    #[config(default = 4)]
    n_stacks: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl DistillationLayerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> DistillationLayer<B> {
        let projection_chain: Vec<(usize, usize)> = (0..(self.n_stacks + 1))
            .map(|i_chain| {
                let weight = i_chain as f64 / self.n_stacks as f64;
                (
                    ((1.0 - weight) * self.input_size as f64 + weight * self.output_size as f64)
                        as usize,
                    ((1.0 - weight) * self.input_n_channels as f64
                        + weight * self.output_n_channels as f64) as usize,
                )
            })
            .collect();

        DistillationLayer {
            stacks: (0..self.n_stacks)
                .map(|i_chain| {
                    (
                        AdaptiveAvgPool1dConfig::new(projection_chain[i_chain + 1].0).init(),
                        Relu::new(),
                        Conv1dConfig::new(
                            projection_chain[i_chain].1,
                            projection_chain[i_chain + 1].1,
                            self.kernel_size.clone(),
                        )
                        .with_padding(PaddingConfig1d::Same)
                        .init(device),
                        Relu::new(),
                    )
                })
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ThinkingLayer<B: Backend> {
    encoder: TransformerEncoder<B>,
    distill: DistillationLayer<B>,
    dropout: Dropout,
}

impl<B: Backend> ThinkingLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let buffer = self
            .encoder
            .forward(TransformerEncoderInput::new(input))
            .transpose();
        let buffer = self.dropout.forward(buffer);
        let buffer = self.distill.forward(buffer);
        self.dropout.forward(buffer).transpose()
    }
}

#[derive(Config, Debug)]
pub struct ThinkingLayerConfig {
    input_size: usize,
    input_n_channels: usize,
    output_size: usize,
    output_n_channels: usize,

    #[config(default = 5)]
    kernel_size: usize,
    #[config(default = 4)]
    n_attention_heads: usize,
    #[config(default = 4)]
    n_reasoning_layers: usize,
    #[config(default = 4)]
    n_distillation_layers: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl ThinkingLayerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> ThinkingLayer<B> {
        ThinkingLayer {
            encoder: TransformerEncoderConfig::new(
                self.input_n_channels,
                self.input_size,
                self.n_attention_heads,
                self.n_reasoning_layers,
            )
            .with_dropout(self.dropout)
            .init(device),
            distill: DistillationLayerConfig::new(
                self.input_size,
                self.input_n_channels,
                self.output_size,
                self.output_n_channels,
            )
            .with_n_stacks(self.n_distillation_layers)
            .with_kernel_size(self.kernel_size)
            .with_dropout(self.dropout.clone())
            .build(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Rooney<B: Backend> {
    ingress: ExpansionLayer<B>,
    stacks: Vec<ThinkingLayer<B>>,
}

impl<B: Backend> Rooney<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let buffer = self.ingress.forward(input.transpose()).transpose();
        softmax(
            self.stacks
                .iter()
                .fold(buffer, |acc, stack| stack.forward(acc)),
            2,
        )
    }
}

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

        let loss = MseLoss::new().forward(pass.clone(), targets.clone(), Reduction::Mean);

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

        let loss = MseLoss::new().forward(pass.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, pass, targets)
    }
}

#[derive(Config, Debug)]
pub struct RooneyConfig {
    input_size: usize,
    input_n_channels: usize,
    output_size: usize,
    output_n_channels: usize,

    #[config(default = 16)]
    latent_size: usize,
    #[config(default = 5)]
    kernel_size: usize,
    #[config(default = 4)]
    n_expansion_stacks: usize,
    #[config(default = 4)]
    n_thinking_stacks: usize,
    #[config(default = 4)]
    n_attention_heads: usize,
    #[config(default = 4)]
    n_reasoning_layers: usize,
    #[config(default = 4)]
    n_distillation_layers: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl RooneyConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> Rooney<B> {
        let projection_chain: Vec<(usize, usize)> = (0..(self.n_thinking_stacks + 1))
            .map(|i_chain| {
                let weight = i_chain as f64 / self.n_thinking_stacks as f64;
                (
                    ((1.0 - weight) * self.input_size as f64 + weight * self.output_size as f64)
                        as usize,
                    ((1.0 - weight) * self.latent_size as f64
                        + weight * self.output_n_channels as f64) as usize,
                )
            })
            .collect();
        Rooney {
            ingress: ExpansionLayerConfig::new(self.input_n_channels, self.latent_size)
                .with_n_stacks(self.n_expansion_stacks)
                .with_kernel_size(self.kernel_size)
                .with_dropout(self.dropout.clone())
                .build(device),
            stacks: (0..self.n_thinking_stacks)
                .map(|i_chain| {
                    ThinkingLayerConfig::new(
                        projection_chain[i_chain].0,
                        projection_chain[i_chain].1,
                        projection_chain[i_chain + 1].0,
                        projection_chain[i_chain + 1].1,
                    )
                    .with_kernel_size(self.kernel_size)
                    .with_n_attention_heads(self.n_attention_heads.clone())
                    .with_n_reasoning_layers(self.n_reasoning_layers.clone())
                    .with_n_distillation_layers(self.n_distillation_layers.clone())
                    .with_dropout(self.dropout.clone())
                    .build(device)
                })
                .collect(),
        }
    }
}
