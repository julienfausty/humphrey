use burn::config::Config;
use burn::module::Module;
use burn::nn::modules::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::modules::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ReasoningLayer<B: Backend> {
    stacks: Vec<(Linear<B>, Relu)>,
    dropout: Dropout,
}

impl<B: Backend> ReasoningLayer<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.stacks.iter().fold(input, |acc, stack| {
            self.dropout.forward(stack.1.forward(stack.0.forward(acc)))
        })
    }
}

#[derive(Config, Debug)]
pub struct ReasoningLayerConfig {
    latent_size: usize,
    #[config(default = 4)]
    n_stacks: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl ReasoningLayerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> ReasoningLayer<B> {
        ReasoningLayer {
            stacks: (0..self.n_stacks)
                .map(|_| {
                    (
                        LinearConfig::new(self.latent_size, self.latent_size).init(device),
                        Relu::new(),
                    )
                })
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

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
    stacks: Vec<(Linear<B>, Relu, Conv1d<B>, Relu)>,
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
                        LinearConfig::new(
                            projection_chain[i_chain].0,
                            projection_chain[i_chain + 1].0,
                        )
                        .init(device),
                        Relu::new(),
                        Conv1dConfig::new(
                            projection_chain[i_chain].1,
                            projection_chain[i_chain + 1].1,
                            self.kernel_size.clone(),
                        )
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
    attention: MultiHeadAttention<B>,
    reason: ReasoningLayer<B>,
    distill: DistillationLayer<B>,
    dropout: Dropout,
}

impl<B: Backend> ThinkingLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let buffer = self
            .attention
            .forward(MhaInput::self_attn(input))
            .context
            .transpose();
        let buffer = self.dropout.forward(buffer);
        let buffer = self.reason.forward(buffer);
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
            attention: MultiHeadAttentionConfig::new(self.input_n_channels, self.n_attention_heads)
                .with_dropout(self.dropout.clone())
                .init(device),
            reason: ReasoningLayerConfig::new(self.input_size)
                .with_n_stacks(self.n_reasoning_layers)
                .with_dropout(self.dropout.clone())
                .build(device),
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
