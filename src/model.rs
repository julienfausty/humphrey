use burn::config::Config;
use burn::module::Module;
use burn::nn::modules::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
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
pub struct DistillationLayer<B: Backend> {
    stacks: Vec<(Linear<B>, Relu)>,
    dropout: Dropout,
}

impl<B: Backend> DistillationLayer<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.stacks.iter().fold(input, |acc, stack| {
            self.dropout.forward(stack.1.forward(stack.0.forward(acc)))
        })
    }
}

#[derive(Config, Debug)]
pub struct DistillationLayerConfig {
    input_size: usize,
    output_size: usize,
    #[config(default = 4)]
    n_stacks: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl DistillationLayerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> DistillationLayer<B> {
        let projection_chain: Vec<usize> = (0..(self.n_stacks + 1))
            .map(|i_chain| {
                let weight = i_chain as f64 / self.n_stacks as f64;
                ((1.0 - weight) * self.input_size as f64 + weight * self.output_size as f64)
                    as usize
            })
            .collect();

        DistillationLayer {
            stacks: (0..self.n_stacks)
                .map(|i_chain| {
                    (
                        LinearConfig::new(projection_chain[i_chain], projection_chain[i_chain + 1])
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
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, 1> {
        let formatted = if D > 3 {
            input.reshape([0, 0, -1])
        } else if D == 2 {
            input.reshape([0, 0, 1])
        } else if D == 1 {
            input.reshape([0, 1, 1])
        } else {
            input.reshape([0, 0, 0])
        };

        let buffer = self
            .attention
            .forward(MhaInput::self_attn(formatted))
            .context
            .flatten(0, -1);
        let buffer = self.dropout.forward(buffer);
        let buffer = self.reason.forward(buffer);
        let buffer = self.dropout.forward(buffer);
        let buffer = self.distill.forward(buffer);
        self.dropout.forward(buffer)
    }
}

#[derive(Config, Debug)]
pub struct ThinkingLayerConfig {
    sequence_size: usize,
    raw_dimension: usize,
    output_size: usize,

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
        let embedding_size = self.sequence_size * self.raw_dimension;
        ThinkingLayer {
            attention: MultiHeadAttentionConfig::new(self.raw_dimension, self.n_attention_heads)
                .with_dropout(self.dropout.clone())
                .init(device),
            reason: ReasoningLayerConfig::new(embedding_size)
                .with_n_stacks(self.n_reasoning_layers)
                .with_dropout(self.dropout.clone())
                .build(device),
            distill: DistillationLayerConfig::new(embedding_size, self.output_size)
                .with_n_stacks(self.n_distillation_layers)
                .with_dropout(self.dropout.clone())
                .build(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Rooney<B: Backend> {
    ingress: ThinkingLayer<B>,
    stacks: Vec<ThinkingLayer<B>>,
}

impl<B: Backend> Rooney<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, 1> {
        let buffer = self.ingress.forward(input);
        self.stacks
            .iter()
            .fold(buffer, |acc, stack| stack.forward(acc))
    }
}

#[derive(Config, Debug)]
pub struct RooneyConfig {
    sequence_size: usize,
    raw_dimension: usize,
    output_size: usize,

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
        let embedding_size = self.sequence_size * self.raw_dimension;
        let projection_chain: Vec<usize> = (0..(self.n_thinking_stacks + 1))
            .map(|i_chain| {
                let weight = i_chain as f64 / self.n_thinking_stacks as f64;
                ((1.0 - weight) * embedding_size as f64 + weight * self.output_size as f64) as usize
            })
            .collect();
        Rooney {
            ingress: ThinkingLayerConfig::new(
                self.sequence_size,
                self.raw_dimension,
                projection_chain[1],
            )
            .with_n_attention_heads(self.n_attention_heads.clone())
            .with_n_reasoning_layers(self.n_reasoning_layers.clone())
            .with_n_distillation_layers(self.n_distillation_layers.clone())
            .with_dropout(self.dropout.clone())
            .build(device),
            stacks: (1..self.n_thinking_stacks)
                .map(|i_chain| {
                    ThinkingLayerConfig::new(
                        projection_chain[i_chain],
                        1,
                        projection_chain[i_chain + 1],
                    )
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
