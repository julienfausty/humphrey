# rooney
An AI analyst capable of identifying buy and sell signals in crypto OHLC data.

## Goal

The goal of the analyst is to predict the asset price probablity distribution over the next hour given the OHLC data over the previous hour.

The output of the model should be a probablity density function:
```math
\begin{split}
  \phi: & [0, 2] \rightarrow [0, 1]\\
     & \phi(p) = P(p | [t, t + \Delta t])
\end{split}
```

where `p` is a normalized price, `t` is a time coordinate and `P` is the probability density for that asset to be at price `p` over the provided time range.


## Setup

This project is a rust project built on top of the [burn](https://github.com/tracel-ai/burn) framework. It uses `cargo` for its build system. As such,

for building:
```shell
cargo build --release
```

for running training:
```shell
cargo run --release --bin train
```

## Data

Data used to train the model can be found at: [https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data]

It is 1-minute OHLC Bitcoin data from 2012 to present day sourced from the [Bitstamp API](https://www.bitstamp.net/api/) and updated regularly.

The data is contained in a single csv file with following columns in order:
- Timestamp
- Open
- High
- Low
- Close
- Volume

## Preprocessing

Data is read into disk and immediate stored in a (N, 6) tensor on device memory.

For training, the data is windowed (following a user defined window size) and each window is normalized around unity as:

```math
\begin{split}
\tilde{t} &= \frac{(t - t_{start})}{(t_{end} - t_{start})}\\
\tilde{p} &= \frac{p}{p_{max}}\\
\tilde{v} &= \frac{v}{v_{max}}
\end{split}
```

where:
* $t$, $p$ and $v$ are time, price (low, high, open and close together) and volume respectively, and
* tilded quantities are normalized (in the rest of the document, we will only use normalized quantities unless specified and so the tilde is omitted)

Each item passed to training contains an OHLC data window and the next unoverlapping window normalized with the values from the current window for evaluating loss and predictive power.

Data is split by:
* first, chunking the data set into blocks of size `4 x window_size + block_size` where the block size is a user parameter,
* then, splitting the blocks randomly into a test and train sets,
* with finally, unblocking the data into individual items once again.

This chunking operation is performed because the windowing of the dataset. If sampled randomly, the majority of portions of the dataset in the train set and test set will overlap. This would not make the test evaluation very trustworthy since the model sees a large portion of it during training. Chunking the dataset in this manner ensures that only the head and tail of chunks overlap between the train and test sets limiting this effect. The larger the block size chosen, the more this effect is mitigated but the less the entire dataset is sampled over the enitre time series.

Data is batched for training and testing to parallelize and speed up the training process using a `batch_size` user provided parameter.

## Evaluation

The model will ultimately output a tensor with floating point values. The evaluation of the loss during training will ultimately determine the meaning of these values and how they evolve as the model improves.

Given that the model is attempting to reconstruct a probability density function, we will use a set of basis functions $\mathbf{G} = \{g_{i}: [0, 2] \rightarrow [0, 1]\}$ such that:
```math
\phi(p) = \omega^{i}g_{i}(p)
```
where $\omega^{i}$ are the weights predicted by the model for a given OHLC window and we are using an implicit Einstein summation notation.

Given the probabilistic nature of the problem, our first approach is to use gaussian kernel functions as the basis functions on a regular grid for the framing of the solution. As such, for a given grid size $N \gt 1$:
```math
g_{i\in[0, N[}(p \in [0, 2]) = \dfrac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(p - 2i/(N-1))^2}{2\sigma^2}}
```

where we choose $\sigma$ such that neighboring kernels overlap at one $\sigma$ intervals:

```math
\sigma = \frac{1}{N-1}
```

such that:
```math
g_{i\in[0, N[}(p \in [0, 2]) = \dfrac{N-1}{\sqrt{2\pi}}e^{-\frac{(p(N-1) - 2i)^2}{2}}
```

In order to satisfy the constraints of a probability density function:
* $\phi(p) > 0, \forall p$
* $\int \phi dp = 1$

we impose that $\omega_{i} > 0, \forall i$ and $\sum_{0}^{N-1}\omega_{i} = 1$.

To construct the target probability density function $\tilde{\phi}$, we take the OHLC data from the hour after the window used for the prediction and convolve the values into the probability density space. We associate a candlestick with 3 uniform distribution of prices: low-min / open-close / max-high where we consider a one sixth / two thirds / one sixth split of the volume:

```math
\chi[o,h,l,c](p) = \left\{
\begin{array}{l}
\frac{1}{6(\min(o,c) - l)}, \, p \in [l, \min(o, c)]\\
\frac{2}{3(|c - o|)}, \, p \in [o, c]\\
\frac{1}{6(h - \max(o, c))}, \, p [\max(o,c), h]
\end{array}
\right .
```

where the $(o, h, l, c)$ stand for open, high, low and close prices.

It would be best if the resulting target probability density function was:
* biased towards earlier times (i.e. more sensitive to prices occuring sooner), and
* sensitive relative to the volume of trades actually occuring

for usefulness in active trading.

For this reason, the influence of each candlestick in the target probability density function is weighted by the volume of trades it represents as well as it's distance from the end of the previous block.

The contribution ultimately befor normalization of each of the candlesticks to each of the target weights $\tilde{\omega}(t)$ is then:

```math
\tilde{\omega}_{i} = \frac{1}{S}\sum_{(o(t+q), h(t+q), l(t+q), c(t+q))} \int_{0}^2 \chi[o(t+q), h(t+q), l(t+q), c(t+q)](p)g_{i}(p) v(t+q) e^{-\frac{q}{\tau}} dp
```

where $S$ is a normalization parameter for keeping the sum of weights equal to 1, $v(t+q)$ is the volume of trades associated with a candlestick and $\tau$ is the dampening associated with the time evolution of the data.

we choose $\tau$ arbitrarily equal to $\frac{1}{3}$.

Given the structure given to the candlestick contribution, in order to calculate the integral in practice, we can use the convolution of the gaussian with a fixed width and height uniform distribution:

```math
\int_{a}^{b} \frac{1}{b-a} g_{i}(p) dp = \frac{1}{b-a} \dfrac{N-1}{\sqrt{2}2} \left(\erf\left(\frac{b(N-1) - 2i}{\sqrt{2}}\right) - \erf\left(\frac{a(N-1) - 2i}{\sqrt{2}}\right)\right)
```

where $\erf$ is the ["error function"](https://mathworld.wolfram.com/Erf.html).

The loss of the model can then computed using the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) for evaluating the non-similarity between probability distributions:
```math
\mathcal{L}(\tilde{\phi}, \phi) = \int_{0}^{2} \tilde{\phi}\log\frac{\tilde{\phi}}{\phi} dp
```

For simplicity, a proxy/approximation of the loss is computed discretely with:
```math
L(\tilde{\phi}, \phi) = \sum_{i = 0}^{N-1} \tilde{\omega}_{i} \log\frac{\tilde{\omega}_{i}}{\omega{i}}
```
