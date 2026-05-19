# humphrey
An AI analyst capable of identifying buy and sell signals in crypto OHLC data.

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

The data is contained in a single csv file with following columns:
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
* tilded quantities are normalized

Each item passed to training contains an OHLC data window and the next unoverlapping window normalized with the values from the current window for evaluating loss and predictive power.

Data is split by:
* first, chunking the data set into blocks of size `4 x window_size + block_size` where the block size is a user parameter,
* then, splitting the blocks randomly into a test and train sets,
* with finally, unblocking the data into individual items once again.

This chunking operation is performed because the windowing of the dataset. If sampled randomly, the majority of portions of the dataset in the train set and test set will overlap. This would not make the test evaluation very trustworthy since the model sees a large portion of it during training. Chunking the dataset in this manner ensures that only the head and tail of chunks overlap between the train and test sets limiting this effect. The larger the block size chosen, the more this effect is mitigated but the less the entire dataset is sampled over the enitre time series.

Data is batched for training and testing to parallelize and speed up the training process using a `batch_size` user provided parameter.
