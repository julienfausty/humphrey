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
