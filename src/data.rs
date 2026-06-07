use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{Mapper, MapperDataset, SelectionDataset};
use burn::prelude::s;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};

use csv;
use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

/// In memory structure for a line of CSV data
#[derive(Debug, serde::Deserialize)]
#[allow(non_snake_case)]
struct McZielinskiOHLC {
    pub Timestamp: f32,
    pub Low: f32,
    pub Open: f32,
    pub Close: f32,
    pub High: f32,
    pub Volume: f32,
}

/// Function for parsing McZielinski formated OHLC data from a readable object
fn parse<T: Read>(source: T) -> Result<Vec<f32>, String> {
    let mut ohlc_data = Vec::new();
    let mut reader = csv::Reader::from_reader(source);
    for parsed in reader.deserialize() {
        let ohlc: McZielinskiOHLC = match parsed {
            Ok(ohlc) => ohlc,
            Err(err) => return Err(format!("Error parsing csv data:\n{:?}", err)),
        };

        ohlc_data.push(ohlc.Timestamp);
        ohlc_data.push(ohlc.Low);
        ohlc_data.push(ohlc.Open);
        ohlc_data.push(ohlc.Close);
        ohlc_data.push(ohlc.High);
        ohlc_data.push(ohlc.Volume);
    }
    Ok(ohlc_data)
}

/// A item pair of OHLC block data and data that immediately follows it
#[derive(Debug, Clone)]
pub struct OHLCItem<B: Backend> {
    pub block: Tensor<B, 2>,
    pub next: Tensor<B, 2>,
}

/// Object implementing the Dataset trait containing the entire raw OHLC data base (on device)
#[derive(Debug, Clone)]
pub struct OHLCDataset<B: Backend> {
    block_size: usize,
    loaded: Tensor<B, 2>,
}

impl<B: Backend> OHLCDataset<B> {
    /// Constructor for the dataset that takes a readable source of data and a desired block size
    /// Data is read into CPU memory and copied over to device memory in tensor format
    pub fn new<Src: Read>(
        block_size: usize,
        source: Src,
        device: &B::Device,
    ) -> Result<OHLCDataset<B>, String> {
        let raw = match parse(source) {
            Ok(raw) => raw,
            Err(message) => return Err(message),
        };

        let n_time_steps = raw.len() / 6;

        if n_time_steps < 2 * block_size {
            return Err(format!(
                "Cannot create dataset with block_size {block_size} and total size {n_time_steps} because toal size is too small."
            ));
        }

        Ok(OHLCDataset {
            block_size,
            loaded: Tensor::<B, 2>::from_data(TensorData::new(raw, [n_time_steps, 6]), device),
        })
    }
}

impl<B: Backend> Dataset<OHLCItem<B>> for OHLCDataset<B> {
    fn get(&self, index: usize) -> Option<OHLCItem<B>> {
        if index >= self.len() {
            return None;
        }

        Some(OHLCItem {
            block: self
                .loaded
                .clone()
                .slice([index..(index + self.block_size)]),
            next: self
                .loaded
                .clone()
                .slice([(index + self.block_size)..(index + 2 * self.block_size)]),
        })
    }

    fn len(&self) -> usize {
        self.loaded.shape()[0] - 2 * self.block_size + 1
    }
}

/// Normalizer for OHLCItem data towards values varying around 1.0
pub struct NormalizeOHLCItem;

impl<B: Backend> Mapper<OHLCItem<B>, OHLCItem<B>> for NormalizeOHLCItem {
    fn map(&self, item: &OHLCItem<B>) -> OHLCItem<B> {
        let first_timestamp = item.block.clone().slice(s![0, 0]);
        let last_timestamp = item.block.clone().slice(s![-1, 0]);
        let range = last_timestamp - first_timestamp.clone();

        let block_times =
            (item.block.clone().slice(s![0.., 0]) - first_timestamp.clone()).div(range.clone());
        let next_times = (item.next.clone().slice(s![0.., 0]) - first_timestamp).div(range);

        let block_vols = item.block.clone().slice(s![0.., 5]);
        let max_vol = block_vols.clone().max().into_scalar();

        let block_vols = block_vols.div_scalar(max_vol.clone());
        let next_vols = item.next.clone().slice(s![0.., 5]).div_scalar(max_vol);

        let close = item.block.clone().slice(s![-1, 3]).into_scalar();

        let block_prices = (item.block.clone().slice(s![0.., 1..5])).div_scalar(close.clone());
        let next_prices = (item.next.clone().slice(s![0.., 1..5])).div_scalar(close.clone());

        let block = item.block.clone().zeros_like();
        let next = item.next.clone().zeros_like();

        let block = block.slice_assign(s![0.., 0], block_times);
        let next = next.slice_assign(s![0.., 0], next_times);
        let block = block.slice_assign(s![0.., 1..5], block_prices);
        let next = next.slice_assign(s![0.., 1..5], next_prices);
        let block = block.slice_assign(s![0.., 5], block_vols);
        let next = next.slice_assign(s![0.., 5], next_vols);

        OHLCItem { block, next }
    }
}

pub struct ToInnerBackend;

impl<B: AutodiffBackend> Mapper<OHLCItem<B>, OHLCItem<B::InnerBackend>> for ToInnerBackend {
    fn map(&self, item: &OHLCItem<B>) -> OHLCItem<B::InnerBackend> {
        OHLCItem {
            block: item.block.clone().inner(),
            next: item.next.clone().inner(),
        }
    }
}

/// A batch of OHLCItems
#[derive(Debug, Clone)]
pub struct OHLCBatch<B: Backend> {
    pub blocks: Tensor<B, 3>,
    pub nexts: Tensor<B, 3>,
}

/// A structure implementing the Batcher trait for combining OHLCItems into OHLCBatches
#[derive(Clone, Default)]
pub struct OHLCBatcher {}

impl<B: Backend> Batcher<B, OHLCItem<B>, OHLCBatch<B>> for OHLCBatcher {
    fn batch(&self, items: Vec<OHLCItem<B>>, _device: &B::Device) -> OHLCBatch<B> {
        OHLCBatch {
            blocks: Tensor::stack(items.iter().map(|item| item.block.clone()).collect(), 0),
            nexts: Tensor::stack(items.iter().map(|item| item.next.clone()).collect(), 0),
        }
    }
}

// Implements a projection operation from normalized OHLC data to price distribution between (0, 2). Projection takes into account volumes and discount from t = 0.0
pub struct OHLC2Distribution;

impl OHLC2Distribution {
    pub fn project<B: Backend>(&self, ohlc: Tensor<B, 3>, grid_size: usize) -> Tensor<B, 2> {
        let prices = ohlc
            .clone()
            .slice(s![0.., 0.., 1..5])
            .sort(2)
            .clamp(0.0, 2.0);
        let time_discount = ohlc
            .clone()
            .slice(s![0.., 0.., 0])
            .div_scalar(-1.0 / 3.0)
            .exp();
        let volumes = ohlc.clone().slice(s![0.., 0.., 5]);

        let anchors = prices.clone().matmul(
            Tensor::<B, 3>::from_data(
                [[
                    [1.0 / 6.0, -1.0 / 4.0],
                    [1.0 / 3.0, -1.0 / 4.0],
                    [1.0 / 3.0, 1.0 / 4.0],
                    [1.0 / 6.0, 1.0 / 4.0],
                ]],
                &prices.clone().device(),
            )
            .repeat_dim(0, prices.shape()[0]),
        );

        // Add epsilon to delta values to avoid zero division
        let anchors = anchors
            + Tensor::<B, 3>::from_data([[[0.0, 1e-16]]], &prices.clone().device())
                .repeat_dim(1, prices.shape()[1])
                .repeat_dim(0, prices.shape()[0]);

        let a = (grid_size - 1) as f64;
        let c = anchors.clone().slice(s![0.., 0.., 1]).recip();
        let d = anchors.clone().slice(s![0.., 0.., 0]).mul(c.clone());

        let a2c2 = c.clone().powf_scalar(2.0) + a.powf(2.0);

        let integral_measure = a2c2
            .clone()
            .mul_scalar(8.0 * std::f64::consts::PI)
            .sqrt()
            .recip()
            .mul_scalar(a)
            .mul(time_discount)
            .mul(volumes);

        let exponential_term = (0..grid_size).fold(
            Tensor::<B, 3>::zeros(
                [prices.shape()[0], prices.shape()[1], grid_size],
                &prices.device(),
            ),
            |acc, i_grid| {
                let b = 2.0 * (i_grid as f64);
                let contribution = (c.clone().mul_scalar(b) - d.clone().mul_scalar(a))
                    .powf_scalar(2.0)
                    .div(a2c2.clone().mul_scalar(2))
                    .neg()
                    .exp();
                acc.clone().slice_assign(
                    s![0.., 0.., i_grid],
                    contribution + acc.clone().slice(s![0.., 0.., i_grid]),
                )
            },
        );

        let erf_term = (0..grid_size).fold(
            Tensor::<B, 3>::zeros(
                [prices.shape()[0], prices.shape()[1], grid_size],
                &prices.device(),
            ),
            |acc, i_grid| {
                let b = 2.0 * (i_grid as f64);
                let sqrt_two_a2c2 = a2c2.clone().mul_scalar(2.0).sqrt();
                let ab_plus_cd = c.clone().mul(d.clone()) + a * b;
                let contribution = (a2c2.clone().mul_scalar(2.0) - ab_plus_cd.clone())
                    .div(sqrt_two_a2c2.clone())
                    .erf()
                    - ab_plus_cd.clone().neg().div(sqrt_two_a2c2.clone()).erf();
                acc.clone().slice_assign(
                    s![0.., 0.., i_grid],
                    contribution + acc.clone().slice(s![0.., 0.., i_grid]),
                )
            },
        );

        let convolution = integral_measure
            .repeat_dim(2, grid_size)
            .mul(exponential_term)
            .mul(erf_term);

        let convolution = convolution.sum_dim(1).reshape([0, -1]);
        convolution
            .clone()
            .div(convolution.clone().sum_dim(1).repeat_dim(1, grid_size))
    }
}

/// Utility struct for configuring the train/test data split and providing data loaders for training
#[derive(Config, Debug)]
pub struct DataConfig {
    #[config(default = 1.0)]
    pub use_only: f32,
    #[config(default = 0.8)]
    pub train_split: f32,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 64)]
    pub window_size: usize,
    #[config(default = 42)]
    pub seed: u64,
}

impl DataConfig {
    /// Method for coalescing the builder pattern into the train and test data (respectively)
    /// User must provide a readable source formatted in McZielinski CSV style
    pub fn build<R: Read, B: AutodiffBackend>(
        &self,
        source: R,
        device: &B::Device,
    ) -> Result<
        (
            Arc<dyn DataLoader<B, OHLCBatch<B>>>,
            Arc<dyn DataLoader<B::InnerBackend, OHLCBatch<B::InnerBackend>>>,
        ),
        String,
    > {
        let base_dataset = match OHLCDataset::<B>::new(self.window_size.clone(), source, &device) {
            Ok(ds) => ds,
            Err(message) => return Err(message),
        };

        let visible_portion = (base_dataset.len() as f32 * self.use_only) as usize;
        let block_size = 4 * self.window_size + self.batch_size.clone();
        let total_visible_blocks = visible_portion / block_size.clone();
        let number_test_blocks =
            (total_visible_blocks.clone() as f32 * (1.0 - self.train_split)) as usize;

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.clone());
        let test_blocks: HashSet<usize> = (0..number_test_blocks)
            .map(|_| rng.random_range(0..total_visible_blocks.clone()))
            .collect();

        let train_blocks: Vec<usize> = (0..total_visible_blocks)
            .filter(|i_block| !test_blocks.contains(i_block))
            .collect();
        let test_blocks = Vec::from_iter(test_blocks.into_iter());

        let unroll = |set: Vec<usize>| {
            let mut unrolled = Vec::with_capacity(set.len() * block_size.clone());
            let origin = base_dataset.len();
            for i_block in set.into_iter() {
                let offset = block_size.clone() * i_block;
                for j_inner in 0..block_size.clone() {
                    unrolled.push(origin - offset - j_inner - 1)
                }
            }
            unrolled
        };

        let batcher = OHLCBatcher {};

        let train_set = DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size.clone())
            .shuffle(self.seed.clone())
            .num_workers(self.num_workers.clone())
            .build(SelectionDataset::from_indices_unchecked(
                MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem),
                unroll(train_blocks),
            ));

        let test_set = DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size.clone())
            .shuffle(self.seed.clone())
            .num_workers(self.num_workers.clone())
            .build(SelectionDataset::from_indices_unchecked(
                MapperDataset::new(
                    MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem),
                    ToInnerBackend,
                ),
                unroll(test_blocks),
            ));

        Ok((train_set, test_set))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use integrate::adaptive_quadrature::adaptive_simpson_method;

    const VALID_TEST_STRING: &'static str = "\
        Timestamp,Low,Open,Close,High,Volume\n\
        1.0,2.0,3.0,4.0,5.0,6.0\n\
        7.0,8.0,9.0,10.0,11.0,12.0\n\
        13.0,14.0,15.0,16.0,17.0,18.0\n\
        19.0,20.0,21.0,22.0,23.0,24.0";

    const EPS: f32 = 1e-6;

    #[test]
    fn test_valid_construction_block_1() {
        let device = NdArrayDevice::default();
        let result = OHLCDataset::<NdArray>::new(1, VALID_TEST_STRING.as_bytes(), &device);
        assert!(
            result.is_ok(),
            "Failed to construct OHLCDataset {:?}",
            result
        );
    }

    #[test]
    fn test_valid_construction_block_2() {
        let device = NdArrayDevice::default();
        let result = OHLCDataset::<NdArray>::new(2, VALID_TEST_STRING.as_bytes(), &device);
        assert!(
            result.is_ok(),
            "Failed to construct OHLCDataset {:?}",
            result
        );
    }

    #[test]
    fn test_construction_block_too_large() {
        let device = NdArrayDevice::default();
        let result = OHLCDataset::<NdArray>::new(3, VALID_TEST_STRING.as_bytes(), &device);
        assert!(result.is_err(), "Created dataset with too large of a block");
    }

    #[test]
    fn test_construction_bad_format() {
        let device = NdArrayDevice::default();
        let result = OHLCDataset::<NdArray>::new(3, "Not right csv format".as_bytes(), &device);
        assert!(
            result.is_err(),
            "Created dataset from badly formatted string"
        );
    }

    #[test]
    fn test_len_block_1() {
        let device = NdArrayDevice::default();
        let Ok(dataset) = OHLCDataset::<NdArray>::new(1, VALID_TEST_STRING.as_bytes(), &device)
        else {
            panic!("Failed to construct valid dataset");
        };

        assert_eq!(3, dataset.len());
    }

    #[test]
    fn test_len_block_2() {
        let device = NdArrayDevice::default();
        let Ok(dataset) = OHLCDataset::<NdArray>::new(2, VALID_TEST_STRING.as_bytes(), &device)
        else {
            panic!("Failed to construct valid dataset");
        };

        assert_eq!(1, dataset.len());
    }

    #[test]
    fn test_gets_block_1() {
        let device = NdArrayDevice::default();
        let Ok(dataset) = OHLCDataset::<NdArray>::new(1, VALID_TEST_STRING.as_bytes(), &device)
        else {
            panic!("Failed to construct valid dataset");
        };

        let get_opt = dataset.get(0);
        assert!(get_opt.is_some());

        let got = get_opt.unwrap();

        assert!(
            got.block
                .equal(Tensor::<NdArray, 2>::from_data(
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                    &device
                ))
                .all()
                .into_scalar()
        );

        assert!(
            got.next
                .equal(Tensor::<NdArray, 2>::from_data(
                    [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
                    &device
                ))
                .all()
                .into_scalar()
        );

        let get_opt = dataset.get(2);
        assert!(get_opt.is_some());

        let got = get_opt.unwrap();

        assert!(
            got.block
                .equal(Tensor::<NdArray, 2>::from_data(
                    [[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]],
                    &device
                ))
                .all()
                .into_scalar()
        );

        assert!(
            got.next
                .equal(Tensor::<NdArray, 2>::from_data(
                    [[19.0, 20.0, 21.0, 22.0, 23.0, 24.0]],
                    &device
                ))
                .all()
                .into_scalar()
        );

        assert!(dataset.get(42).is_none());
    }

    #[test]
    fn test_gets_block_2() {
        let device = NdArrayDevice::default();
        let Ok(dataset) = OHLCDataset::<NdArray>::new(2, VALID_TEST_STRING.as_bytes(), &device)
        else {
            panic!("Failed to construct valid dataset");
        };

        let get_opt = dataset.get(0);
        assert!(get_opt.is_some());

        let got = get_opt.unwrap();

        assert!(
            got.block
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                    ],
                    &device
                ))
                .all()
                .into_scalar()
        );

        assert!(
            got.next
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
                    ],
                    &device
                ))
                .all()
                .into_scalar()
        );

        assert!(dataset.get(1).is_none());
    }

    #[test]
    fn test_normalize_simple() {
        let device = NdArrayDevice::default();
        let test_item = OHLCItem::<NdArray> {
            block: Tensor::<NdArray, 2>::from_data(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 10.0, 9.0, 11.0],
                ],
                &device,
            ),
            next: Tensor::<NdArray, 2>::from_data(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 10.0, 9.0, 11.0],
                ],
                &device,
            ),
        };

        let normalizer = NormalizeOHLCItem {};
        let normalized = normalizer.map(&test_item);

        assert!(
            normalized
                .block
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [0.0, 0.1, 0.2, 0.3, 0.4, 5.0 / 11.0],
                        [1.0, 0.7, 0.8, 1.0, 0.9, 1.0],
                    ],
                    &device,
                ))
                .all()
                .into_scalar()
        );

        assert!(
            normalized
                .next
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [0.0, 0.1, 0.2, 0.3, 0.4, 5.0 / 11.0],
                        [1.0, 0.7, 0.8, 1.0, 0.9, 1.0],
                    ],
                    &device,
                ))
                .all()
                .into_scalar()
        );
    }

    #[test]
    fn test_normalize_long() {
        let device = NdArrayDevice::default();
        let test_item = OHLCItem::<NdArray> {
            block: Tensor::<NdArray, 2>::from_data(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                    [18.0, 19.0, 20.0, 22.0, 21.0, 23.0],
                ],
                &device,
            ),
            next: Tensor::<NdArray, 2>::from_data(
                [
                    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
                    [36.0, 37.0, 38.0, 39.0, 40.0, 41.0],
                    [42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                    [48.0, 49.0, 50.0, 52.0, 51.0, 53.0],
                ],
                &device,
            ),
        };

        let normalizer = NormalizeOHLCItem {};
        let normalized = normalizer.map(&test_item);

        assert!(
            normalized
                .block
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [
                            0.0,
                            1.0 / 22.0,
                            2.0 / 22.0,
                            3.0 / 22.0,
                            4.0 / 22.0,
                            5.0 / 23.0
                        ],
                        [
                            6.0 / 18.0,
                            7.0 / 22.0,
                            8.0 / 22.0,
                            9.0 / 22.0,
                            10.0 / 22.0,
                            11.0 / 23.0
                        ],
                        [
                            12.0 / 18.0,
                            13.0 / 22.0,
                            14.0 / 22.0,
                            15.0 / 22.0,
                            16.0 / 22.0,
                            17.0 / 23.0
                        ],
                        [1.0, 19.0 / 22.0, 20.0 / 22.0, 1.0, 21.0 / 22.0, 1.0],
                    ],
                    &device,
                ))
                .all()
                .into_scalar()
        );

        assert!(
            normalized
                .next
                .equal(Tensor::<NdArray, 2>::from_data(
                    [
                        [
                            30.0 / 18.0,
                            31.0 / 22.0,
                            32.0 / 22.0,
                            33.0 / 22.0,
                            34.0 / 22.0,
                            35.0 / 23.0
                        ],
                        [
                            36.0 / 18.0,
                            37.0 / 22.0,
                            38.0 / 22.0,
                            39.0 / 22.0,
                            40.0 / 22.0,
                            41.0 / 23.0
                        ],
                        [
                            42.0 / 18.0,
                            43.0 / 22.0,
                            44.0 / 22.0,
                            45.0 / 22.0,
                            46.0 / 22.0,
                            47.0 / 23.0
                        ],
                        [
                            48.0 / 18.0,
                            49.0 / 22.0,
                            50.0 / 22.0,
                            52.0 / 22.0,
                            51.0 / 22.0,
                            53.0 / 23.0
                        ],
                    ],
                    &device,
                ))
                .all()
                .into_scalar()
        );
    }

    #[test]
    fn test_batch_items() {
        let device = NdArrayDevice::default();
        let test_item = OHLCItem::<NdArray> {
            block: Tensor::<NdArray, 2>::from_data(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                    [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                ],
                &device,
            ),
            next: Tensor::<NdArray, 2>::from_data(
                [
                    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
                    [36.0, 37.0, 38.0, 39.0, 40.0, 41.0],
                    [42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                    [48.0, 49.0, 50.0, 51.0, 52.0, 53.0],
                ],
                &device,
            ),
        };

        let batcher = OHLCBatcher {};
        let batched = batcher.batch(
            vec![test_item.clone(), test_item.clone(), test_item.clone()],
            &device,
        );

        assert!(batched.blocks.shape().dims() == [3, 4, 6]);
        assert!(batched.nexts.shape().dims() == [3, 4, 6]);

        for i_dim in 0..3 {
            assert!(
                batched
                    .blocks
                    .clone()
                    .slice(s![i_dim, .., ..])
                    .reshape([4, 6])
                    .equal(test_item.block.clone())
                    .all()
                    .into_scalar()
            );

            assert!(
                batched
                    .nexts
                    .clone()
                    .slice(s![i_dim, .., ..])
                    .reshape([4, 6])
                    .equal(test_item.next.clone())
                    .all()
                    .into_scalar()
            );
        }
    }

    #[test]
    fn test_simple_ohlc_2_distribution() {
        let projector = OHLC2Distribution;

        let device = NdArrayDevice::default();

        let test_ohlc = Tensor::<NdArray, 1>::from_data([0.0, 0.0, 0.0, 2.0, 2.0, 1.0], &device)
            .reshape([1, 1, 6]);

        let projection = projector.project(test_ohlc, 10);

        assert!(projection.shape().len() == 2);
        assert!(projection.shape()[0] == 1);
        assert!(projection.shape()[1] == 10);

        assert!(projection.clone().sum().into_scalar() == 1.0);
        let projection_data: Vec<f32> = projection.clone().to_data().into_vec().unwrap();

        let integrand = |p: f64, i_grid: usize| {
            (9.0 / (2.0 * std::f64::consts::PI))
                * (((-1.0 / 2.0) * ((9.0 * p - 2.0 * (i_grid as f64)).powf(2.0))).exp())
                * (((-1.0 / 2.0) * ((p - 1.0).powf(2.0))).exp())
        };

        let reference = (0..10).map(|i_grid| {
            adaptive_simpson_method(
                |p: f32| integrand(p as f64, i_grid) as f32,
                0.0,
                2.0,
                1e-6,
                EPS,
            )
            .unwrap()
        });
        let sum: f32 = reference.clone().sum();
        let reference: Vec<f32> = reference.map(|val| val / sum).collect();
        for i_grid in 0..10 {
            assert!((reference[i_grid] - projection_data[i_grid]).powf(2.0) < EPS);
        }
    }

    #[test]
    fn test_ohlc_to_distribution() {
        let projector = OHLC2Distribution;

        let device = NdArrayDevice::default();

        let test_ohlc = Tensor::<NdArray, 2>::from_data(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 1.0],
                [0.1, 0.1, 0.3, 0.6, 1.0, 0.4],
            ],
            &device,
        )
        .reshape([1, 2, 6]);

        let projection = projector.project(test_ohlc, 10);

        assert!(projection.shape().len() == 2);
        assert!(projection.shape()[0] == 1);
        assert!(projection.shape()[1] == 10);

        assert!(projection.clone().sum().into_scalar() == 1.0);
        let projection_data: Vec<f32> = projection.clone().to_data().into_vec().unwrap();

        let integrand = |p: f64, i_grid: usize| {
            (9.0 / (2.0 * std::f64::consts::PI))
                * (((-1.0 / 2.0) * ((9.0 * p - 2.0 * (i_grid as f64)).powf(2.0))).exp())
                * (((-1.0 / 2.0) * ((p - 1.0).powf(2.0))).exp()
                    + ((-0.1 * 3.0) as f64).exp()
                        * 0.4
                        * ((-1.0 / 2.0) * (((p - 0.483333333333333) / (0.3)).powf(2.0))).exp())
        };

        let reference = (0..10).map(|i_grid| {
            adaptive_simpson_method(
                |p: f32| integrand(p as f64, i_grid) as f32,
                0.0,
                2.0,
                1e-6,
                EPS,
            )
            .unwrap()
        });
        let sum: f32 = reference.clone().sum();
        let reference: Vec<f32> = reference.map(|val| val / sum).collect();

        for i_grid in 0..10 {
            assert!((reference[i_grid] - projection_data[i_grid]).powf(2.0) < EPS);
        }
    }

    #[test]
    fn test_random_ohlc_projection() {
        let projector = OHLC2Distribution;

        let device = NdArrayDevice::default();

        let test_ohlc = Tensor::<NdArray, 3>::random(
            [4, 3, 6],
            burn::tensor::Distribution::Uniform(0.1, 2.0),
            &device,
        );

        let projection = projector.project(test_ohlc, 10);

        assert!(projection.shape().len() == 2);
        assert!(projection.shape()[0] == 4);
        assert!(projection.shape()[1] == 10);

        for i_batch in 0..4 {
            assert!(
                projection
                    .clone()
                    .slice(s![i_batch, 0..])
                    .sum()
                    .into_scalar()
                    .powf(2.0)
                    - 1.0
                    < EPS
            );
        }
    }
}
