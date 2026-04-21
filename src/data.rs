use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::Mapper;
use burn::prelude::s;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use csv;
use std::io::Read;

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

#[derive(Debug, Clone)]
pub struct OHLCItem<B: Backend> {
    pub block: Tensor<B, 2>,
    pub next: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct OHLCDataset<B: Backend> {
    block_size: usize,
    loaded: Tensor<B, 2>,
}

impl<B: Backend> OHLCDataset<B> {
    pub fn new<Src: Read>(
        block_size: usize,
        source: Src,
        device: &<B as Backend>::Device,
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
        self.loaded.shape().dims[0] - 2 * self.block_size + 1
    }
}

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

        let high = item.block.clone().slice(s![0.., 4]).max().into_scalar();

        let block_prices = (item.block.clone().slice(s![0.., 1..5])).div_scalar(high.clone());
        let next_prices = (item.next.clone().slice(s![0.., 1..5])).div_scalar(high.clone());

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

#[derive(Debug, Clone)]
pub struct OHLCBatch<B: Backend> {
    pub blocks: Tensor<B, 3>,
    pub nexts: Tensor<B, 3>,
}

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

#[cfg(test)]
mod tests {

    use super::*;

    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    const VALID_TEST_STRING: &'static str = "\
        Timestamp,Low,Open,Close,High,Volume\n\
        1.0,2.0,3.0,4.0,5.0,6.0\n\
        7.0,8.0,9.0,10.0,11.0,12.0\n\
        13.0,14.0,15.0,16.0,17.0,18.0\n\
        19.0,20.0,21.0,22.0,23.0,24.0";

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
                    [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                ],
                &device,
            ),
            next: Tensor::<NdArray, 2>::from_data(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
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
                        [1.0, 0.7, 0.8, 0.9, 1.0, 1.0],
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
                        [1.0, 0.7, 0.8, 0.9, 1.0, 1.0],
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
                        [1.0, 19.0 / 22.0, 20.0 / 22.0, 21.0 / 22.0, 1.0, 1.0],
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
                            51.0 / 22.0,
                            52.0 / 22.0,
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
}
