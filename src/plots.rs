use plotters::prelude::*;

use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::MapperDataset;
use burn::prelude::s;

use std::io::{Stdin, stdin};

use std::f64::consts::PI;

mod data;
use data::{NormalizeOHLCItem, OHLC2Distribution, OHLCBatcher, OHLCDataset, OHLCItem};

const ASSET_DIR: &'static str = "assets/";

fn main() -> Result<(), String> {
    let device = NdArrayDevice::default();

    println!("Reading dataset...");
    let base_dataset = OHLCDataset::<NdArray>::new::<Stdin>(64, stdin(), &device).unwrap();
    let dataset: MapperDataset<_, _, OHLCItem<NdArray>> =
        MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem);

    let dset_index = base_dataset.len() - 1;

    println!("Preparing data...");
    let one_batch = OHLCBatcher {}.batch(vec![dataset.get(dset_index).unwrap()], &device);

    let grid_size = 512;

    let prices = (0..grid_size)
        .map(|i_grid| (i_grid as f64) / ((grid_size - 1) as f64))
        .collect::<Vec<_>>();

    let projector = OHLC2Distribution;
    let block_distribution: Vec<f32> = projector
        .project(
            one_batch.blocks.clone().slice_assign(
                s![0.., 0.., 0],
                one_batch.blocks.clone().slice(s![0.., 0.., 0]) - 1.0,
            ),
            grid_size,
        )
        .reshape([grid_size])
        .to_data()
        .to_vec()
        .unwrap();

    let next_distribution: Vec<f32> = projector
        .project(
            one_batch.nexts.clone().slice_assign(
                s![0.., 0.., 0],
                one_batch.nexts.clone().slice(s![0.., 0.., 0]) - 1.0,
            ),
            grid_size,
        )
        .reshape([grid_size])
        .to_data()
        .to_vec()
        .unwrap();

    println!("Plotting...");

    let image_location = format!("{ASSET_DIR}/images/latest_distributions.png");
    let root_area = BitMapBackend::new(&image_location, (1200, 800)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let max_probability = block_distribution
        .iter()
        .fold(0.0, |max, &val| val.max(max))
        .max(next_distribution.iter().fold(0.0, |max, &val| val.max(max)))
        as f64;

    let mut context = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Latest Distribution Pair", ("monospace", 40))
        .build_cartesian_2d(0.0..2.0, 0.0..max_probability)
        .unwrap();

    context.configure_mesh().draw().unwrap();

    let distribution = |x: f64, weights: &Vec<f32>| -> f64 {
        let grid_m1 = grid_size as f64 - 1.0;
        (0..grid_size)
            .map(|i_grid| {
                (weights[i_grid] as f64)
                    * (grid_m1 / (2.0 * PI).sqrt())
                    * ((-1.0 / 2.0) * (x * grid_m1 - 2.0 * (i_grid as f64)).powf(2.0)).exp()
            })
            .sum()
    };

    let mut draw_distribution = |distro: Vec<f32>, color: RGBColor| {
        context
            .draw_series(
                AreaSeries::new(
                    std::iter::zip(
                        prices.clone().into_iter(),
                        prices
                            .clone()
                            .into_iter()
                            .map(|val| distribution(val, &distro)),
                    ),
                    0.0,
                    &color.mix(0.2),
                )
                .border_style(&color),
            )
            .unwrap();
    };

    draw_distribution(block_distribution, BLUE);
    draw_distribution(next_distribution, GREEN);

    Ok(())
}
