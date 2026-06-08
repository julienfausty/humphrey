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
const PRICE_RANGE: (f64, f64) = (0.95, 1.05);
const PLOT_GRID_SIZE: usize = 512;
const N_KERNELS: usize = 32;
const BLOCK_SIZE: usize = 128;

fn distribution(x: f64, weights: &Vec<f32>) -> f64 {
    let price_width = PRICE_RANGE.1 - PRICE_RANGE.0;
    let grid_m1 = weights.len() as f64 - 1.0;
    (0..weights.len())
        .map(|i_kernel| {
            (weights[i_kernel] as f64)
                * (2.0 * grid_m1 / (price_width * (2.0 * PI).sqrt()))
                * ((-1.0 / 2.0)
                    * ((x - PRICE_RANGE.0) * 2.0 * grid_m1 / price_width - 2.0 * (i_kernel as f64))
                        .powf(2.0))
                .exp()
        })
        .sum()
}

fn latest_distributions(
    dataset: &MapperDataset<OHLCDataset<NdArray>, NormalizeOHLCItem, OHLCItem<NdArray>>,
) -> Result<(), String> {
    println!("Lates Distributions:");
    let dset_index = dataset.len() - 1;

    let price_width = PRICE_RANGE.1 - PRICE_RANGE.0;

    println!("Preparing data...");
    let item = dataset.get(dset_index).unwrap();
    let one_batch = OHLCBatcher {}.batch(vec![item.clone()], &item.block.device());

    let prices = (0..PLOT_GRID_SIZE)
        .map(|i_grid| price_width * (i_grid as f64) / ((PLOT_GRID_SIZE - 1) as f64) + PRICE_RANGE.0)
        .collect::<Vec<_>>();

    let projector = OHLC2Distribution;
    let block_weights: Vec<f32> = projector
        .project(
            one_batch.blocks.clone().slice_assign(
                s![0.., 0.., 0],
                1.0 - one_batch.blocks.clone().slice(s![0.., 0.., 0]),
            ),
            N_KERNELS,
            PRICE_RANGE,
        )
        .reshape([N_KERNELS])
        .to_data()
        .to_vec()
        .unwrap();

    let next_weights: Vec<f32> = projector
        .project(
            one_batch.nexts.clone().slice_assign(
                s![0.., 0.., 0],
                one_batch.nexts.clone().slice(s![0.., 0.., 0]) - 1.0,
            ),
            N_KERNELS,
            PRICE_RANGE,
        )
        .reshape([N_KERNELS])
        .to_data()
        .to_vec()
        .unwrap();

    let block_distribution: Vec<f64> = prices
        .clone()
        .into_iter()
        .map(|val| distribution(val, &block_weights))
        .collect();

    let next_distribution: Vec<f64> = prices
        .clone()
        .into_iter()
        .map(|val| distribution(val, &next_weights))
        .collect();

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
        .build_cartesian_2d(PRICE_RANGE.0..PRICE_RANGE.1, 0.0..(1.1 * max_probability))
        .unwrap();

    context.configure_mesh().draw().unwrap();

    let mut draw_distribution = |distro: Vec<f64>, color: RGBColor, label| {
        context
            .draw_series(
                AreaSeries::new(
                    std::iter::zip(prices.clone().into_iter(), distro),
                    0.0,
                    &color.mix(0.2),
                )
                .border_style(&color),
            )
            .unwrap()
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    };

    draw_distribution(block_distribution, BLUE, "Present");
    draw_distribution(next_distribution, GREEN, "Future");

    context
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    Ok(())
}

fn latest_n_differences(
    dataset: &MapperDataset<OHLCDataset<NdArray>, NormalizeOHLCItem, OHLCItem<NdArray>>,
) -> Result<(), String> {
    println!("N Differences:");
    let n_distributions = 20;

    let price_width = PRICE_RANGE.1 - PRICE_RANGE.0;

    println!("Preparing data...");
    let items =
        (0..n_distributions).map(|i| dataset.get(dataset.len() - 1 - i * BLOCK_SIZE).unwrap());
    let one_batches =
        items.map(|item| OHLCBatcher {}.batch(vec![item.clone()], &item.block.device()));

    let prices = (0..PLOT_GRID_SIZE)
        .map(|i_grid| price_width * (i_grid as f64) / ((PLOT_GRID_SIZE - 1) as f64) + PRICE_RANGE.0)
        .collect::<Vec<_>>();

    let projector = OHLC2Distribution;
    let block_weights: Vec<Vec<f32>> = one_batches
        .clone()
        .map(|one_batch| {
            projector
                .project(
                    one_batch.blocks.clone().slice_assign(
                        s![0.., 0.., 0],
                        1.0 - one_batch.blocks.clone().slice(s![0.., 0.., 0]),
                    ),
                    N_KERNELS,
                    PRICE_RANGE,
                )
                .reshape([N_KERNELS])
                .to_data()
                .to_vec()
                .unwrap()
        })
        .collect();

    let next_weights: Vec<Vec<f32>> = one_batches
        .map(|one_batch| {
            projector
                .project(
                    one_batch.nexts.clone().slice_assign(
                        s![0.., 0.., 0],
                        one_batch.nexts.clone().slice(s![0.., 0.., 0]) - 1.0,
                    ),
                    N_KERNELS,
                    PRICE_RANGE,
                )
                .reshape([N_KERNELS])
                .to_data()
                .to_vec()
                .unwrap()
        })
        .collect();

    let diff_weights: Vec<Vec<f32>> =
        std::iter::zip(block_weights.into_iter(), next_weights.into_iter())
            .map(|(block, next)| {
                std::iter::zip(block.into_iter(), next.into_iter())
                    .map(|(w_block, w_next)| w_next - w_block)
                    .collect()
            })
            .collect();

    let diff_distributions: Vec<Vec<f64>> = diff_weights
        .into_iter()
        .map(|weights| {
            prices
                .clone()
                .into_iter()
                .map(|val| distribution(val, &weights))
                .collect()
        })
        .collect();

    println!("Plotting...");

    let image_location = format!("{ASSET_DIR}/images/latest_N_differences.png");
    let root_area = BitMapBackend::new(&image_location, (1200, 800)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let max_probability = diff_distributions.iter().fold(0.0, |max, distro| {
        distro
            .iter()
            .fold(0.0, |inner_max, &val| val.max(inner_max))
            .max(max)
    }) as f64;

    let min_probability = diff_distributions
        .iter()
        .fold(f64::INFINITY, |min, distro| {
            distro
                .iter()
                .fold(f64::INFINITY, |inner_min, &val| val.min(inner_min))
                .min(min)
        }) as f64;

    let mut context = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(
            format!("Latest {n_distributions} Differences"),
            ("monospace", 40),
        )
        .build_cartesian_2d(
            PRICE_RANGE.0..PRICE_RANGE.1,
            (min_probability - 0.1 * min_probability.abs())..(1.1 * max_probability),
        )
        .unwrap();

    context.configure_mesh().draw().unwrap();

    let mut draw_distribution = |distro: Vec<f64>, color: RGBColor| {
        context
            .draw_series(
                AreaSeries::new(
                    std::iter::zip(prices.clone().into_iter(), distro),
                    0.0,
                    &color.mix(0.2),
                )
                .border_style(&color),
            )
            .unwrap();
    };

    for i_distro in 0..n_distributions {
        let portion = 1.0 - ((i_distro as f64) / (n_distributions as f64));
        let green = (255.0 * portion) as u8;
        draw_distribution(diff_distributions[i_distro].clone(), RGBColor(0, green, 0));
    }

    Ok(())
}

fn main() -> Result<(), String> {
    let device = NdArrayDevice::default();

    println!("Reading dataset...");
    let base_dataset = OHLCDataset::<NdArray>::new::<Stdin>(BLOCK_SIZE, stdin(), &device).unwrap();
    let dataset: MapperDataset<_, _, OHLCItem<NdArray>> =
        MapperDataset::new(base_dataset.clone(), NormalizeOHLCItem);

    latest_distributions(&dataset).expect("Failed to plot latest distributions");
    latest_n_differences(&dataset).expect("Failed to plot latest 100 differences");

    Ok(())
}
