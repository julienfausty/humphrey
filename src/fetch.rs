use kraken_async_rs::clients::core_kraken_client::CoreKrakenClient;
use kraken_async_rs::clients::http_response_types::ResultErrorResponse;
use kraken_async_rs::clients::kraken_client::KrakenClient;
use kraken_async_rs::crypto::nonce_provider::{IncreasingNonceProvider, NonceProvider};
use kraken_async_rs::request_types::{CandlestickInterval, OHLCRequest};
use kraken_async_rs::secrets::secrets_provider::{SecretsProvider, StaticSecretsProvider};

use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), String> {
    let providers: (
        Box<Arc<Mutex<dyn SecretsProvider>>>,
        Box<Arc<Mutex<dyn NonceProvider>>>,
    ) = (
        Box::new(Arc::new(Mutex::new(StaticSecretsProvider::new("", "")))),
        Box::new(Arc::new(Mutex::new(IncreasingNonceProvider::new()))),
    );

    let mut client = CoreKrakenClient::new(providers.0, providers.1);

    let asset = "XETHZEUR".to_string();

    let request = OHLCRequest::builder(asset.clone())
        .interval(CandlestickInterval::Minute)
        .since(0)
        .build();

    let response = client.get_ohlc(&request).await;
    let ohlc_data = match response {
        Ok(ResultErrorResponse {
            result: ohlc_response,
            error: errs,
        }) => {
            if !errs.is_empty() {
                println!("{:?}", errs);
            }
            match ohlc_response {
                Some(ohlc_data) => ohlc_data,
                None => {
                    return Err("No OHLC data found in response.".to_string());
                }
            }
        }
        Err(network_fail) => {
            return Err(format!("{:?}", network_fail));
        }
    };

    let asset_data = match ohlc_data.ohlc.get(&asset) {
        Some(data) => data,
        None => return Err(format!("Could not find {} asset in OHLC response.", asset)),
    };

    println!("{:?}", asset_data);

    Ok(())
}
