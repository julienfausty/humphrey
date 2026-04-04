use csv;
use std::io::{Read, stdin};

#[derive(Debug, serde::Deserialize)]
struct OHLC {
    pub Timestamp: f32,
    pub Low: f32,
    pub Open: f32,
    pub Close: f32,
    pub High: f32,
    pub Volume: f32,
}

pub fn parse<T: Read>(source: T) -> Result<Vec<OHLC>, String> {
    let mut ohlc_data = Vec::new();
    let mut reader = csv::Reader::from_reader(source);
    for parsed in reader.deserialize() {
        match parsed {
            Ok(ohlc) => {
                ohlc_data.push(ohlc);
            }
            Err(err) => return Err(format!("Error parsing csv data:\n{:?}", err)),
        }
    }
    Ok(ohlc_data)
}

fn main() -> Result<(), String> {
    let ohlc_data = parse(stdin());


    Ok(())
}
