mod layer;
mod cell;
mod embeddings;

use std::fs::File;
use std::io::{BufReader, Read};
use protobuf::Message;
use queue::Queue;
use layer::LSTMLayer;

include!("mod.rs");
use records::{LogData, BondingCurveData};

fn format(str: String, with_padding: bool) -> Vec<f32> {
    let chunk_size: usize = 13;
    let mut bytes: Vec<f32> = str.as_bytes().iter().map(|x| (x - 24) as f32).collect();
    if with_padding {
        let padding = chunk_size.saturating_sub(bytes.len());
        bytes.extend(std::iter::repeat(0.0).take(padding));
        bytes.into_iter().collect()
    } else {
       bytes
    }
    
}

fn aggregate(data: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut agg = Vec::new();
    let  (mut slot, mut price, mut id, mut logs) = data.clone();
    agg.append(&mut slot);
    agg.append(&mut price);
    agg.append(&mut id);
    agg.append(&mut logs);
    agg.to_vec()
}

fn chunk_tokens(tokens: Vec<i32>) -> Vec<Vec<f32>> {
    let chunk_size = 250;
    tokens.chunks(chunk_size).map(|chunk| {
        if chunk.len() < chunk_size {
            let mut short = chunk.to_vec();
            let padding = chunk_size.saturating_sub(chunk.len());
            short.extend(std::iter::repeat(0).take(padding));
            short
        } else {
            chunk.to_vec()
        }
    })
    .map(|chunk| chunk.iter().map(|x| *x as f32).collect())
    .collect()
}

#[derive(Clone, Debug)]
struct ChunkedLog {
    tokens: Vec<f32>,
    slot: u64
}

fn main() {
    let data_file = File::open("../data.bin").expect("Failed to open file");
    let mut reader = BufReader::new(data_file);
    let mut log_q: Queue<ChunkedLog> = Queue::new();
    let mut data = Vec::new();

    loop {
        let mut size_buffer = [0u8; 4];
        let mut type_buffer = [0u8; 1];
        match reader.read_exact(&mut size_buffer) {
            Ok(_) => {
                reader.read_exact(&mut type_buffer).expect("Failed to read type buffer");

                let size = u32::from_be_bytes(size_buffer) as usize;
                let mut msg_buff = vec![0u8; size];
    
                reader.read_exact(&mut msg_buff).expect("Failed to read msg buffer");
                
                match type_buffer[0] {
                    0 => {
                        let logs = LogData::parse_from_bytes(&msg_buff).unwrap();
                        for tokens in chunk_tokens(logs.tokenizedLogs) {
                            let log_queue = ChunkedLog {
                                slot: logs.slot,
                                tokens
                            };
                            log_q.queue(log_queue).unwrap();
                        }
                    },
                    1 => {
                        let bonding = BondingCurveData::parse_from_bytes(&msg_buff).unwrap();  
                        let mint_tokens: Vec<f32> = bonding.mintId.iter().map(|x| *x as f32).collect();
                        let slot_tokens = format(bonding.slot.to_string(), false);
                        let price_tokens = format(bonding.priceInSol.to_string(), true);
                        while let Some(logs) = log_q.dequeue() {
                            if logs.slot > bonding.slot { break; }   
                            let seq = aggregate(&(
                                slot_tokens.clone(), 
                                price_tokens.clone(), 
                                mint_tokens.clone(),  
                                logs.tokens
                            ));
                            data.push(seq);
                        }
            
                        if data.len() > 1000 { break; }
                    }
                    _ => println!("Unknown msg type: {}", type_buffer[0]),
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                println!("End of file reached");
                break;  
            },
            Err(e) => panic!("Error reading message size: {}", e),
        }
    }
 
    let epochs = 100;
    let batch_size = 32;   
    let mut lstm_layer = LSTMLayer::new();
    let _ = lstm_layer.load_weights("weights.bin");

    for epoch in 0..epochs {
        for i in batch_size..data.len() {
            lstm_layer.train(&data[i-batch_size..i-1]);
        }
        println!("Epoch {}", epoch);
    }

}


