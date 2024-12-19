mod layer;
mod cell;
mod pricing;

use std::fs::File;
use std::io::{BufReader, Read};
use protobuf::Message;
use queue::Queue;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use layer::LSTMLayer;
use pricing::PriceEmbedding;

include!("mod.rs");
use records::{LogData, BondingCurveData};

fn embed_chunk(chunks: Vec<i32>, embedding_matrix: &Array2<f32>) -> Array2<f32> {
    let mut embed: Array2<f32> = Array2::zeros((embedding_matrix.shape()[1], 0));
    chunks.iter().for_each(|&val| {
        let slice = embedding_matrix.row(val as usize);
        let _ = embed.push(Axis(1), slice);
    });
    
    embed
}

fn chunk_tokens(tokens: Vec<i32>) -> Vec<Vec<i32>> {
    let chunk_size = 250;
    tokens.chunks(chunk_size).map(|chunk| {
        if chunk.len() < chunk_size {
            let mut short = chunk.to_vec();
            let padding = chunk_size.saturating_sub(chunk.len());
            short.extend(std::iter::repeat(0).take(padding));
            return short;
        }
        return chunk.to_vec();
    }).collect()
}

#[derive(Clone, Debug)]
struct LogQueue {
    embeded_tokens: Array2<f32>,
    slot: u64
}

fn main() {
    let vocab_dim = 128;
    let hidden_dim = 40; 
    let log_embedding = Array2::random((vocab_dim, hidden_dim), Uniform::new(0., 1.));
    let data_file = File::open("../data.bin").expect("Failed to open file");
    let mut reader = BufReader::new(data_file);
    let mut log_q: Queue<LogQueue> = Queue::new();
    let mut data: Vec<Array2<f32>> = Vec::new();

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
                        for chunk in chunk_tokens(logs.tokenizedLogs) {
                            let token_e = embed_chunk(chunk, &log_embedding);
                        
                            let log_queue = LogQueue {
                                slot: logs.slot,
                                embeded_tokens: token_e 
                            };
                            log_q.queue(log_queue).unwrap();
                        }
                    },
                    1 => {
                        let bonding = BondingCurveData::parse_from_bytes(&msg_buff).unwrap();  
                        let mint_e = embed_chunk(bonding.mintId, &log_embedding);

                        while let Some(logs) = log_q.dequeue() {
                            if logs.slot > bonding.slot { break; }
                            
                            let mut combined_e = logs.embeded_tokens;     
                            for row in mint_e.rows() {
                                let _ = combined_e.push(Axis(0), row); 
                            }
                            let price_embedding = PriceEmbedding::new(1, hidden_dim);
                            let price_weights = price_embedding.forward(bonding.priceInSol as f32); 
                            let _ = combined_e.push(Axis(0), price_weights.view());

                            let _ = data.push(combined_e);
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
    let input_dim = 250;
    let mut lstm_layer = LSTMLayer::new(input_dim, hidden_dim);
    let _ = lstm_layer.load_weights("weights.bin");

    for epoch in 0..epochs {
        for i in batch_size..data.len() {
            let sequence_slice = &data[i-batch_size..i-1];
            let total_loss = lstm_layer.train(&sequence_slice.to_vec());

            println!("Batch: {},\tLoss: {}", i, total_loss);
        }
        println!("Epoch {}", epoch);
    }

}


