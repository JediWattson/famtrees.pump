use std::path::PathBuf;

fn main() {
    let proto_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..");

    protobuf_codegen::Codegen::new()
        .out_dir("src")
        .include(proto_path.to_str().unwrap())
        .input(proto_path.join("records.proto").to_str().unwrap())
        .run_from_script();
}
