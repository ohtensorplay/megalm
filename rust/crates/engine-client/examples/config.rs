use megalm_engine_client::EngineConfig;

fn main() {
    let config = EngineConfig {
        model_path: "/tmp/model".into(),
        served_model_name: "served".into(),
        ..EngineConfig::default()
    };
    println!("{}", config.encode().unwrap());
}
