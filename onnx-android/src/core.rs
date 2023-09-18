use ndarray::s;
use std::{
    path::PathBuf,
    str::FromStr,
};
use tokenizers::tokenizer::Tokenizer;
use tract_onnx::prelude::*;


pub fn inference(text: String, model_path: String, tokenizer_path: String) -> String {
    let model_dir = PathBuf::from_str(&model_path).expect("Model Path not found");

    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Tokenizer not found");

    let tokenizer_output = tokenizer.encode(text, true).expect("Tokenizer encode invalid");
    let input_ids = tokenizer_output.get_ids();
    let attention_mask = tokenizer_output.get_attention_mask();
    let token_type_ids = tokenizer_output.get_type_ids();

    let length = input_ids.len();
    let mask_pos =
        input_ids.iter().position(|&x| x == tokenizer.token_to_id("[MASK]").unwrap()).unwrap();

    let model = tract_onnx::onnx()
        .model_for_path(&model_dir).expect("Invalid load model")
        .into_optimized().expect("Invalid optimization")
        .into_runnable().expect("Model");

    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    ).expect("Input ids").into();
    let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    ).expect("Attention mask").into();
    let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    ).expect("Token type ids").into();

    let outputs =
        model.run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into())).expect("Outputs");
    let logits = outputs[0].to_array_view::<f32>().expect("Logits");
    let logits = logits.slice(s![0, mask_pos, ..]);
    let word_id = logits.iter().zip(0..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap()).unwrap().1;
    let word = tokenizer.id_to_token(word_id).expect("Tokens").into();

    return word;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference() {
        let result = inference(String::from("Paris is the [MASK] of France."), 
                    String::from("../assets/albert/model.onnx"), 
                    String::from("../assets/albert/tokenizer.json"));
        
        assert_eq!(result, "▁capital");
    }
}