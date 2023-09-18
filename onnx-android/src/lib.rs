pub mod core;

use crate::core::inference;
use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_com_minimalapp_onnx_AlbertModel_inference(
    env: JNIEnv,
    _class: JClass,
    input: JString,
    model_path: JString,
    tokenizer_path: JString,
) -> jstring {

    let input: String = env
        .get_string(input)
        .expect("Couldn't get java string!")
        .into();

    let model_path: String = env
        .get_string(model_path)
        .expect("Couldn't get java string!")
        .into();

    let tokenizer_path: String = env
        .get_string(tokenizer_path)
        .expect("Couldn't get java string!")
        .into();

    let output = env
        .new_string(inference(input, model_path, tokenizer_path))
        .expect("Coudn't create java string into android!");

    return output.into_raw();

}
