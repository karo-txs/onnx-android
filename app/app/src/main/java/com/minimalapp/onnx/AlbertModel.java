package com.minimalapp.onnx;

class AlbertModel {

    static {
        System.loadLibrary("onnx_android");
    }

    public static native String inference(String prompt, String modelPath, String tokenizer);

}
