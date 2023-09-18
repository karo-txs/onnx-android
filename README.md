# onnx-android
Creation of a minimal app using an ONNX model with Rust. 

Dependencies:
- [tract](https://github.com/sonos/tract/tree/main): Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference.
- [cargo-ndk](https://github.com/bbqsrc/cargo-ndk): Compile Rust projects against the Android NDK without hassle.

## 1. Export model
```
pip install -q torch transformers onnx sentencepiece
python3 scripts/export.py
wget -P assets/albert/ https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json
```

## 2. Setup ndk
```
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install openjdk-8-jdk -y
sudo apt-get install clang llvm lld sdkmanager -y

sudo sdkmanager --list
sudo sdkmanager "build-tools;34.0.0"
sudo sdkmanager "platform-tools;34.0.0"
sudo sdkmanager "ndk;25.1.8937393"
sudo sdkmanager "ndk-bundle;25.0.8775105"
sudo sdkmanager "platforms;android-34"

rustup target add aarch64-linux-android

cargo install cargo-ndk

export ANDROID_NDK_HOME=/opt/android-sdk/ndk/25.1.8937393/
export ANDROID_SDK_HOME=/opt/android-sdk/
export ANDROID_NDK_API_LEVEL=34
export ANDROID_API_LEVEL=34
export ANDROID_BUILD_TOOLS_VERSION=34.0.0

sudo chown -R $(whoami) /home/<user>/.cargo/
sudo chown -R $(whoami) /opt/android-sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin/
```

## 3. Export lib
```
cd onnx-android/
cargo ndk -t arm64-v8a -o ../app/app/src/main/jniLibs/ build --release

cd ../
cp /opt/android-sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so app/app/src/main/jniLibs/arm64-v8a/

cp assets/albert/model.onnx  app/app/src/main/assets/
cp assets/albert/tokenizer.json  app/app/src/main/assets/
```