# nnapi-sample-convolution
This is a simple example performs one convolution layer model with [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)

```
# [Host]

mkdir build

cd build

cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 ..

adb push nnapi-sample-convolution /data/local/tmp

adb shell

# [Device]

cd /data/local/tmp

./nnapi-sample-convolutions
```

Reference:

https://github.com/JDAI-CV/DNNLibrary

