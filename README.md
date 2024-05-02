# Simple Speech Recognition (simple-sr)
Based on the simple audio TensorFlow tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio
The data set from http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz provides more
samples than the one referenced in the tutorial, and results in more accurate predictions.

This application takes a wav file as input and outputs prediction values for each of 8 trained words:
- 'down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'

The application is the inference portion of the TensorFlow model in C++11.
The motivation for this project is to be able to run a simple voice recognition model on older platforms that
do not support C++ versions greater than C++11.  The implementation is simplified and not meant as a general
purpose model framework.

The model weights are taken from the trained TensorFlow model and generated into C arrays.

## Layers
### 1. Input and preprocessing
1. Read in a wav file.  The audio sample must be 1 second in length, 16 KHz, single channel, and 16 bit.
2. Normalize the signal to scale betweem -1.0 and 1.0.
3. Convert the normalized signal into a spectrogram image.
4. Resize the spectrogram to 64x64 using bilinear interpolation.
5. Normalize the spectrogram using the mean and variance from the TensorFlow model.

### 2. Run the input through the model
1. The first Conv2D 32 filter layer takes the 64x64x1 image and produces a 62x62x32 output matrix, using a 3x3 kernel with 32 output channels.
   The image is reduced to 62x62.  The input matrix is not zero padded while applying the 3x3 kernel, so no calculations are made for the border values.
   ReLU is applied as the activation function.
2. The MaxPooling2D layer slides a 2x2 window across each channel's matrix and selects the max value, thus reducing the output to 64 channels of 31x31 matrices.
3. The Conv2D 64 filter layer takes the 31x31x32 input and applies a 3x3 kernel to each of the 32 channels and produces 64 output channels of 29x29 matrices.
   ReLU is applied as the activation function.
4. The MaxPooling2D layer reduces the data to 14x14x64.
5. The Conv2D 128 filter layer takes the 14x14x64 input and applies a 3x3 kernel to each of the 64 channels and produces 128 output channels of 12x12 matrices.
   ReLU is applied as the activation function.
6. The MaxPooling2D layer reduces the data to 6x6x128.
4. There is a dropout layer in the TensorFlow model, but it is only applied for training, so it is not implemented here.
5. The resulting matrix is then 'flattened' as input to the next layer.
6. The dense layer is a fully connected network layer of the linear equation: Y = W * X + b.
   It takes a 4608 sized vector and applies it to 128 output units to get a 128 sized vector.
   This is a hidden unit layer with a ReLU activation function.
8. There is another dropout layer, which is ignored.
9. The final layer is the output dense layer, which has 8 outputs for the labels: 'down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'.

## References
Conv2D
- fast.ai has a really good explanation of the Conv2D and MaxPooling2D layers: https://course.fast.ai/Lessons/lesson8.html

## Model and layer examinations and implementations in python using jupyter-lab
Training:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/simple_sr_train_v2.ipynb

Overview of the layers and their output dimensions:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/simple_sr_layers_v2.ipynb

Exploring the saved model parameters in HDF5 format:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/explore_hdf5.ipynb

Resizing and normalization implementations:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/resizing_normalization.ipynb

Conv2D implementation:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/conv2d.ipynb

MaxPooling2D and Dense layers implementations:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/post_conv2d.ipynb

Converting the TensorFlow model weights into C arrays:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/weights.ipynb

## Dependencies
libsoundfile
- Reads in the wav file.
- http://www.mega-nerd.com/libsndfile/
- https://github.com/libsndfile/libsndfile

fftw
- Performs the RFFT transform for the spectrogram.
- https://www.fftw.org
- https://github.com/FFTW/fftw3

## Building
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

To also build the tests:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DTESTS=ON
```

## Running
```
$ ../build/simple-sr test-16.wav

[Prediction]
go          no          down        stop        up          left        yes         right
1.286628    1.226844    0.704659    -0.365673   -1.032117   -1.678373   -2.012884   -2.128404

==========>   go   <==========
```

The tools directory has a script that will record a 1 second audio sample, down sample it to 16KHz, and run the model on it:
```
$ ./record
sox WARN formats: can't set sample rate 16000; using 44100
sox WARN formats: can't set 1 channels; using 2

Input File     : 'default' (coreaudio)
Channels       : 2
Sample Rate    : 44100
Precision      : 32-bit
Sample Encoding: 32-bit Signed Integer PCM
...
Output #0, wav, to 'test-16.wav':
  Metadata:
    ISFT            : Lavf61.1.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
      Metadata:
        encoder         : Lavc61.3.100 pcm_s16le
...
[Prediction]
right       go          up          left        down        stop        no          yes
2.545856    1.049048    0.566780    -0.340170   -0.791220   -1.468079   -1.997716   -2.915035

==========>   right   <==========
```
