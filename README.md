# Simple Speech Recognition (simple-sr)
Based on the simple audio TensorFlow tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio

This application takes a wav file as input and outputs prediction values for each of 8 trained words:
- 'down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'

The application is the feed forward portion of the TensorFlow model in C++11.
The motivation for this project is to be able to run a simple voice recognition model on older platforms that
do not support versions greater than C++11.  The implementation is simplified and not meant as a general purpose model framework.

The model weights are taken from the trained TensorFlow model and generated into C arrays.

## Layers
### 1. Input and preprocessing
1. Read in a wav file.  The audio sample must be 1 second in length, 16 KHz, single channel, and 16 bit.
2. Normalize the signal to scale betweem -1.0 and 1.0.
3. Convert the normalized signal into a spectrogram image.
4. Resize the spectrogram to 32x32 using bilinear interpolation.
5. Normalize the spectrogram using the mean and variance from the TensorFlow model.

### 2. Run the input through the model
1. The first Conv2D layer takes the 32x32x1 image and produces a 30x30x32 output matrix, using a 3x3 kernel with 32 output channels.
   The image is reduced to 30x30.  The input matrix is not zero padded while applying the 3x3 kernel, so no calculations are made for the border values.
   ReLU is applied as the activation function.
2. The second Conv2D layer takes the 30x30x32 input and applies a 3x3 kernel to each of the 32 channels and produces 64 output channels of 28x28 matrices.
   ReLU is applied as the activation function.
3. The MaxPooling2D layer slides a 2x2 window across each channel's matrix and selects the max value, thus reducing the output to 64 channels of 14x14 matrices.
4. There is a dropout layer in the TensorFlow model, but it is only applied for training, so it is not implemented here.
5. The resulting matrix is then 'flattened' as input to the next layer.
6. The dense layer is a fully connected network layer of the linear form: Y = W * X + b.
   It takes a 12544 sized vector and applies it to 128 output units to get a 128 sized vector.
   This is a hidden unit layer with a ReLU activation function.
8. There is another dropout layer, which is ignored.
9. The final layer is the output dense layer, which has 8 outputs for the labels: 'down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'.

### 3. Post processing
'go' may often have a better prediction value for 'no' and 'down' inputs, but the inverse is not the same, so 'no' or 'down' may be selected instead of 'go'
if either is close enough to 'go'.

## References
Conv2D
- fast.ai has a really good explanation of the Conv2D and MaxPooling2D layers: https://course.fast.ai/Lessons/lesson8.html

## Model and layer examinations and implementations in python using jupyter-lab
Training:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/simple_sr_train.ipynb

Overview of the layers and their output dimensions:
- https://github.com/dwrank/ml-notebooks/blob/master/audio/simple-sr/simple_sr_layers.ipynb

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
With no arguments, the model will run on ../data/yes.wav

```
$ ./simple-sr

[Prediction]
down        go          left        no          right       stop        up          yes
-3.174652   -3.007938   1.979785    -1.735928   -4.648821   -1.321403   -4.770293   6.679570

==========>   yes   <==========
```

To input a wav file:
```
$ ../build/simple-sr test-16.wav

[Prediction]
down        go          left        no          right       stop        up          yes
6.893702    7.134671    -8.687324   3.409425    -7.497922   -1.306404   -4.478090   -5.634733

==========>   down   <==========
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
down        go          left        no          right       stop        up          yes
-5.282815   -1.309025   2.154530    -6.874960   10.475742   -5.820977   1.990969    -4.639996

==========>   right   <==========
```
