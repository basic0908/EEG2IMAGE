# EEG-based classification of imagined digits using low-density EEG
Digit classification using low-density EEG

![framework](https://github.com/user-attachments/assets/67ebcabd-4574-4d10-aeb4-e97151581142)


Aim : to classify the imagined speech of numerical digits from EEG signals by exploiting the past and future temporal characteristics of the signal using several deep learning models  
EEG signal processing : EEG signals were filtered and preprocessed using the discrete wavelet transform to remove artifacts and retrieve feature information  
Feature classification : multiple version sof multilayer bidirectional recurrent neural networks were used

### EEG data
EEG signals from each trial were recorded for 2 seconds. 
- EPOC 14 channels 14 channels 128Hz
- MUSE 4 channels 220 Hz

### Signal Processing
1. Butterworth high-pass filter fo order 5 at 0.1 Hz to erase the low-frequencies noise
1. Notch filter to remove 60 Hz electrical environment noise.
1. Discrete wavelet transform(DWT) using the Daubechies-4 wavelet with two-level decomposition on EPOC and the three-level decompostion on MUSE for denoising and informaiton extraction. 
1. Inverse reconstruction of the original EEG waveform using DWT components
1. standardized using the Z-score normalization

### Deep Learning Model Architecture
![model architecture](https://github.com/user-attachments/assets/11afa959-51aa-44b4-be10-1d1c882d4729)
- multilayer bidrectional-RNN
- multilayer bidirectional-GRU
- multilayer bidirectional-LSTM


### Results 
![results](https://github.com/user-attachments/assets/4e9681a4-3567-4a79-81b3-cc0d7d1e24e6)


### Data preprocess
Raw -> preprocess(filters) -> reconstruction(wavelet transformation) -> standardization
![raw](plot/3_plot.png)