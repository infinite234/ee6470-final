# ee6470-FINAL TERM PROJECT:
# Design and Implementation of LSTM Digital pre-distortion model inference on RISC V

# BY - Shaswat Satapathy (309591029) AND Shivani Singh (309591030 ) 

## Abstract

With the ever growing number of devices, the signals are getting more complex with dynamic
behaviour, which causes non-linearity in the Power Amplifier (PA) output.
▪ Digital Pre-distortion (DPD) is considered an effective technique to reduce these effects and to
make this system more real-time, scholars are now diving into deep learning methodologies like
CNN and LSTM.
▪ To run these algorithms in devices with less capability of resource storage and computation, is a
major challenge since these models are computational and memory escalated.
▪ We need a scalable and flexible implementation to meet requirements from IoT to high-end
applications, supporting both inference and on-device learning for edge devices.
▪ In this work, we propose a step-by-step guide to build LSTM deep learning based DPD model
inference accelerators using RISC V ISA

## What is Digital Perdistortion (DPD)?
A technique for improving the linearity of power amplifiers. Ideally the output signal of a PA is the input scaled up perfectly Instead the semiconductor physics causes distortions in Amplitude, frequency and phase errors
■ If we can predict the errors, we can try to reverse them

## Introduction to LSTM?
Long Short Term Memory networks - usually just called “LSTMs” - are a
special kind of Recurrent Neural Network (RNN), capable of learning
long-term dependencies.Unlike regular RNN networks, LSTMs also have this chain like structure
but the repeating module has a different structure.

### LSTM Gates
The first step in our LSTM is to decide what information we’re going to throw away from the cell state<br/>
![source image](https://github.com/infinite234/ee6470-final/blob/main/new.PNG)<br/>

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state.<br/>
![source image](https://github.com/infinite234/ee6470-final/blob/main/new1.PNG)<br/>

 update the old cell state, Ct−1, into the new cell state Ct.<br/>
 ![source image](https://github.com/infinite234/ee6470-final/blob/main/new2.PNG)<br/>
 
 The output will be based on our cell state, but will be a filtered version.<br/>
 ![source image](https://github.com/infinite234/ee6470-final/blob/main/new3.PNG)<br/>
 
 
 

