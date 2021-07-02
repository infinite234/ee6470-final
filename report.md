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
 
 
## II. Implementation details
### 1. Implementation steps 

### 2. Software Implementation

To model the power amplifer we use python with tesnsorflow running in backend. We take some real power amplifier data and model the PA accordingly. Some of the snaps of training are mention here: 

```python

from keras.layers import LSTM, Dropout, Activation, TimeDistributed, Bidirectional, concatenate, Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, TimeDistributed, Bidirectional, concatenate

# In[] Load data

data = sio.loadmat('./Data/indatapa_1G.mat')
indata_1G = np.reshape(data['indatapa_1G'],(-1,1))[:8000]
data = sio.loadmat('./Data/outdatapa_1G.mat')
outdata_1G = np.reshape(data['outdatapa_1G']/11.5,(-1,1))[:8000]

# In[]  PA model


i_r = Input(batch_shape=(batch_size, timesteps, input_dim), name='main_input')
i_i = Input(batch_shape=(batch_size, timesteps, input_dim), name='aux_input')

x = concatenate([i_r, i_i])

o = LSTM(400, return_sequences=True, stateful=True)(x)

o_r = Dense(input_dim, name='main_output')(o)
o_i = Dense(input_dim, name='aux_output')(o)


# In[] Load Weight
m_r.load_weights('./PA_Modeling/weight/LSTM_inv.h5py')

predict_r, predict_i = m_r.predict({'main_input': train_x_r, 'aux_input': train_x_i})
trainPredict_r = np.reshape(scaler_inr.inverse_transform(predict_r_),(-1,))
trainPredict_i = np.reshape(scaler_ini.inverse_transform(predict_i_),(-1,))

```

### Stratus HLS Implementation 

In this part we implemented the DPD model in Stratus HLS for hardware level sysnthesis using SystemC and TLM. We first design the lstm.cpp module and then communicate it with testbench.cpp file. The lstm module is designed as follows: 

```c++
void lstm::do_filter(){
    sc_int<32> res;
    sc_dt::sc_uint<24> buffer_r;
    sc_dt::sc_uint<24> buffer_i;
    { wait(CLOCK_PERIOD, SC_NS); }
    
    
    while (true) {
        input1 =0; input2 =0; int j = 0;
        wait(CLOCK_PERIOD, SC_NS);
	
       int first = 0;
       for (unsigned int v = 0; v < hidden_layer; ++v) {
          
          wait(CLOCK_PERIOD, SC_NS);
          buffer_r = i_1.read();
          buffer_i = i_2.read() ;
          input1 = buffer_r*precision;
          input2 = buffer_i*precision;
          
         //input state	
      	 i_state = bias_lstm[v] + ker[v]*input1 + ker[N1+v]*input2;
      	
        //forget state
      	f_state = bias_lstm[N1+v] + ker[2*N1+v]*input1 + ker[3*N1+v]*input2;
      	
      	//output state
      	o_state = bias_lstm[2*N1+v] + ker[4*N1+v]*input1 + ker[5*N1+v]*input2;
        
        //cell compute state
      	g_state = bias_lstm[3*N1+v] + ker[6*N1+v]*input1 + ker[7*N1+v]*input2;
	
        //cell state
      	if(v-1<0){
      		c_state = (f_state * 0) + (g_state * i_state);
     		}
     		else{
      		c_state = (f_state * c_state) + (g_state * i_state);
     		}
        
      	//hidden state
      	h_state = o_state * tanh_func(c_state); 
        }
        int result1 = (int)((output_main + main_bias)*precision);
        int result2 = (int)((output_aux+ aux_bias)*precision);
    
        o_result1.put(result1);
        o_result2.put(result2);
}     
```
### 4. RISC-V implementation 

For hardware/software co-realtion we port the DPD model created in SystemC with a RISC V virtual platform using "basic-acc". The sodtware and hardware code written in the VP communicate through DMA. The sample code is mention here : 

```c++
static char* const lstm_START_ADDR = reinterpret_cast<char* const>(0x77000000);
static char* const lstm_START1_ADDR = reinterpret_cast<char* const>(0x77000036);
static char* const lstm_READ_ADDR  = reinterpret_cast<char* const>(0x77000070);
static char* const lstm_READ1_ADDR = reinterpret_cast<char* const>(0x7700000120);

 for(int i = 0; i < 2000; i++){
            buffer_r = input_r[i];
            buffer_i = input_i[i];
         
          write_data_to_ACC(lstm_START_ADDR, buffer_r, 4);
          write_data_to_ACC(lstm_START1_ADDR, buffer_i, 4);
        
      
      read_data_from_ACC(lstm_READ_ADDR, buffer_r, 4);
      read_data_from_ACC(lstm_READ1_ADDR, buffer_i, 4);
```

### RISC V multicore implementation 

In this part we use multiple RISC V processors (2) for parallel simulation. Since there are total 100 hidden layers, in the first core first 50 are computed and next 50 in the other core. The code for muti-core implementation is described below: 

```c++
#define PROCESSORS 2
int main(unsigned hart_id) {
	if (hart_id == 0) {
		sem_init(&barrier_lock, 1);
		sem_init(&barrier_sem, 0); //lock all cores initially
		for(int i=0; i< PROCESSORS; ++i){
			sem_init(&print_sem[i], 0); //lock printing initially
		}
		sem_init(&lock, 1);
	
	   
	  sem_wait(&lock);
	  if (hart_id == 0) write_data_to_ACC(GAUSSIANFILTER_START_ADDR, buffer, 4);
	  else write_data_to_ACC(GAUSSIANFILTER_START_1_ADDR, buffer, 4);
	  sem_post(&lock);
}
```

## III. Design Model
The stratus HLS Model is describe here: 

![design](12.PNG)



## IV. Experimental results

- Stratus <br/>
![o](41.PNG)
- RISC V  <br/>
![o](4.png)
- RISC V Multicore <br/>
Core - 0 <br/>
![i](31.png) <br/>
Core - 1 <br/>
![o](32.png)  <br/>

- Comparitive Study  <br/>

| |Simulated Time|
|---------------|---------------|
|Single - Core| 229953920 ns|
|Multi - Core (2 cores) |27873540 ns|
|Software implementation |1131425656 ns|
|RISC V implementation |27873540 ns|

## V. Discussion and Conclusion
In this project we implemented a neural network based digital pre-distortion model is SystemC. For hardware level systhesis we pushed this code to Stratus HLS. And finally for hw/sw co-design we implemented the entire architecture in RISC V VP in single as well as multi-core. This project helped us a lot to apply all the knowledge we learnt in this course. 


 
 

