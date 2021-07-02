#include <string.h>
#include "stdio.h"
#include "math.h"
 #include "input_real.h" //included here to avoid compiler issue of not initializing global arrays
  #include "input_imag.h"

// Gaussian Filter ACC
static char* const lstm_START_ADDR = reinterpret_cast<char* const>(0x45000000);
static char* const lstm_READ_ADDR  = reinterpret_cast<char* const>(0x45000070);
static char* const lstm_START1_ADDR = reinterpret_cast<char* const>(0x45000036);
static char* const lstm_READ1_ADDR  = reinterpret_cast<char* const>(0x45000120);

// Gaussian Filter ACC 1
static char* const lstm1_START_ADDR = reinterpret_cast<char* const>(0x45000000);
static char* const lstm1_READ_ADDR  = reinterpret_cast<char* const>(0x45000070);
static char* const lstm1_START1_ADDR = reinterpret_cast<char* const>(0x45000036);
static char* const lstm1_READ1_ADDR  = reinterpret_cast<char* const>(0x45000120);

// DMA 
static volatile uint32_t * const DMA_SRC_ADDR  = (uint32_t * const)0x40000000;
static volatile uint32_t * const DMA_DST_ADDR  = (uint32_t * const)0x40000004;
static volatile uint32_t * const DMA_LEN_ADDR  = (uint32_t * const)0x40000008;
static volatile uint32_t * const DMA_OP_ADDR   = (uint32_t * const)0x4000000C;
static volatile uint32_t * const DMA_STAT_ADDR = (uint32_t * const)0x40000010;
static const uint32_t DMA_OP_MEMCPY = 1;

bool _is_using_dma = true;

unsigned int ReadfromByteArray(unsigned char* array, unsigned int offset) {
	unsigned int output = (array[offset] << 0) | (array[offset + 1] << 8) | (array[offset + 2] << 16) | (array[offset + 3] << 24);
	return output;
}

union word{
    int uint;
    char uc[4];
};

void write_data_to_ACC(char* ADDR, int buffer, int len){
	word data;
	unsigned char buff[4];
	
	data.uint = buffer;
	buff[0] = data.uc[0];
        buff[1] = data.uc[1];
        buff[2] = data.uc[2];
        buff[3] = data.uc[3];
	
	 if(_is_using_dma){  
	    // Using DMA 
	    *DMA_SRC_ADDR = (uint32_t)(buff);
	    *DMA_DST_ADDR = (uint32_t)(ADDR);
	    *DMA_LEN_ADDR = len;
	    *DMA_OP_ADDR  = DMA_OP_MEMCPY;
	  }else{
	    // Directly Send
	    memcpy(ADDR, buff, sizeof(unsigned char)*len);
	  }
	}

void read_data_from_ACC(char* ADDR, int buffer, int len){

	word data1;
	unsigned char buff[4];
	
	
	
	 if(_is_using_dma){  
	    // Using DMA 
	    *DMA_SRC_ADDR = (uint32_t)(ADDR);
	    *DMA_DST_ADDR = (uint32_t)(buff);
	    *DMA_LEN_ADDR = len;
	    *DMA_OP_ADDR  = DMA_OP_MEMCPY;
	  }else{
	    // Directly Send
	    memcpy(buff,ADDR, sizeof(unsigned char)*len);
	  }
	  data1.uc[0] = buff[0]; 
         data1.uc[1] = buff[1];
         data1.uc[2] = buff[2];
	 data1.uc[3] = buff[3];
	 //printf("%d",buff[0]);
	 buffer = data1.uint;
	}

int sem_init (uint32_t *__sem, uint32_t count) __THROW
{
  *__sem=count;
  return 0;
}

int sem_wait (uint32_t *__sem) __THROW
{
  uint32_t value, success; //RV32A
  __asm__ __volatile__("\
L%=:\n\t\
     lr.w %[value],(%[__sem])            # load reserved\n\t\
     beqz %[value],L%=                   # if zero, try again\n\t\
     addi %[value],%[value],-1           # value --\n\t\
     sc.w %[success],%[value],(%[__sem]) # store conditionally\n\t\
     bnez %[success], L%=                # if the store failed, try again\n\t\
"
    : [value] "=r"(value), [success]"=r"(success)
    : [__sem] "r"(__sem)
    : "memory");
  return 0;
}

int sem_post (uint32_t *__sem) __THROW
{
  uint32_t value, success; //RV32A
  __asm__ __volatile__("\
L%=:\n\t\
     lr.w %[value],(%[__sem])            # load reserved\n\t\
     addi %[value],%[value], 1           # value ++\n\t\
     sc.w %[success],%[value],(%[__sem]) # store conditionally\n\t\
     bnez %[success], L%=                # if the store failed, try again\n\t\
"
    : [value] "=r"(value), [success]"=r"(success)
    : [__sem] "r"(__sem)
    : "memory");
  return 0;
}

int barrier(uint32_t *__sem, uint32_t *__lock, uint32_t *counter, uint32_t thread_count) {
	sem_wait(__lock);
	if (*counter == thread_count - 1) { //all finished
		*counter = 0;
		sem_post(__lock);
		for (int j = 0; j < thread_count - 1; ++j) sem_post(__sem);
	} else {
		(*counter)++;
		sem_post(__lock);
		sem_wait(__sem);
	}
	return 0;
}

void sprintfloat(char* buf, float num) {
	const char* tmpSign = (num < 0) ? "-" : "+";
	float tmpVal = (num < 0) ? -num : num;

	int tmpInt1 = (int) tmpVal;                  // Get the integer (678).
	float tmpFrac = tmpVal - tmpInt1;      // Get fraction (0.0123).
	int tmpInt2 = trunc(tmpFrac * 10000);  // Turn into integer (123).

	// Print as parts, note that you need 0-padding for fractional bit.
	sprintf(buf, "%s%d.%04d", tmpSign, tmpInt1, tmpInt2);
}

//Total number of cores
//static const int PROCESSORS = 2;
#define PROCESSORS 2
//the barrier synchronization objects
uint32_t barrier_counter=0; 
uint32_t barrier_lock; 
uint32_t barrier_sem; 
//the mutex object to control global summation
uint32_t lock;  
//print synchronication semaphore (print in core order)
uint32_t print_sem[PROCESSORS]; 
float pi_over_4 = 0;
int result_saver[2][1100];

int main(unsigned hart_id) {

	/////////////////////////////
	// thread and barrier init //
	/////////////////////////////
	if (hart_id == 0) {
		// create a barrier object with a count of PROCESSORS
		sem_init(&barrier_lock, 1);
		sem_init(&barrier_sem, 0); //lock all cores initially
		for(int i=0; i< PROCESSORS; ++i){
			sem_init(&print_sem[i], 0); //lock printing initially
		}
		// Create mutex lock
		sem_init(&lock, 1);
	}

	/////////////////////////////////////////
	// accumulate local sum to shared data //
	/////////////////////////////////////////
  //unsigned char* source_array= input_r;

  unsigned int input_rgb_raw_data_offset = 8;
  unsigned int width = 2000;
  unsigned int length = 0;
  unsigned int bytes_per_pixel =  32;
  //unsigned char* source_bitmap = &source_array[input_rgb_raw_data_offset];
  sem_wait(&lock);
  printf ("hart_id = %d\n",hart_id);
  printf("======================================\n");
  printf("\t  Reading from array\n");
  printf("======================================\n");
  printf(" input_rgb_raw_data_offset\t= %d\n", input_rgb_raw_data_offset);
  printf(" width\t\t\t\t= %d\n", width);
  printf(" length\t\t\t\t= %d\n", length);
  printf(" bytes_per_pixel\t\t= %d\n",bytes_per_pixel);
  printf("======================================\n");
  sem_post(&lock);

  int start_width = width / PROCESSORS * hart_id, end_width = width / PROCESSORS * hart_id + width / PROCESSORS;
printf("\n%d\n",start_width);
printf("\n%d\n",end_width);
int buffer_r= 0; int buffer_i =0; int out_r = 0; int out_i = 0;
  for(int i = start_width; i < end_width; i++){

          buffer_r = input_r[i];
          buffer_i = input_i[i];
          //printf ("i = %d j = %d v = %d u = %d\n",i,j,v,u);
          sem_wait(&lock);
          //printf ("i = %d j = %d v = %d u = %d\n",i,j,v,u);
          if (hart_id == 0) {
          write_data_to_ACC(lstm_START_ADDR, buffer_r, 4);
          write_data_to_ACC(lstm_START1_ADDR, buffer_i, 4);
          }
          else {
          write_data_to_ACC(lstm1_START_ADDR, buffer_r, 4);
          write_data_to_ACC(lstm1_START1_ADDR, buffer_i, 4);
          }
          sem_post(&lock);
          printf ("Finish write data\n");
        
      
      printf ("Start read data\n");
      sem_wait(&lock);
      if (hart_id == 0) {
      read_data_from_ACC(lstm_READ_ADDR, buffer_r, 4);
      read_data_from_ACC(lstm_READ1_ADDR, buffer_i, 4);
      }
      else {
      read_data_from_ACC(lstm1_READ_ADDR, buffer_r, 4);
      read_data_from_ACC(lstm1_READ1_ADDR, buffer_i, 4);
      
      }
      result_saver[0][i] = buffer_r;
      /*result_saver[0][j][1] = buffer_r[1];
      result_saver[0][j][2] = buffer_r[2];
      result_saver[0][j][3] = buffer_r[3];*/
      
      result_saver[1][i] = buffer_i;
      /*result_saver[1][j][1] = buffer_i[1];
      result_saver[1][j][2] = buffer_i[2];
      result_saver[1][j][3] = buffer_i[3];*/
      sem_post(&lock);
      //printf ("buffer = %d %d %d %d\n",buffer[0],buffer[1],buffer[2],buffer[3]);
      printf ("Finish read data\n");
      printf ("%d\n",i);
    
  }

	////////////////////////////
	// barrier to synchronize //
	////////////////////////////
	//Wait for all threads to finish
	barrier(&barrier_sem, &barrier_lock, &barrier_counter, PROCESSORS);

	if (hart_id == 0) {  // Core 0 print first and then others
    sem_wait(&lock);
		for(int j = start_width; j < start_width+10; j++){

        printf ("real : [%d %d] %d %d %d %d\n",j,j,result_saver[0][j],result_saver[0][j],result_saver[0][j],result_saver[0][j]);
        printf ("imag : [%d %d] %d %d %d %d\n",j,j,result_saver[1][j],result_saver[1][j],result_saver[1][j],result_saver[1][j]);
    }
    sem_post(&lock);
	} else {
		for (int i = 1; i < PROCESSORS; ++i) {
      sem_wait(&lock);
      for(int j = start_width; j < start_width+10; j++){
          //sem_wait(&print_sem[i]); 
          printf ("real :[%d %d] %d %d %d %d\n",j,j,result_saver[0][j],result_saver[0][j],result_saver[0][j],result_saver[0][j]);
          printf ("imag :[%d %d] %d %d %d %d\n",j,j,result_saver[1][j],result_saver[1][j],result_saver[1][j],result_saver[1][j]);
          //sem_post(&print_sem[i + 1]);

      }
      sem_post(&lock);
		}
	}

en	return 0;
}


