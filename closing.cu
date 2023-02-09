#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */
 
int M = 10; //heigth
int N = 10; //witdh

void generateImg(int witdh,int height,int *img){
    int img_size = witdh * height;
    for (int i = 0; i < img_size; i++){
        if (i < img_size/2){
            img[i]=1;
        }
        else img[i]=0;
        
    }
}

void generateClc(int *img,int *clc,int size_of_filter, int witdh, int height){
  int index = (size_of_filter-1)/2;
  for(int i =0;i<witdh;i++){
      for(int j=0;j<height;j++){
         int row = i;
         int col = j;
         int index_row = row+index;
         int index_col = col+index;
         clc[index_row+index_col*(witdh+2*index)] = img[row+col*witdh];
        }
     }      
}



__global__ void erosionImg(int *res,int *clc,int witdh,int height,int size_of_filter){
    
    int index = (size_of_filter -1) /2;
    int row = blockIdx.x * blockDim.x + threadIdx.x+ index;
    int col = blockIdx.y * blockDim.y + threadIdx.y+ index;
    int pixel = 1;
    
    for (int i = 0; i<size_of_filter; i++){
      for (int j= 0;j<size_of_filter;j++){
          if (row < witdh +index && col < height + index){
            int tmp = row + col * (witdh + index * 2);
            if (pixel > clc[row + (i - index) + (col +(j- index)) * (witdh + index *2)]){
              pixel = clc[tmp +i - index + (j- index) * (witdh+ index *2)];
            }
          }
        
      }
    }
    
    if (row < witdh +index && col < height + index){
        res[(col-index)*witdh+row-index] = pixel;
    }
    
}


void dilationCPU(int *res_cpu, int *clc, int witdh, int height, int size_of_filter){
    int index = (size_of_filter -1) /2;
    int pixel;
    for (int i = index; i<height+ index; i++){
        for (int j = index; j<witdh+index;j++){
            pixel =0;
            for (int k = 0;k<size_of_filter;k++){
                for (int l =0; l<size_of_filter;l++){
                    if (pixel < clc[j-index+k+(i-index+l)*(witdh+2*index)]){
                        pixel = clc[j-index+k+(i-index+l)*(witdh+2*index)];
                    }
                }
            }
            res_cpu[j-index + (i-index) * witdh] = pixel;
        }
    }
}

void erosionCPU(int *res_cpu, int *clc, int witdh, int height, int size_of_filter){
    int index = (size_of_filter -1) /2;
    int pixel;
    for (int i = index; i<height+ index; i++){
        for (int j = index; j<witdh+index;j++){
            pixel =1;
            for (int k = 0;k<size_of_filter;k++){
                for (int l =0; l<size_of_filter;l++){
                    if (pixel > clc[j-index+k+(i-index+l)*(witdh+2*index)]){
                        pixel = clc[j-index+k+(i-index+l)*(witdh+2*index)];
                    }
                }
            }
            res_cpu[j-index + (i-index) * witdh] = pixel;
        }
    }
}



__global__ void dilationImg(int *res,int *clc,int witdh,int height,int size_of_filter){
    
    int index = (size_of_filter -1) /2;
    int row = blockIdx.x * blockDim.x + threadIdx.x+ index;
    int col = blockIdx.y * blockDim.y + threadIdx.y+ index;
    int pixel = 0;
    
    
    for (int i = 0; i<size_of_filter; i++){
      for (int j= 0;j<size_of_filter;j++){
          if (row < witdh +index && col < height){
            int tmp = row + col * (witdh + index * 2);
            if (pixel < clc[row + (i - index) + (col +(j- index)) * (witdh + index *2)]){
              pixel = clc[tmp +i - index + (j- index) * (witdh+ index *2)];
            }
          }
        
      }
    }
    
    if (row < witdh +index && col < height){
        res[(col-index)*witdh+row-index] = pixel;
    }
}


int main()
{

  
  //Assume that the image is black and white.
  
  int witdh = N;
  int height = M;

  //filter can only be an odd number
  int size_of_filter = 3;


  //Prepare the size of the matrix who'll help us to do the calculation
  int clc_witdh = witdh + size_of_filter -1;
  int clc_height = height + size_of_filter -1;

  
  int size = witdh * height * sizeof(int);
  int clc_size = clc_witdh * clc_height * sizeof(int);
  int *img;
  int *res;
  int *res_ero_gpu;
  int *res_cpu;
  int *res_ero_cpu;
  int *clc_cpu;
  int *clc_gpu;
  cudaMallocManaged(&img,size);
  cudaMallocManaged(&res,size);
  cudaMallocManaged(&res_cpu,size);
  cudaMallocManaged(&res_ero_gpu,size);
  cudaMallocManaged(&res_ero_cpu,size);
  cudaMallocManaged(&clc_cpu,clc_size);
  cudaMallocManaged(&clc_gpu,clc_size);
  
  //Call function to create the image;
  generateImg(witdh,height,img);
  
  printf("Image : \n");
      for (int i =0;i<height;i++){
          printf("\n");
          for (int j=0;j<witdh;j++){
              printf ("%d   ",img[i*witdh+j]);
          }
      }
  
  generateClc(img,clc_cpu,size_of_filter,witdh,height);
  generateClc(img,clc_gpu,size_of_filter,witdh,height);
  

  dim3 threads_per_block (16, 16, 1);
  dim3 number_of_blocks ((witdh / threads_per_block.x) + 1, (height / threads_per_block.y) + 1, 1);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  dilationImg<<<number_of_blocks, threads_per_block>>>(res_ero_gpu,clc_gpu, witdh, height,size_of_filter);
  cudaDeviceSynchronize();
  dilationCPU(res_ero_cpu,clc_cpu,witdh,height,size_of_filter);
  
  generateClc(res_ero_gpu,clc_gpu,size_of_filter,witdh,height);
  generateClc(res_ero_cpu,clc_cpu,size_of_filter,witdh,height);
  
  
  erosionImg<<<number_of_blocks, threads_per_block>>>(res,clc_gpu, witdh, height,size_of_filter);
  cudaDeviceSynchronize();
  erosionCPU(res_cpu,clc_cpu,witdh,height,size_of_filter);
  
  printf("\n");
  
  for (int i = 0; i<N*M;i++){
     if (res[i] != res_cpu[i]){
         printf("res = %d res_cpu = %d ____ i value = %d\n",res[i],res_cpu[i],i);
     }
  }

  printf("\n\n GPU erosion: \n"); 
  for (int i =0;i<height;i++){
      printf("\n");
      for (int j=0;j<witdh;j++){
          printf ("%d   ",res[i*witdh+j]);
      }
  }
  printf("\n\n CPU erosion: \n");
  for (int i =0;i<height;i++){
      printf("\n");
      for (int j=0;j<witdh;j++){
          printf ("%d   ",res_cpu[i*witdh+j]);
      }
  }


  
  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaFree(img);
  cudaFree(res);
  cudaFree(res_ero_gpu);
  cudaFree(res_cpu);
  cudaFree(res_ero_cpu);
  cudaFree(clc_cpu);
  cudaFree(clc_gpu);
}
;
