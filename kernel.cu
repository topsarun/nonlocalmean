#include <fstream>
#include <iostream>
#include <ctime> //For cpu beamfroming
#include <thread>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#define SIGNAL_SIZE		8192	  //H
#define SCAN_LINE		81	      //W

using namespace std;

void make_kernel(int f1, int f2, double *kernel){
	double val1, val2, sum = 0;
	for (int d1 = 1; d1 <= f1; d1++)
	{
		for (int d2 = 1; d2 <= f2; d2++)
		{
			val1 = 1 / powf((2 * d1 + 1), 2);
			val2 = 1 / powf((2 * d2 + 1), 2);
			for (int i = 0 - d1; i <= d1; i++)
			{
				for (int j = 0 - d2; j <= d2; j++)
				{
					kernel[f2 - j + ((f1 - i)*(2 * f2 + 1))] += val1 + val2;
				}
			}
		}
	}
	for (int i = 0; i < 2 * f1 + 1; i++)
	{
		for (int j = 0; j < 2 * f2 + 1; j++)
		{
			kernel[j + (i * (2 * f2 + 1))] /= (f1 + f2 / 2.0);
			sum += kernel[j + (i * (2 * f2 + 1))];
		}
	}
	for (int i = 0; i < 2 * f1 + 1; i++)
	{
		for (int j = 0; j < 2 * f2 + 1; j++)
		{
			kernel[j + (i * (2 * f2 + 1))] /= sum;
		}
	}
}

__global__ void gpu_make_kernel(int f1, int f2, double *kernel){
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	double val1, val2, sum = 0;
	for (int d1 = 1; d1 <= f1; d1++)
	{
		for (int d2 = 1; d2 <= f2; d2++)
		{
			val1 = 1 / powf((2 * d1 + 1), 2);
			val2 = 1 / powf((2 * d2 + 1), 2);
			for (int i = 0 - d1; i <= d1; i++)
			{
				for (int j = 0 - d2; j <= d2; j++)
				{
					kernel[f2 - j + ((f1 - i)*(2 * f2 + 1))] += val1 + val2;
				}
			}
		}
	}
	for (int i = 0; i < 2 * f1 + 1; i++)
	{
		for (int j = 0; j < 2 * f2 + 1; j++)
		{
			kernel[j + (i * (2 * f2 + 1))] /= (f1 + f2 / 2.0);
			sum += kernel[j + (i * (2 * f2 + 1))];
		}
	}
	for (int i = 0; i < 2 * f1 + 1; i++)
	{
		for (int j = 0; j < 2 * f2 + 1; j++)
		{
			kernel[j + (i * (2 * f2 + 1))] /= sum;
		}
	}
}

__global__ void gpu_padarray(double *in, double *out, const int f1, const int f2, const int Wx, const int Wy)
{
	/*
		size in [m,n]
		size out[m+2f1,n+2f2]

		step X -> 1 -> 2 -> 3 -> 4
		-----------
		-    3    -
		-----------
		 1 | X | 2 
		-----------
		-    4    -
		-----------
	*/
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	char count;

	for (int i = tIDy; i < Wy; i += nThdy)
	{
		for (int j = tIDx; j < Wx; j += nThdx)
			out[(j +f2) + ((i+f1)*(Wx+(2*f2)))] = in[j + (i*Wy)]; //X
	}

	for (int i = f1 + tIDy; i < Wy + f1; i += nThdy)
	{
		count = f2 - 1;
		for (int j = tIDx; j < f2; j += nThdx)
			out[j + (i*(Wx + (2 * f2)))] = out[i + (f2 + count--)*(Wx + (2 * f2))]; //1
	}

	for (int i = f1 + tIDy; i < Wy + f1; i += nThdy)
	{
		count = f2 + Wx - 1;
		for (int j = Wx + f2 + tIDx; j < Wx + 2 * f2; j += nThdx)
			out[j + (i * (Wx + 2 * f2))] = out[count-- + (i * (Wx + 2 * f2))]; //2
	}

	count = 2 * f1 - 1;
	for (int i = tIDy; i < f1; i += nThdy)
	{
		for (int j = tIDx; j < Wx + 2 * f1; j += nThdx)
			out[j + (i * (Wx + 2 * f2))] = out[j + ((count)* (Wx + 2 * f2))]; //3
		count--;
	}

	count = f1 + Wy - 1;
	for (int i = f1 + Wy + tIDy; i < Wy + 2 * f1; i += nThdy)
	{
		for (int j = tIDx; j < Wx + 2 * f2; j += nThdx)
			out[j + (i * (Wx + 2 * f2))] = out[j + ((count)* (Wx + 2 * f2))]; //4
		count--;
	}
}

void padarray(double *in, double *out, const int f1, const int f2, const int Wx, const int Wy)
{
	/*
		size in [m,n]
		size out[m+2f1,n+2f2]

		step X -> 1 -> 2 -> 3 -> 4
		-----------
		-    3    -
		-----------
		 1 | X | 2
		-----------
		-    4    -
		-----------
	*/
	int count;
	for (int i = 0; i < Wx + 2 * f1; i++)
	{
		for (int j = 0; j < Wy + 2 * f2; j++)
			out[j + (i * (Wx + (2 * f1)))] = 0; //fill zero
	}

	for (int i = 0; i < Wy; i++)
	{
		for (int j = 0; j < Wx; j++)
			out[(j + f2) + ((i + f1) * (Wx + (2 * f2)))] = in[j + (i * Wy)]; //X
	}

	for (int i = f1; i < Wy + f1; i++)
	{
		count = f2 - 1;
		for (int j = 0; j < f2; j++)
			out[j + (i*(Wx + (2 * f2)))] = out[i + (f2 + count--)*(Wx + (2 * f2))]; //1
	}

	for (int i = f1; i < Wy + f1; i++)
	{
		count = f2 + Wx - 1;
		for (int j = Wx + f2; j < Wx + 2 * f2; j++)
			out[j + (i * (Wx + 2 * f2))] = out[count-- + (i * (Wx + 2 * f2))]; //2
	}

	count = 2 * f1 - 1;
	for (int i = 0; i < f1; i++)
	{
		for (int j = 0; j < Wx + 2 * f1; j++)
			out[j + (i * (Wx + 2 * f2))] = out[j + ((count)* (Wx + 2 * f2))];  //3
		count--;
	}

	count = f1 + Wy - 1;
	for (int i = f1 + Wy; i < Wy + 2 * f1; i++)
	{
		for (int j = 0; j < Wx + 2 * f2; j++)
			out[j + (i * (Wx + 2 * f2))] = out[j + ((count)* (Wx + 2 * f2))];  //4
		count--;
	}
}

void main()
{
	const int f1 = 3;
	const int f2 = 3;
	const int Wx = 5;
	const int Wy = 5;
	
	double *in = new double[Wx * Wy];
	double *out = new double[(Wx + (2 * f1))*(Wy + (2 * f2))];
	
	for (int i = 0; i < Wx;i++)
	{
		for (int j = 0; j < Wy; j++)
			in[j + (i * Wx)] = 5 + (i*j);
	}

	double *d_in;
	double *d_out;

	cudaMalloc((void **)&d_in, Wx*Wy*sizeof(double));
	cudaMalloc((void **)&d_out, (Wx+2*f1)*(Wy+2*f2)*sizeof(double));
	cudaMemcpy(d_in, in, Wx*Wy*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, (Wx + 2 * f1)*(Wy + 2 * f2)*sizeof(double));

	gpu_padarray <<<1024,1024>>> (d_in, d_out, f1, f2, Wx, Wy);
	cudaMemcpy(out, d_out, (Wx + 2 * f1)*(Wy + 2 * f2)*sizeof(double), cudaMemcpyDeviceToHost);

	cout << endl;
	for (int i = 0; i < Wx + 2 * f1; i++)
	{
		for (int j = 0; j < Wy + 2 * f2; j++)
		{
			printf("%d\t", j + (i * (Wx + (2 * f1))));
		}
		cout << endl;
	}

	cout << endl;
	for (int i = 0; i < Wx + 2 * f1; i++)
	{
		for (int j = 0; j < Wy + 2 * f2; j++)
		{
			printf("%lf\t", out[j + (i * (Wx + (2 * f1)))]);
		}
		cout << endl;
	}
	cout << endl;

	double *kernel = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *d_kernel;
	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
		kernel[i] = 0;

	cudaMalloc((void **)&d_kernel, (2 * f1 + 1) * (2 * f2 + 1) * sizeof(double));
	cudaMemset(d_kernel, 0, (2 * f1 + 1) * (2 * f2 + 1) * sizeof(double));

	//gpu_make_kernel << <(32,32), (32,32) >> >(f1, f2, d_kernel);
	//cudaMemcpy(kernel, d_kernel, (2 * f1 + 1) * (2 * f2 + 1) * sizeof(double), cudaMemcpyDeviceToHost);

	make_kernel(f1, f2, kernel);
	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
	{
		if (i % (2 * f2 + 1) == 0) printf("\n");
		printf("%lf ", kernel[i]);
	}
}

void nonlocalmean(const int t1, const int t2, const int f1, const int f2, const int Wx, const int Wy) // Dev
{
	double **w1 = new double *[2 * f1 + 1];
	for (int i = 0; i < 2 * f1 + 1; i++)
		w1[i] = new double[2 * f2 + 1];

	double **w2 = new double *[2 * f1 + 1];
	for (int i = 0; i < 2 * f1 + 1; i++)
		w2[i] = new double[2 * f2 + 1];

	double **input2;

	int i1, j1;
	for (int i = 0; i < Wx; i++)
	{
		for (int j = 0; j < Wy; j++)
		{

			i1 = i + f1;
			j1 = j + f2;

			//W1 = input2(i1 - f1:i1 + f1, j1 - f2 : j1 + f2);
			int Ln = i1 + f1;
			for (int tmp1 = 0; tmp1 < 2 * f1 + 1; tmp1++)
			{
				int Lm = j1 - f2;
				for (int tmp2 = 0; tmp2 < 2 * f2 + 1; tmp2++)
				{
					w1[tmp1][tmp2] = input2[Ln][Lm++];
				}
				Ln++;
			}

			double wmax = 0;
			double average = 0;
			double sweight = 0;
			/*
			rmin = max(i1-t1,f1+1);
			rmax = min(i1+t1,m+f1);
			smin = max(j1-t2,f2+1);
			smax = min(j1+t2,n+f2);
			*/
			double rmin = (i1 - t1>f1 + 1) ? i1 - t1 : f1 + 1  ;
			double rmax = (i1 - t1>f1 + 1) ? f1 + 1  : i1 - t1 ;
			double smin = (j1 - t2>f2 + 1) ? j1 - t2 : f2 + 1  ;
			double smax = (j1 - t2>f2 + 1) ? f2 + 1  : j1 - t2 ;

			for (int r = rmin; r <= rmax; r++)
			{
				for (int s = smin; r <= smax; r++)
				{
					if (r == i1 && s == j1) continue;

					//W2= input2(r-f1:r+f1 , s-f2:s+f2);
					int Ln2 = r - f1;
					for (int tmp1 = 0; tmp1 < 2 * f1 + 1; tmp1++)
					{
						int Lm2 = s - f2;
						for (int tmp2 = 0; tmp2 < 2 * f2 + 1; tmp2++)
						{
							w2[tmp1][tmp2] = input2[Ln2][Lm2++];
						}
						Ln2++;
					}

				}
			}

		}
	}
}