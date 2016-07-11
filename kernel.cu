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

void multi(double *out, double *a, double *b, const int Ra, const int Rb, const int Ca, const int Cb);
void subMaxtrix(double *out, double *a, double *b, const int R, const int C);
double sumMaxtrix(double *in, const int R, const int C);
void nonlocalmean(const int t1, const int t2, const int f1, const int f2, const int Wx, const int Wy, double *in, double *out, int h);

void writeFile(const char *filename, const int size, double* readArray)
{
	//const char *filename = "D:\\save.dat";
	ofstream output(filename, std::ios::binary | std::ios::out);



	for (int i = 0; i < size; i++)
	{
		//file.read( (char*)(&aDouble), sizeof( double ) );  
		//readArray[m + (n*SIGNAL_SIZE) + (i*SIGNAL_SIZE*CHANNEL)] = aDouble;
		output.write((char *)&readArray[i], sizeof(double));

	}

	output.close();
}

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
	const int t1 = 5;
	const int t2 = 5;
	const int Wx = 81;
	const int Wy = 8192;
	const int h  = 15;

	double *kernel  = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *kernel0 = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
		kernel[i] = 0; // Fill zero in kernel

	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
		kernel0[i] = 9; // Fill zero in kernel

	make_kernel(f1, f2, kernel);

	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
	{
		if (i % (2 * f2 + 1) == 0) printf("\n");
		printf("%lf ", kernel[i]);
	}

	// * matic
	//multi(kernel0, kernel, kernel, (2 * f1 + 1), (2 * f2 + 1), (2 * f1 + 1), (2 * f2 + 1)); // -> before test bug 
	//subMaxtrix(kernel0, kernel0, kernel, (2 * f1 + 1), (2 * f2 + 1));
	//double x = sumMaxtrix(kernel, (2 * f1 + 1), (2 * f2 + 1)); 
	//cout << "AAA" << x;
	double *A = new double[Wx*Wy];
	for (int l = 0; l < Wx*Wy; l++)
	{
		//if (l%Wx == 0) cout << endl;
		A[l] = l*(rand() % 100 / 100.0);
		//printf("%lf\t", A[l]);
	}

	double *B = new double[Wx*Wy];
	for (int l = 0; l < Wx*Wy; l++)
		B[l] = 0;
	
	cout << endl;
	nonlocalmean(t1, t2, f1, f2, Wx, Wy, A, B, h);
	writeFile("D://A.dat", 8192 * 81, A);
	writeFile("D://B.dat", 8192 * 81, B);
}

void multi(double *out, double *a, double *b, const int Ra, const int Rb, const int Ca, const int Cb)
{
	for (int i = 0; i < Ra; i++)
	{
		for (int j = 0; j < Ca; j++)
		{
			for (int k = 0; k < Cb; k++)
				out[j + (i * Cb)] += a[k + i*Ca] * b[j + k*Cb];
		}
	}
	
	//display output
	/*
	for (int i = 0; i < Ra; i++)
	{
		for (int j = 0; j < Cb; j++)
		{
			printf("%lf\t", out[j + (i*Cb)]);
		}
		cout << endl ;
	}
	*/
}

void subMaxtrix(double *out, double *a, double *b, const int R, const int C)
{
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			out[j + (i*C)] = a[j + (i*C)] - b[j + (i*C)];
		}
	}

	//display output
	/*
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			printf("%lf\t", out[j + (i*C)]);
		}
		cout << endl;
	}
	*/
}

double sumMaxtrix(double *in, const int R, const int C)
{
	double sum = 0;
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			sum += in[j + (i*C)];
		}
	}
	return sum;
}

void nonlocalmean(const int t1, const int t2, const int f1, const int f2, const int Wx, const int Wy, double *in, double *out, int h) // Dev
{
	double *padout = new double[(2 * f1 + Wx) * (2 * f2 + Wy)];
	
	double *kernel = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *W1 = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *W2 = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *temp0 = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *temp1 = new double[(2 * f1 + 1) * (2 * f2 + 1)];
	double *temp2 = new double[(2 * f1 + 1) * (2 * f2 + 1)];

	for (int i = 0; i < (2 * f1 + 1)*(2 * f2 + 1); i++)
		kernel[i] = 0; // Fill zero in kernel

	int i1, j1, count=0;
	double wmax = 0;
	double average = 0;
	double sweight = 0;
	double d, w;
	int rmin, rmax, smin, smax ;
	gpu_padarray <<<1,128>>> (in, padout, f1, f2, Wx, Wy);
	make_kernel(f1, f2, kernel);
	h = h*h;

	for (int i = 0; i < Wy; i++)
	{
		printf("%d ", i);
		for (int j = 0; j < Wx; j++)
		{
			//printf("A");
			i1 = i + f1 -1;
			j1 = j + f2 -1;

			//W1= input2(i1-f1:i1+f1 , j1-f2:j1+f2);
			count = 0;
			for (int tmp1 = i1 - f1; tmp1 < i1 + f1; tmp1++)
			{
				for (int tmp2 = j1 - f2; tmp2 < j1 + f2; tmp2++)
				{
					//printf("%d\n", tmp2);
					W1[count++] = padout[tmp2 + (tmp1 * (2 * f2 + 1))];

				}
			}

			wmax = 0;
			average = 0;
			sweight = 0;

			rmin = (i1 - t1>f1 + 1) ? i1 - t1 : f1 + 1;
			rmax = (i1 + t1 < Wy + f1) ? i1 + t1 : Wx + f1;
			smin = (j1 - t2>f2 + 1) ? j1 - t2 : f2 + 1;
			smax = (j1 + t2 < Wx + f2) ? j1 + t2 : Wx + f2;

			for (int r = rmin; r < rmax; r++)
			{
				for (int s = smin; s < smax; s++)
				{
					if (r == i1 && s == j1) continue;

					//W2= input2(r-f1:r+f1 , s-f2:s+f2);
					count = 0;
					for (int tmp1 = r - f1; tmp1 < r + f1; tmp1++)
					{
						for (int tmp2 = s - f2; tmp2 < s + f2; tmp2++)
						{
							W2[count++] = padout[tmp2 + (tmp1 * (2 * f2 + 1))];
						}
					}

					//d = sum(sum(kernel.*(W1-W2).*(W1-W2)));
					subMaxtrix(W1, W1, W2, (2 * f1 + 1), (2 * f2 + 1)); //W1 = W1-W2
					multi(temp0, kernel, W1, (2 * f1 + 1), (2 * f2 + 1), (2 * f1 + 1), (2 * f2 + 1)); //temp = kernel * (W1-w2)
					multi(temp1, temp0, W1, (2 * f1 + 1), (2 * f2 + 1), (2 * f1 + 1), (2 * f2 + 1)); //temp = kernel * (W1-w2) * (W1-W2)
					d = sumMaxtrix(temp1, (2 * f1 + 1), (2 * f2 + 1));
					w = exp(0-d / h);
					if (w>wmax) wmax = w;
					sweight = sweight + w;
					average = average + w * padout[s + r*(2 * f2 + 1)];
				}
			}

			average = average + wmax*padout[j1 + i1*(2 * f2 + 1)];
			sweight = sweight + wmax;

			if (sweight > 0)
				out[j + i*Wx] = average / sweight;
			else
				out[j + i*Wx] = in[j + i*Wx];
		}
	}
}