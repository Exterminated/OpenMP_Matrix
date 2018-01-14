#include "iostream"
#include "math.h"
#include <conio.h>
#include <omp.h>

using namespace std;

double GetRand() {
	return ((double)rand() / (RAND_MAX));
}
void UsualCalculations(int N, double **a, double **b, double **c) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			c[i][j] += a[i][j] * b[i][j];
		}
	}
	
}
void OpenMPCalculations(int N, double **a, double **b, double **c) {
	int i, j;
#pragma omp parallel 
	{
#pragma omp for firstprivate(j) lastprivate(i) reduction(+: c)
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				c[i][j] += a[i][j] * b[i][j];				 
			}
		}
	}
}
//void OpenCLCalculations(int N, double **a, double **b, double **c) {
//		// размер кэша тредов
//	#define TS 16
//		// конвертер индексов матрицы в линейный адрес 
//	#define IDX2LIN(i,j,l) (i+j*l) 
//		__kernel void myGEMM2(const int M, const int N, const int K, const __global float* A, const __global float* B, __global float* C){ 
//			// 2D номер треда в группе
//			const int r = get_local_id(0); const int c = get_local_id(1); 
//		// номер €чейки в матрице результата 
//		const int gr = get_group_id(0)*TS + r;
//		// 0..M
//		const int gc = get_group_id(1)*TS + c;
//		// 0..N
//		// общий кэш дл€ тредов группы 
//		__local float Asub[TS][TS]; 
//		__local float Bsub[TS][TS]; float acc = 0.0f; 
//		// результат работы треда 
//		for (int t=0; t<K/TS; t++) { 
//			// цикл по всем блокам матриц 
//			const int tr = t*TS + r; 
//			const int tc = t*TS + c; 
//			// загружаем блоки в кэш 
//			Asub[c][r] = A[ IDX2LIN(gr,tc,M) ];
//			Bsub[c][r] = B[ IDX2LIN(tr,gc,K) ]; 
//			barrier(CLK_LOCAL_MEM_FENCE); 
//			// ждЄм пока треды группы заполн€т общий кэш
//			// вычисл€ем 
//			for (int k=0; k<TS; k++) {
//				acc += Asub[k][r] * Bsub[c][k]; 
//			} 
//			barrier(CLK_LOCAL_MEM_FENCE); 
//			// ждЄм пока треды группы завершат вычислени€ 
//		} 
//		C[ IDX2LIN(gr,gc,M) ] = acc;
//		// сохран€ем результат 
//	}
//}
int main() {
	int N = 10000;

	int i, j, l;
	
	double **a = new double*[N];
	for (i = 0; i < N; i++) a[i] = new double[N];
	double **b = new double*[N];
	for (i = 0; i < N; i++) b[i] = new double[N];
	double **c = new double*[N];
	for (i = 0; i < N; i++) c[i] = new double[N];

	double start = 0.0;	
	double end = 0.0;

	//double wtick = omp_get_wtick();

	cout << "Matrix size: " << N << endl;

	cout << "a matrix" << endl;
	for (i = 0; i<N; i++) {
		for (j = 0; j<N; j++) {
			a[i][j] = GetRand();
		}
	}

	cout << "b matrix" << endl;
	for (i = 0; i<N; i++) {
		for (j = 0; j<N; j++) {
			b[i][j] = GetRand();
		}
	}

	start = omp_get_wtime();
	UsualCalculations(N, a, b, c);
	end = omp_get_wtime();
	cout << "Time usual: " << end - start << endl;

	omp_set_dynamic(16);
	omp_set_num_threads(8);
	
	start = 0.0;
	end = 0.0;

	start = omp_get_wtime();
	OpenMPCalculations(N, a, b, c);
	end = omp_get_wtime();
	cout << "Time OMP: " << end - start << endl;

	for (i = 0; i < N; i++) delete[] a[i];
	delete[] a;
	for (i = 0; i < N; i++) delete[] b[i];
	delete[] b;
	for (i = 0; i < N; i++) delete[] c[i];
	delete[] c;

	_getch();

	return 0;
}