#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"



int main(void) {
	double pd[100];
	double rho[100];
	double DR[100];
	int i;

	/*PDとrhoをそれぞれARモデルに従ってシミュレーション用にサンプリング*/
	AR_sim(100,pd, 0.04, 0.05, 0.95);
	AR_sim(100,rho, 0.1, 0.05, 0.95);
	
	for (i = 1; i < 100; i++) {
		DR[i] = g_DR_fn(i / 100, pd[1], rho[1]);
	}


	for (i = 0; i < 100; i++) {
		DR[i] = reject_sample(pd[i], rho[i]);
	}

	return 0;
}

