#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define T 10
#define N 1000
#define tau_rho 0.95
#define tau_pd 0.95
#define mean_rho 0.1
#define mean_pd 0.04
#define sd_sig_rho 0.05
#define sd_sig_pd 0.05



int main(void) {
	/*Answer�i�[*/
	double pd[T];
	double rho[T];
	double DR[T];

	/*�������z����̃T���v�����O�i�[*/
	double first_pd_sig[N];
	double first_rho_sig[N];

	/*���_t�̗\���l�i�[*/
	double pred_pd_sig[N];
	double pred_rho_sig[N];
	double pred_pd[N];
	double pred_rho[N];
	

	int i;
	int n;
	double j;

	/*PD��rho�����ꂼ��AR���f���ɏ]���ăV�~�����[�V�����p�ɃT���v�����O*/
	AR_sim(T,pd, mean_pd, sd_sig_pd, tau_pd);
	AR_sim(T,rho, mean_rho, sd_sig_rho, tau_rho);
	
	/*���p�@��p���āA�e���_�ł̃p�����[�^����DR�𔭐�*/
	for (i = 0; i < T; i++) {
		DR[i] = reject_sample(pd[i], rho[i]);
	}

/*�ȉ��͊֐��ɂ���K�v������*/
	/*���_1�ł̃t�B���^�����O�J�n*/
	/*�������z����̃T���v�����O���A���̂܂܎��_1�̃T���v�����O*/
	for (n = 0; n < N; n++) {
		/*�������z����@���_0�ƍl����*/
		first_pd_sig[n] = rnorm(sig_env(mean_pd), sd_sig_pd);
		first_rho_sig[n] = rnorm(sig_env(mean_rho), sd_sig_rho);
		/*���̌��ʂ���T���v�����O*/
		pred_pd_sig[n] = rnorm(mean_pd+tau_pd*(first_pd_sig[n]-mean_pd), sd_sig_pd);
		pred_rho_sig[n] = rnorm(mean_rho + tau_rho*(first_rho_sig[n] - mean_rho), sd_sig_rho);
		/*pd �� rho�ɕϊ�*/
		pred_pd[n] = sig(pred_pd_sig[n]);
		pred_rho[n] = sig(pred_rho_sig[n]);
	}
	
	


	return 0;
}




/*gnuplot��p����DR�̖��x���m�F
for (i = 1; i < 100; i++) {
j = i / 100.0;
DR[i] = g_DR_fn(j, pd[1], rho[1]);
}

FILE *gp;
gp = _popen(GNUPLOT_PATH, "w");
fprintf(gp, "set xrange [0:1]\n");
fprintf(gp, "set yrange [0:20]\n");
fprintf(gp, "plot '-' with lines linetype 1 title \"DR\"\n");
for (i = 1; i < 100; i++) {
fprintf(gp, "%f\t%f\n", i/100.0,DR[i]);
}
fprintf(gp, "e\n");
fflush(gp);

fprintf(gp, "exit\n");	// gnuplot�̏I��
_pclose(gp);

*/

