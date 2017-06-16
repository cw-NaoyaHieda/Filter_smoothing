/* �֐���`*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "MT.h"
#define M_PI 3.14159265359	

/*�V�O���C�h�֐�*/
double sig(double x) {
	return (tanh(x) + 1) / 2;
}

double sig_env(double x) {
	double y;
	y = x / (1 - x);
	return log(y)/2;
}


/*�����Z���k�c�C�X�^�[�@�ɂ���Ĕ��������^������(��l���z)*/
double Uniform(void) {
	return genrand_real3();
}

/*���K���z���痐���@Box-muller�@*/
double rnorm(double mu, double sigma) {
	double z = sqrt(-2.0*log(Uniform())) * sin(2.0*M_PI*Uniform());
	return mu + sigma*z;
}

/*���K���z�̖��x�֐�*/
double dnorm(double x, double mu, double sigma) {
	return 1 / (sigma * sqrt(2 * M_PI))*exp(-(x - mu) * (x - mu)/ (sigma * sqrt(2)) );
}

/*���K���z�̕��ʓ_*/
double qnorm(double q, double mu, double sigma) {
	return  (1 + erf((q - mu) / (sqrt(2)* sigma))) / 2;
}

/*�f�t�H���g���̖��x�֐� Hull*/
double g_DR_fn(double DR, double PD, double rho) {
	double prob;
	prob = sqrt((1 - rho) / rho)*
		exp(0.5 * (pow(qnorm(DR,0,1), 2) - pow((sqrt(1 - rho)*qnorm(DR,0,1) - qnorm(PD,0,1)) / sqrt(rho), 2)));
	return prob;
}

