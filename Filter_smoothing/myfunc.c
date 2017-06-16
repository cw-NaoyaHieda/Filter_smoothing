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

/*���K���z�̗ݐϊm���_*/
double pnorm(double q, double mu, double sigma) {
	return  (1 + erf((q - mu) / (sqrt(2)* sigma))) / 2;
}

/*���K���z�̃p�[�Z���g�_ �˓c�̋ߎ����@R�Ƃ͈Ⴄ�\������*/
double qnorm(double qn)
{
	static double b[11] = { 1.570796288,     0.03706987906,  -0.8364353589e-3,
		-0.2250947176e-3, 0.6841218299e-5, 0.5824238515e-5,
		-0.104527497e-5,  0.8360937017e-7,-0.3231081277e-8,
		0.3657763036e-10,0.6936233982e-12 };
	double w1, w3;
	int i;

	if (qn < 0. || 1. < qn)
	{
		fprintf(stderr, "Error : qn <= 0 or qn >= 1  in pnorm()!\n");
		return 0.;
	}
	if (qn == 0.5)	return 0.;

	w1 = qn;
	if (qn > 0.5)	w1 = 1. - w1;
	w3 = -log(4. * w1 * (1. - w1));
	w1 = b[0];
	for (i = 1; i < 11; i++)	w1 += (b[i] * pow(w3, (double)i));
	if (qn > 0.5)	return sqrt(w1 * w3);
	return -sqrt(w1 * w3);
}

/*�f�t�H���g���̖��x�֐� Hull*/
double g_DR_fn(double DR, double PD, double rho) {
	double prob;
	prob = sqrt((1 - rho) / rho)*
		exp(0.5 * (pow(qnorm(DR,0,1), 2) - pow((sqrt(1 - rho)*qnorm(DR,0,1) - qnorm(PD,0,1)) / sqrt(rho), 2)));
	return prob;
}

