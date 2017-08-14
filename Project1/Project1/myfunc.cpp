/* �֐���`*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#define M_PI 3.14159265359	
#include <iostream>
#include <vector>
#include <random>

std::mt19937 mt2(180);
std::uniform_real_distribution<double> r_rand2(0.0, 1.0);

using namespace std;

/*�V�O���C�h�֐�*/
double sig(double x) {
	return  1 / (1 + exp(-x));
}

double sig_env(double x) {
	double y;
	y = x / (1 - x);
	return  log(y);
}


/*���K���z���痐���@Box-muller�@*/
double rnorm(double mu, double sd) {
	double z = sqrt(-2.0*log(r_rand2(mt2))) * sin(2.0*M_PI*r_rand2(mt2));
	return mu + sd*z;
}

/*���K���z�̖��x�֐�*/
double dnorm(double x, double mu, double sd) {
	return 1 / (sd * sqrt(2 * M_PI))*exp(-pow((x - mu), 2) / (pow(sd, 2) * 2));
}

/*���K���z�̗ݐϊm���_*/
double pnorm(double q, double mu, double sd) {
	return  (1 + erf((q - mu) / (sqrt(2)* sd))) / 2;
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
		exp(0.5 * (pow(qnorm(DR), 2) - pow((sqrt(1 - rho)*qnorm(DR) - qnorm(PD)) / sqrt(rho), 2)));
	return prob;
}

/*�f�t�H���g���̕��z�֐� Hull*/
double g_DR_distribution(double DR, double PD, double rho) {
	double prob;
	prob = pnorm(
		(sqrt(1 - rho)*qnorm(DR) - qnorm(PD)) / sqrt(rho), 0, 1
	);
	return prob;
}

/*�f�t�H���g���̊m�� Hull*/
double g_DR_(double DR, double PD, double rho) {
	double prob;
	prob = g_DR_distribution(DR + 0.00005, PD, rho) - g_DR_distribution(DR - 0.00005, PD, rho);
	return prob;
}


/*Dynamicdefaultrate�ł̖��x�֐��@�������K���z�@DR�𐳋K���z�̋t�֐��ŕϊ�����K�v�����邱�Ƃɒ���*/
double g_DR_dinamic(double tilde_DR, double X_t_1, double q_qnorm, double beta, double rho) {
	return dnorm(tilde_DR, (q_qnorm - sqrt(rho)*sqrt(beta)*X_t_1) / sqrt(1 - rho), sqrt(rho)*sqrt(1 - beta) / sqrt(1 - rho));
}
