/* �֐���`*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#define M_PI 3.14159265359	

double *data_vector(int T) {
	double *data_vector = malloc(sizeof(int) * T);
	return data_vector;
}

/*AR���f������̃T���v�� �n��̏����l�̃|�C���^���󂯎���Ă���*/
double* AR_sim(int T,double *y,double mu,double sigma,double phi) {
	/*AR���f���̕���*/
	double sig_mu;
	/*���̃f�[�^�̌n��*/
	double y_;
	/*���Ԗڂ̏������̃J�E���g*/
	int i;
	i = 0;
	/*���ς��V�O���C�h�֐��ŕϊ��@�S�Ă̌n����V�O���C�h�֐��ŕϊ��������AR���f������T���v�����O����*/
	sig_mu = sig_env(mu);
	/*�����l���v�Z�@����+�덷*/
	*y = sig_mu + rnorm(0, sigma);
	/*�n��̍Ō���܂Ń��[�v*/
	while (i != T) {
		/*AR���f���ɏ]���Ď��̒l���v�Z����*/
		y_ = sig_mu + phi * (*y - sig_mu) + rnorm(0, sigma);
		/*��O�̒l���V�O���C�h�֐��ŕϊ�*/
		*y = sig(*y);
		/*�n��̃|�C���^������ɐi�߂�*/
		++y;
		/*��قǌv�Z�����l��y�ɑ��*/
		*y = y_;
		/*�J�E���g��i�߂�*/
		++i;
	}
	/*���[�v���I���������_�ōŌ�̈���ϊ�����Ă��Ȃ��͂��Ȃ̂ŕϊ�*/
	*y = sig(*y);
	return 0;
}

/*���p�@�ɂ����DR�𔭐�������*/
double reject_sample(double pd, double rho) {
	int i;
	double y;
	double prob[10000];
	double max_density = 0;
	double density_range;

	/*���݂̃p�����[�^�ł̊m�����x�̒��_�ƁA���x��0�ȏ�̓_�͈̔͂����߂�*/
	for (i = 1; i < 9999; i++) {
		prob[i] = g_DR_fn(i / 10000.0, pd, rho);
		if (prob[i] > max_density) {
			max_density = prob[i];
		}
		if (prob[i] > 0) {
			density_range = i / 10000.0;
		}
	}
	
	/*���p�@��DR����*/
	while (1) {
		y = Uniform() * density_range;
		if (g_DR_fn(y, pd, rho) > max_density*Uniform()) {
			return y;
		}
	}
}

/*�T���v�����O DynamicDefaultRate*/
double r_DDR(double X_t, double q_qnorm, double rho, double beta) {
	return (q_qnorm - sqrt(rho)*sqrt(beta)*X_t) / sqrt(1 - rho) - sqrt(rho)*sqrt(1 - beta) / sqrt(1 - rho) * rnorm(0, 1);
}

