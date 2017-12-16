#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include "myfunc.h"
#include "sampling_DR.h"
#include "lbfgs.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359
#define beta 0.85
#define q_qnorm -2.053749 //q�ɒ������Ƃ��ɁA��0.02
#define rho 0.07
#define X_0 -2.5
#define a_grad 0.0001
#define b_grad 0.5
#include <fstream> //iostream�̃t�@�C�����o�͂��T�|�[�g
#include <iostream> //���o�̓��C�u����
#include <string>
#include <sstream>
#include <chrono>
#include <time.h>

#define T 100
#define N 1000
#define S 100

std::mt19937 mt(100);
std::uniform_real_distribution<double> r_rand(0.0, 1.0);
std::uniform_real_distribution<double> r_rand_choice(0.0, 4.0);
std::uniform_real_distribution<double> r_rand_X_0(-3.0, -1.0);
std::uniform_real_distribution<double> r_rand_q(-3.0, -1.0);
std::uniform_real_distribution<double> r_rand_q_new(-3.0, -3.0);

/*�t�B���^�����O�̌��ʊi�[*/
std::vector<std::vector<double> > filter_X(T, std::vector<double>(N));
std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));

/*�������̌��ʊi�[*/
std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));

/*Q�̌v�Z�̂��߂�wight*/
std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));

/*Answer�i�[*/
std::vector<double> X(T);
std::vector<double> DR(T);

/*���T���v�����O�֐�*/
int resample(std::vector<double>& cumsum_weight, int num_of_particle, double x) {
	int particle_number = 0;
	while (particle_number != num_of_particle) {
		if (cumsum_weight[particle_number] > x) {
			return particle_number;
		}
		++particle_number;
	}
	return num_of_particle;
}



/*�t�B���^�����O*/
void particle_filter(double beta_est, double q_qnorm_est, double rho_est, double X_0_est,std::vector<double>& state_X_mean, std::vector<double>& predict_Y_mean) {
	int n;
	int t;
	double pred_X_mean_tmp;
	double state_X_mean_tmp;
	double predict_Y_mean_tmp;
	/*���_t�̗\���l�i�[*/
	std::vector<double> pred_X(N), weight(N); //X��Particle weight

	/*�r���̏����p�ϐ�*/
	double sum_weight, resample_check_weight; //���K�����q(weight�̍��v) ���T���v�����O�̔��f�(���K���ޓx�̓��̍��v)
	std::vector<double> cumsum_weight(N); //�ݐϖޓx�@���K��������Ōv�Z��������
	std::vector<int> resample_numbers(N); //���T���v�����O�������ʂ̔ԍ�
	int check_resample; //���T���v�����O�������ǂ����̕ϐ� 0�Ȃ炵�ĂȂ��A1�Ȃ炵�Ă�

	/*�S���Ԃ̐���l�i�[*/
	std::vector<std::vector<double>> pred_X_all(T, std::vector<double>(N)); //X��Particle  �\���l X��Particle �t�B���^�����O
	std::vector<double> pred_X_mean(T); //X�̗\���l,X�̃t�B���^�����O����
	std::vector<std::vector<double>> weight_all(T, std::vector<double>(N)); // weight �\���l weight �t�B���^�����O

	/*����O�̌���*/
	std::vector<double> post_X(N), post_weight(N);

	/*���_1�ł̃t�B���^�����O�J�n*/
	/*�������z����̃T���v�����O���A���̂܂܎��_1�̃T���v�����O*/
#pragma omp parallel for
	for (n = 0; n < N; n++) {
		/*�������z����@���_0�ƍl����*/
		pred_X[n] = sqrt(beta_est)*X_0_est - sqrt(1 - beta_est) * rnorm(0, 1);
	}

	/*�d�݂̌v�Z*/
	sum_weight = 0;
	resample_check_weight = 0;
#pragma omp parallel for reduction(+:sum_weight)
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_dinamic(DR[1], pred_X[n], q_qnorm_est, beta_est, rho_est);
		sum_weight += weight[n];
	}
	/*�d�݂𐳋K�����Ȃ���A���T���v�����O���f�p�ϐ��̌v�Z�Ɨݐϖޓx�̌v�Z*/
	weight[0] = weight[0] / sum_weight;
	resample_check_weight += pow(weight[0], 2);
	cumsum_weight[0] = weight[0];
	for (n = 1; n < N; n++) {
		weight[n] = weight[n] / sum_weight;
		resample_check_weight += pow(weight[n], 2);
		cumsum_weight[n] = weight[n] + cumsum_weight[n - 1];
	}

	/*���T���v�����O���K�v���ǂ������f���������ŕK�v�Ȃ烊�T���v�����O �K�v�Ȃ��ꍇ�͏��Ԃɐ���������*/
	if (1 / resample_check_weight < N / 10) {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers[n] = resample(cumsum_weight, N, (r_rand(mt) + n - 1) / N);
		}
		check_resample = 1;
	}
	else {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers[n] = n;
		}
		check_resample = 0;
	}

	/*���ʂ̊i�[*/
	pred_X_mean_tmp = 0;
	state_X_mean_tmp = 0;
	predict_Y_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp) reduction(+:predict_Y_mean_tmp)
	for (n = 0; n < N; n++) {
		pred_X_all[0][n] = pred_X[n];
		filter_X[0][n] = pred_X[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		predict_Y_mean_tmp += weight_all[0][n] * r_DDR(filter_X[0][n], q_qnorm_est, rho_est, beta_est);
		if (check_resample == 0) {
			filter_weight[0][n] = weight[n];
		}
		else {
			filter_weight[0][n] = 1.0 / N;
		}
		pred_X_mean_tmp += pred_X_all[0][n] * 1.0 / N;
		state_X_mean_tmp += filter_X[0][n] * filter_weight[0][n];
	}
	predict_Y_mean[1] = predict_Y_mean_tmp;

	pred_X_mean[0] = pred_X_mean_tmp;
	state_X_mean[0] = state_X_mean_tmp;
	/*��������͌J��Ԃ�����*/
	for (t = 2; t < T; t++) {
		/*����O��(����Ӗ����O)���ʎ擾*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_X[n] = filter_X[t - 2][n];
			post_weight[n] = filter_weight[t - 2][n];
		}
		/*���_t�̃T���v�����O*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			pred_X[n] = sqrt(beta_est)*post_X[n] - sqrt(1 - beta_est)*rnorm(0, 1);
		}

		/*�d�݂̌v�Z*/
		sum_weight = 0.0;
		resample_check_weight = 0.0;
#pragma omp parallel for reduction(+:sum_weight)
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_dinamic(DR[t], pred_X[n], q_qnorm_est, beta_est, rho_est) * post_weight[n];
			sum_weight += weight[n];
		}

		/*�d�݂𐳋K�����Ȃ���A���T���v�����O���f�p�ϐ��̌v�Z�Ɨݐϖޓx�̌v�Z*/
		for (n = 0; n < N; n++) {
			weight[n] = weight[n] / sum_weight;
			resample_check_weight += pow(weight[n], 2);
			if (n != 0) {
				cumsum_weight[n] = weight[n] + cumsum_weight[n - 1];
			}
			else {
				cumsum_weight[n] = weight[n];
			}
		}

		/*���T���v�����O���K�v���ǂ������f���������ŕK�v�Ȃ烊�T���v�����O �K�v�Ȃ��ꍇ�͏��Ԃɐ���������*/
		if (1 / resample_check_weight < N / 10) {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers[n] = resample(cumsum_weight, N, r_rand(mt));
			}
			check_resample = 1;
		}
		else {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers[n] = n;
			}
			check_resample = 0;
		}

		/*���ʂ̊i�[*/
		pred_X_mean_tmp = 0;
		state_X_mean_tmp = 0;
		predict_Y_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp) reduction(+:predict_Y_mean_tmp)
		for (n = 0; n < N; n++) {
			pred_X_all[t - 1][n] = pred_X[n];
			filter_X[t - 1][n] = pred_X[resample_numbers[n]];
			weight_all[t - 1][n] = weight[n];
			predict_Y_mean_tmp += weight_all[t - 1][n] * r_DDR(filter_X[t - 1][n], q_qnorm_est, rho_est, beta_est);
			if (check_resample == 0) {
				filter_weight[t - 1][n] = weight[n];
			}
			else {
				filter_weight[t - 1][n] = 1.0 / N;
			}
			pred_X_mean_tmp += pred_X_all[t - 1][n] * filter_weight[t - 2][n];
			state_X_mean_tmp += filter_X[t - 1][n] * filter_weight[t - 1][n];
		}
		pred_X_mean[t - 1] = pred_X_mean_tmp;
		state_X_mean[t - 1] = state_X_mean_tmp;
		predict_Y_mean[t] = predict_Y_mean_tmp;
	}
}

/*������*/
void particle_smoother(double beta_est, std::vector<double>& state_X_all_bffs_mean) {
	int check_resample;
	double state_X_all_bffs_mean_tmp;
	//std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T���_��weight�͕ς��Ȃ��̂ł��̂܂ܑ��*/
	state_X_all_bffs_mean_tmp = 0;


#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
	for (int n = 0; n < N; n++) {
		smoother_weight[T - 2][n] = filter_weight[T - 2][n];
		state_X_all_bffs_mean_tmp += filter_X[T - 2][n] * smoother_weight[T - 2][n];
	}
	state_X_all_bffs_mean[T - 2] = state_X_all_bffs_mean_tmp;



for (int t = T - 3; t > -1; t--) {
		double sum_weight = 0;
		double bunbo_sum = 0;
		double bunsi_sum = 0;
		std::vector<double> bunsi(N);
		//std::vector<std::vector<double>> bunbo(N, std::vector<double>(N));
		/*����v�Z*/
#pragma omp parallel for reduction(+:bunbo_sum) reduction(+:bunsi_sum)
		for (int n = 0; n < N; n++) {
			bunsi_sum = 0;
			for (int n2 = 0; n2 < N; n2++) {
				bunbo_sum += filter_weight[t][n2] *
					dnorm(filter_X[t + 1][n],
						sqrt(beta_est) * filter_X[t][n2],
						sqrt(1 - beta_est));

				bunsi_sum += smoother_weight[t + 1][n2] *
					dnorm(filter_X[t + 1][n2],
						sqrt(beta_est) *  filter_X[t][n],
						sqrt(1 - beta_est));
			}
			bunsi[n] = bunsi_sum;
		}


#pragma omp parallel for reduction(+:sum_weight)
		for (int n = 0; n < N; n++) {
			smoother_weight[t][n] = filter_weight[t][n] * bunsi[n] / bunbo_sum;
			sum_weight += smoother_weight[t][n];
		}

		/*���K���Ɨݐϑ��Ζޓx�̌v�Z*/
		smoother_weight[t][0] = smoother_weight[t][0] / sum_weight;
		cumsum_weight[0] = smoother_weight[t][0];

		for (int n = 1; n < N; n++) {
			smoother_weight[t][n] = smoother_weight[t][n] / sum_weight;
			cumsum_weight[n] = smoother_weight[t][n] + cumsum_weight[n - 1];
		}

		/*��������������l���v�Z*/
		state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
		for (int n = 0; n < N; n++) {
			state_X_all_bffs_mean_tmp += filter_X[t][n] * smoother_weight[t][n];
		}

		state_X_all_bffs_mean[t] = state_X_all_bffs_mean_tmp;

	}

}
/*Q�̌v�Z�ɕK�v�ȐV����wight*/
void Q_weight_calc(double beta_est) {
#pragma omp parallel for
	for (int t = T - 3; t > -1; t--) {
		for (int n2 = 0; n2 < N; n2++) {
			double bunbo = 0;
			for (int n = 0; n < N; n++) {
				/*����v�Z*/
				bunbo += filter_weight[t][n] * dnorm(filter_X[t + 1][n2], sqrt(beta_est) * filter_X[t][n], sqrt(1 - beta_est));
			}
			for (int n = 0; n < N; n++) {
				/*���q�v�Z�����*/
				Q_weight[t + 1][n][n2] = filter_weight[t][n] * smoother_weight[t + 1][n2] *
					dnorm(filter_X[t + 1][n2], sqrt(beta_est) * filter_X[t][n], sqrt(1 - beta_est)) / bunbo;
			}
		}

	}

}


/*EM�A���S���Y���ōő剻��������*/
double Q(double beta_est, double q_qnorm_est, double rho_est, double X_0_est) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;

#pragma omp parallel for reduction(+:Q_state)
	for (int t = 1; t < T; t++) {
		for (int n = 0; n < N; n++) {
			for (int n2 = 0; n2 < N; n2++) {
				Q_state += Q_weight[t][n2][n] * //weight
					log(
						dnorm(filter_X[t][n], sqrt(beta_est)*filter_X[t - 1][n2], sqrt(1 - beta_est))//X�̑J�ڊm��
					);
			}
		}
	}

#pragma omp parallel for reduction(+:Q_obeserve)
	for (int t = 1; t < T; t++) {
		for (int n = 0; n < N; n++) {
			Q_obeserve += smoother_weight[t - 1][n] *//weight
				log(
					g_DR_dinamic(DR[t], filter_X[t - 1][n], q_qnorm_est, beta_est, rho_est)//�ϑ��̊m��
				);
		}
	}

#pragma omp parallel for reduction(+:first_state)
	for (int n = 0; n < N; n++) {
		first_state += smoother_weight[0][n] *//weight
			log(
				dnorm(filter_X[0][n], sqrt(beta_est) * X_0_est, sqrt(1 - beta_est))//�������z����̊m��
			);
	}
	return Q_state + Q_obeserve + first_state;
}

double Q_grad_beta(double beta_est, double q_qnorm_est, double rho_est, double X_0_est) {

	double beta_grad = 0;

	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	double sig_beta_est = sig_env(beta_est);
	double sig_rho_est = sig_env(rho_est);


#pragma omp parallel for reduction(+:beta_grad)
	for (int t = 1; t < T; t++) {
		for (int n = 0; n < N; n++) {
			for (int n2 = 0; n2 < N; n2++) {
				//beta �����ϐ��̎��ɂ��āAbeta���V�O���C�h�֐��ŕϊ������l�̔���
				beta_grad += Q_weight[t][n2][n] * (
					-exp(sig_beta_est) / 2 * (((1 + exp(-sig_beta_est))*pow(filter_X[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * filter_X[t][n] * filter_X[t - 1][n2] + pow(filter_X[t - 1][n2], 2))) -
					exp(sig_beta_est) / 2 * ((-exp(-sig_beta_est) *pow(filter_X[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*filter_X[t][n] * filter_X[t - 1][n2])) +
					exp(sig_beta_est) / (2 + 2 * exp(sig_beta_est))
					);
			}
			beta_grad += smoother_weight[t - 1][n] * (
				exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
				(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(filter_X[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*filter_X[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*filter_X[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(-sig_beta_est + sig_rho_est) / pow((1 + exp(-sig_beta_est)), 2)*pow(filter_X[t - 1][n], 2) - sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 1.5)*q_qnorm_est*filter_X[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 1.5) * filter_X[t - 1][n]))
				);
		}
	}
#pragma omp parallel for reduction(+:beta_grad)
	for (int n = 0; n < N; n++) {
		//�Ō�͏����_����̔����ɂ���
		beta_grad += smoother_weight[0][n] * (
			-exp(sig_beta_est) / 2 * ((1 + exp(-sig_beta_est))*pow(filter_X[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * filter_X[0][n] * X_0_est + pow(X_0_est, 2)) -
			exp(sig_beta_est) / 2 * ((-exp(-sig_beta_est) * pow(filter_X[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*filter_X[0][n] * X_0_est)) +
			exp(sig_beta_est) / (2 + 2 * exp(sig_beta_est))
			);
	}

	return beta_grad;
}

double Q_grad_rho(double beta_est, double q_qnorm_est, double rho_est, double X_0_est) {

	double rho_grad = 0;

	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	double sig_beta_est = sig_env(beta_est);
	double sig_rho_est = sig_env(rho_est);

	rho_grad = 0;

#pragma omp parallel for reduction(+:rho_grad)
	for (int t = 1; t < T; t++) {
		for (int n = 0; n < N; n++) {
			rho_grad += smoother_weight[t - 1][n] * (
				-1.0 / 2.0 +
				((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(filter_X[t - 1][n], 2) -
						2 * sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*q_qnorm_est * filter_X[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*filter_X[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(filter_X[t - 1][n], 2) -
				(exp(sig_rho_est) + 2 * exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*filter_X[t - 1][n]) -
					DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*filter_X[t - 1][n]))
				);
		}
	}

	return rho_grad;
}

double Q_grad_q_qnorm(double beta_est, double q_qnorm_est, double rho_est, double X_0_est) {

	double q_qnorm_grad = 0;

	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	double sig_beta_est = sig_env(beta_est);
	double sig_rho_est = sig_env(rho_est);

#pragma omp parallel for reduction(+:q_qnorm_grad)
	for (int t = 1; t < T; t++) {
		for (int n = 0; n < N; n++) {
			q_qnorm_grad += smoother_weight[t - 1][n] * (
				(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
				(-(1 + exp(sig_rho_est))*q_qnorm_est +
					sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*filter_X[t - 1][n] +
					DR[t] * sqrt(1 + exp(sig_rho_est)))
				);
		}
	}

	return q_qnorm_grad;
}

double Q_grad_X_0(double beta_est, double q_qnorm_est, double rho_est, double X_0_est) {

	double X_0_grad = 0;

	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	double sig_beta_est = sig_env(beta_est);
	double sig_rho_est = sig_env(rho_est);


#pragma omp parallel for reduction(+:X_0_grad)
	for (int n = 0; n < N; n++) {
		//X_0 �����ϐ��ɂ���
		X_0_grad += smoother_weight[0][n] * (
			exp(sig_beta_est) * (sqrt(1 + exp(-sig_beta_est))*filter_X[0][n] - X_0_est)
			);
	}

	return X_0_grad;
}



static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
)
{
	int i;
	lbfgsfloatval_t fx = 0.0;
	fx = -Q(sig(x[0]), x[1],sig(x[2]),X_0);
	g[0] = -Q_grad_beta(sig(x[0]), x[1],sig(x[2]),X_0);
	g[1] = -Q_grad_q_qnorm(sig(x[0]), x[1], sig(x[2]), X_0);
	g[2] = -Q_grad_rho(sig(x[0]), x[1], sig(x[2]), X_0);
	return fx;
}
static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
)
{
	printf("Iteration %d:\n", k);
	printf("  fx = %f, beta = %f, q = %f, rho = %f, X_0 = %f\n", fx, sig(x[0]), pnorm(x[1],0,1), sig(x[2]), X_0);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}



int main(void) {


	double calc_time[100] = {};
	int iterate_count[100] = {};
	char filepath[256];
	clock_t start = clock(); // �v���X�^�[�g������ۑ�
	int n,t,s;
	double beta_est_pre;
	double rho_est_pre;
	double q_qnorm_est_pre;
	double norm;
	/*�t�B���^�����O�̌��ʊi�[*/
	std::vector<double> filter_X_mean(T);
	std::vector<double> predict_Y_mean(T);
	/*�������̌��ʊi�[*/
	std::vector<double> smoother_X_mean(T);




	/*X�����f���ɏ]���ăV�~�����[�V�����p�ɃT���v�����O�A������DR���T���v�����O ���_t��DR�͎��_t-1��X���p�����[�^�ɂ����K���z�ɏ]���̂ŁA��������_�ɒ���*/


	/*
	beta_est = r_rand(mt);
	rho_est = r_rand(mt);
	q_qnorm_est = r_rand_parameter(mt);
	X_0_est = r_rand_parameter(mt);
	*/




	int grad_stop_check = 1;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(3);
	lbfgs_parameter_t param;

	FILE *fp,*fp2;
	if (fopen_s(&fp, "result/parameter.csv", "w") != 0) {
		return 0;
	}



	fprintf(fp, "number,Iteration,beta,q,rho\n");
	fprintf(fp, "-1,-1,%f,%f,%f,%f\n", beta, pnorm(q_qnorm, 0, 1), rho);


	

	for (s = 0; s < S; s++) {
		X[0] = sqrt(beta)*X_0 + sqrt(1 - beta) * rnorm(0, 1);
		DR[0] = -2;
		for (t = 1; t < T; t++) {
			X[t] = sqrt(beta)*X[t - 1] + sqrt(1 - beta) * rnorm(0, 1);
			DR[t] = r_DDR(X[t - 1], q_qnorm, rho, beta);
		}
		
		start = clock();

		x[0] = sig_env(r_rand(mt)); //beta
		x[1] = (r_rand_q(mt)); //q_qnorm
		x[2] = sig_env(r_rand(mt) / 5); //rho
		beta_est_pre = sig(x[0]);
		q_qnorm_est_pre = pnorm(x[1],0,1);
		rho_est_pre = sig(x[2]);
		printf("%d,0,%f,%f,%f,%f\n", s, sig(x[0]), pnorm(x[1],0,1), sig(x[2]));
		fprintf(fp, "%d,0,%f,%f,%f,%f\n",s,sig(x[0]), pnorm(x[1], 0, 1), sig(x[2]));




		grad_stop_check = 1;
		norm = 100;
		while (grad_stop_check < 50 && (norm > 0.001)) {
			particle_filter(sig(x[0]), x[1], sig(x[2]), X_0, filter_X_mean, predict_Y_mean);
			particle_smoother(sig(x[0]), smoother_X_mean);

			clock_t end = clock();      // �v���I��������ۑ�
			std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
			printf("smoothing end\n");





			Q_weight_calc(sig(x[0]));


			end = clock();      // �v���I��������ۑ�
			std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
			printf("Pair Wise Smoothing Weight Calc end\n");

			lbfgs_parameter_init(&param);
			lbfgs(3, x, &fx, evaluate, progress, NULL, &param);
			printf("%d,%d,%f,%f,%f,%f\n", s, grad_stop_check, sig(x[0]), pnorm(x[1], 0, 1), sig(x[2]));
			fprintf(fp, "%d,%d,%f,%f,%f,%f\n", s, grad_stop_check, sig(x[0]), pnorm(x[1], 0, 1), sig(x[2]));
			grad_stop_check += 1;
			norm = sqrt(pow(sig(x[0]) - beta_est_pre, 2) + pow(pnorm(x[1], 0, 1) - q_qnorm_est_pre, 2) + pow(sig(x[2]) - rho_est_pre, 2));
			beta_est_pre = sig(x[0]);
			q_qnorm_est_pre = pnorm(x[1], 0, 1);
			rho_est_pre = sig(x[2]);
			printf("norm = %f\n", norm);
		}

		sprintf_s(filepath, "result/X_path_%d.csv", s);
		if (fopen_s(&fp2, filepath, "w") != 0) {
			return 0;

		}
		for (t = 0; t < T - 1; t++) {
			fprintf(fp2, "%d, %f, %f, %f, %f\n", t, X[t], DR[t], filter_X_mean[t], smoother_X_mean[t]);
		}

		fclose(fp2);

		clock_t end = clock();      // �v���I��������ۑ�
		std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
		calc_time[s] = (double)(end - start) / CLOCKS_PER_SEC;
		iterate_count[s] = grad_stop_check;
	}
	fclose(fp);


	FILE *fp3;
	if (fopen_s(&fp3, "result/calc_time.csv", "w") != 0) {
		return 0;
	}
	fprintf(fp, "number,time,count\n");
	for (s = 0; s < S; s++) {
		fprintf(fp, "%d,%f,%d\n", s,calc_time[s],iterate_count[s]);
	}
	fclose(fp3);

	/*
	if (fopen_s(&fp, "particle.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f\n", t, filter_X[t][n], filter_weight[t][n], N / 20 * filter_weight[t][n]);

		}
	}
	fclose(fp);

	if (fopen_s(&fp, "X.csv", "w") != 0) {
		return 0;
	}
	for (t = 0; t < T - 1; t++) {
		fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, X[t], filter_X_mean[t], smoother_X_mean[t], pnorm(DR[t],0,1), predict_Y_mean[t]);
	}

	fclose(fp);
	FILE *gp, *gp2;
	gp = _popen(GNUPLOT_PATH, "w");

	//
	fprintf(gp, "set term postscript eps color\n");
	fprintf(gp, "set term pdfcairo enhanced size 12in, 9in\n");
	fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "reset\n");
	fprintf(gp, "set datafile separator ','\n");
	fprintf(gp, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp, "set border lc rgb 'white'\n");
	fprintf(gp, "set border lc rgb 'white'\n");
	fprintf(gp, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp, "set key textcolor rgb 'white'\n");
	fprintf(gp, "set size ratio 1/3\n");
	fprintf(gp, "plot 'particle.csv' using 1:2:4:3 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp);
	fprintf(gp, "replot 'X.csv' using 1:2 with lines linetype 1 lw 4 linecolor rgb '#ff0000 ' title 'Answer'\n");
	fflush(gp);
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "replot 'X.csv' using 1:3 with lines linetype 1 lw 4 linecolor rgb '#ffff00 ' title 'Filter'\n");
	fflush(gp);
	fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
	fflush(gp);
	//fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'Predict'\n");
	//fflush(gp);

	gp2 = _popen(GNUPLOT_PATH, "w");
	fprintf(gp2, "set term pdfcairo enhanced size 12in, 9in\n");
	fprintf(gp2, "set output 'DR.pdf'\n");
	fprintf(gp2, "reset\n");
	fprintf(gp2, "set datafile separator ','\n");
	fprintf(gp2, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp2, "set border lc rgb 'white'\n");
	fprintf(gp2, "set border lc rgb 'white'\n");
	fprintf(gp2, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp2, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp2, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp2, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp2, "set key textcolor rgb 'white'\n");
	fprintf(gp2, "set size ratio 1/3\n");
	fprintf(gp2, "set output 'DR.pdf'\n");
	fprintf(gp2, "plot 'X.csv' using 1:5 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'DR'\n");
	fflush(gp2);
	//fprintf(gp2, "replot 'X.csv' using 1:6 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'predict DR'\n");
	//fflush(gp2);


	system("pause");
	fprintf(gp, "exit\n");    // gnuplot�̏I��
	_pclose(gp);

	*/





	return 0;
}
