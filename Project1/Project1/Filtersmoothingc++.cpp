#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <float.h>
#include <random>
#include <algorithm>
#include "myfunc.h"
#include "sampling_DR.h"
#include "lbfgs.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359
#define pd_phi 0.9
#define pd_mu 0.05
#define pd_sd 0.1
#define pd_0 0.05
#define rho 0.1
#define a_grad 0.0001
#define b_grad 0.5
#define N 1000
#define T 100
/*�t�B���^�����O�̌��ʊi�[*/

std::mt19937 mt(120);
std::uniform_real_distribution<double> r_rand(0.0, 1.0);
// ����0.0�A�W���΍�1.0�ŕ��z������
std::normal_distribution<> dist(0.0, 1.0);


std::vector<std::vector<double> > filter_pd(T, std::vector<double>(N));
std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
/*�������̌��ʊi�[*/
std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));
/*Q�̌v�Z�̂��߂�wight*/
std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));
/*Answer�i�[*/
std::vector<double> pd(T);
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
void particle_filter(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,std::vector<double>& state_pd_mean) {
	int n;
	int t;
	double pred_pd_mean_tmp;
	double state_pd_mean_tmp;
	/*���_t�̗\���l�i�[*/
	std::vector<double> pred_pd(N), weight(N); //X��Particle weight

															/*�r���̏����p�ϐ�*/
	double sum_weight, resample_check_weight; //���K�����q(weight�̍��v) ���T���v�����O�̔��f�(���K���ޓx�̓��̍��v)
	std::vector<double> cumsum_weight(N); //�ݐϖޓx�@���K��������Ōv�Z��������
	std::vector<int> resample_numbers(N); //���T���v�����O�������ʂ̔ԍ�
	int check_resample; //���T���v�����O�������ǂ����̕ϐ� 0�Ȃ炵�ĂȂ��A1�Ȃ炵�Ă�

						/*�S���Ԃ̐���l�i�[*/
	std::vector<std::vector<double>> pred_pd_all(T, std::vector<double>(N)); //X��Particle  �\���l X��Particle �t�B���^�����O
	std::vector<double> pred_pd_mean(T); //X�̗\���l,X�̃t�B���^�����O����
	std::vector<std::vector<double>> weight_all(T, std::vector<double>(N)); // weight �\���l weight �t�B���^�����O

																			/*����O�̌���*/
	std::vector<double> post_pd(N), post_weight(N);

	/*���_1�ł̃t�B���^�����O�J�n*/
	/*�������z����̃T���v�����O���A���̂܂܎��_1�̃T���v�����O*/
#pragma omp parallel for
	for (n = 0; n < N; n++) {
		/*�������z����@���_0�ƍl����*/
		pred_pd[n] = sig(pd_mu_est + pd_phi_est*(pd_0_est - pd_mu_est) + pd_sd_est * dist(mt));
	}

	/*�d�݂̌v�Z*/
	sum_weight = 0;
	resample_check_weight = 0;
#pragma omp parallel for reduction(+:sum_weight)
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_fn(DR[0], pred_pd[n], rho_est);
		sum_weight += weight[n];
	}
	/*�d�݂𐳋K�����Ȃ���A���T���v�����O���f�p�ϐ��̌v�Z�Ɨݐϖޓx�̌v�Z*/
	for (n = 0; n < N; n++) {
		weight[n] = weight[n] / sum_weight;
		resample_check_weight += pow(weight[n], 2.0);
		if (n != 0) {
			cumsum_weight[n] = weight[n] + cumsum_weight[n - 1];
		}
		else {
			cumsum_weight[n] = weight[n];
		}
	}

	/*���T���v�����O���K�v���ǂ������f���������ŕK�v�Ȃ烊�T���v�����O �K�v�Ȃ��ꍇ�͏��Ԃɐ���������*/
	if (1 / resample_check_weight < N / 100) {
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
	pred_pd_mean_tmp = 0;
	state_pd_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_pd_mean_tmp) reduction(+:state_pd_mean_tmp)
	for (n = 0; n < N; n++) {
		pred_pd_all[0][n] = pred_pd[n];
		filter_pd[0][n] = pred_pd[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		if (check_resample == 0) {
			filter_weight[0][n] = weight[n];
		}
		else {
			filter_weight[0][n] = 1.0 / N;
		}
		pred_pd_mean_tmp += pred_pd_all[0][n] * 1.0 / N;
		state_pd_mean_tmp += filter_pd[0][n] * filter_weight[0][n];
	}

	pred_pd_mean[0] = pred_pd_mean_tmp;
	state_pd_mean[0] = state_pd_mean_tmp;
	/*��������͌J��Ԃ�����*/
	for (t = 1; t < T; t++) {
		/*����O��(����Ӗ����O)���ʎ擾*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_pd[n] = filter_pd[t - 1][n];
			post_weight[n] = filter_weight[t - 1][n];
		}
		/*���_t�̃T���v�����O*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			pred_pd[n] = sig(pd_mu_est + pd_phi_est*(sig_env(post_pd[n]) - pd_mu_est) + pd_sd_est * dist(mt));
		}

		/*�d�݂̌v�Z*/
		sum_weight = 0.0;
		resample_check_weight = 0.0;
#pragma omp parallel for reduction(+:sum_weight)
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_fn(DR[t], pred_pd[n], rho_est) * post_weight[n];
			sum_weight += weight[n];
		}

		/*�d�݂𐳋K�����Ȃ���A���T���v�����O���f�p�ϐ��̌v�Z�Ɨݐϖޓx�̌v�Z*/
		for (n = 0; n < N; n++) {
			weight[n] = weight[n] / sum_weight;
			resample_check_weight += pow(weight[n], 2.0);
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
		pred_pd_mean_tmp = 0;
		state_pd_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_pd_mean_tmp) reduction(+:state_pd_mean_tmp)
		for (n = 0; n < N; n++) {
			pred_pd_all[t][n] = pred_pd[n];
			filter_pd[t][n] = pred_pd[resample_numbers[n]];
			weight_all[t][n] = weight[n];
			if (check_resample == 0) {
				filter_weight[t][n] = weight[n];
			}
			else {
				filter_weight[t][n] = 1.0 / N;
			}
			pred_pd_mean_tmp += pred_pd_all[t][n] * filter_weight[t][n];
			state_pd_mean_tmp += filter_pd[t][n] * filter_weight[t][n];
		}
		pred_pd_mean[t] = pred_pd_mean_tmp;
		state_pd_mean[t] = state_pd_mean_tmp;
	}
}

/*������*/
void particle_smoother(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,std::vector<double>& state_pd_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight, bunsi_sum, bunbo_sum, state_pd_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T���_��weight�͕ς��Ȃ��̂ł��̂܂ܑ��*/
	state_pd_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		smoother_weight[T - 1][n] = filter_weight[T - 1][n];
		state_pd_all_bffs_mean_tmp += filter_pd[T - 1][n] * smoother_weight[T - 1][n];
	}
	state_pd_all_bffs_mean[T - 1] = state_pd_all_bffs_mean_tmp;
	for (t = T - 2; t > -1; t--) {
		sum_weight = 0;
		bunbo_sum = 0;
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:bunbo_sum)
			for (n2 = 0; n2 < N; n2++) {
				/*����v�Z*/
				bunbo[n][n2] = filter_weight[t][n2] *
					dnorm(sig_env(filter_pd[t + 1][n]),
						pd_mu_est + pd_phi_est*(sig_env(filter_pd[t][n2]) -pd_mu_est),
						pd_sd_est);
				bunbo_sum += bunbo[n][n2];
			}
		}

		sum_weight = 0;
		for (n = 0; n < N; n++) {
			bunsi_sum = 0;
			/*���q�v�Z*/
#pragma omp parallel for reduction(+:bunsi_sum)
			for (n2 = 0; n2 < N; n2++) {
				bunsi[n][n2] = smoother_weight[t + 1][n2] *
					dnorm(sig_env(filter_pd[t + 1][n2]),
						pd_mu_est + pd_phi_est*(sig_env(filter_pd[t][n]) - pd_mu_est),
						pd_sd_est);
				bunsi_sum += bunsi[n][n2];
			}
			smoother_weight[t][n] = filter_pd[t][n] * bunsi_sum / bunbo_sum;
			sum_weight += smoother_weight[t][n];
		}
		/*���K���Ɨݐϑ��Ζޓx�̌v�Z*/
		for (n = 0; n < N; n++) {
			smoother_weight[t][n] = smoother_weight[t][n] / sum_weight;
			if (n != 0) {
				cumsum_weight[n] = smoother_weight[t][n] + cumsum_weight[n - 1];
			}
			else {
				cumsum_weight[n] = smoother_weight[t][n];
			}
		}

		/*��������������l���v�Z*/
		state_pd_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			state_pd_all_bffs_mean_tmp += filter_pd[t][n] * smoother_weight[t][n];
		}
		state_pd_all_bffs_mean[t] = state_pd_all_bffs_mean_tmp;
	}
}

/*Q�̌v�Z�ɕK�v�ȐV����wight*/
void Q_weight_calc(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est) {
	int n, n2, t;
	double bunbo;
	for (t = T - 1; t > 0; t--) {
		for (n2 = 0; n2 < N; n2++) {
			bunbo = 0;
#pragma omp parallel for reduction(+:bunbo)
			for (n = 0; n < N; n++) {
				/*����v�Z*/
				bunbo += filter_weight[t - 1][n] * dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est);
			}
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				/*���q�v�Z�����*/
				Q_weight[t][n][n2] = filter_weight[t - 1][n] * smoother_weight[t][n2] *
					dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est) / bunbo;
				//printf("%f\n", Q_weight[t + 1][n][n2]);
			}
		}

	}

}

/*EM�A���S���Y���ōő剻��������*/
double Q(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;
	int t, n, n2;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:Q_state)
			for (n2 = 0; n2 < N; n2++) {
				Q_state += Q_weight[t][n2][n] * //weight
					log(
						std::max(dnorm(sig_env(filter_pd[t][n]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est), pd_sd_est), 0.000000000000001)//X�̑J�ڊm��
					);
			}
		}
	}
	for (t = 1; t < T; t++) {
#pragma omp parallel for reduction(+:Q_obeserve)
		for (n = 0; n < N; n++) {
			Q_obeserve += smoother_weight[t][n] *//weight
				log(
					std::max(g_DR_(DR[t], filter_pd[t][n], rho_est), 0.000000000000001)//�ϑ��̊m��
				);
		}
	}
#pragma omp parallel for reduction(+:first_state) reduction(+:Q_obeserve)
	for (n = 0; n < N; n++) {
		first_state += smoother_weight[0][n] *//weight
			log(
				dnorm(sig_env(filter_pd[0][n]), pd_mu_est + pd_phi_est*(pd_0_est - pd_mu_est), pd_sd_est)//�������z����̊m��
			);
		Q_obeserve += smoother_weight[0][n] *//weight
			log(
				g_DR_(DR[0], filter_pd[0][n], rho_est)//�ϑ��̊m��
			);
	}
	//printf("Q_State %f Q_Obeserve %f First %f\n", Q_state, Q_obeserve, first_state);
	return Q_state + Q_obeserve + first_state;
}

double phi_Q_grad(double pd_phi_est, double pd_mu_est, double pd_sd_est, double pd_0_est, double rho_est) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	log_pd_sd_est = log(pd_sd_est);
	sig_pd_phi_est = sig_phi_env(pd_phi_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
	sig_pd_phi_est_tmp = sig_pd_phi_est;
	sig_rho_est_tmp = sig_rho_est;

	phi_grad = 0;
	mu_grad = 0;
	sd_grad = 0;
	zero_grad = 0;
	rho_grad = 0;

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:phi_grad)
			for (n2 = 0; n2 < N; n2++) {
				//phi �����ϐ��̎��ɂ��āAphi���V�O���C�h�֐��ŕϊ������l�̔���
				phi_grad += Q_weight[t][n2][n] * (
					-1 / (2 * exp(log_pd_sd_est))*(
						-2 * sig_env(filter_pd[t][n]) * 2 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2)*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est) +
						2 * pd_mu_est * 2 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2)*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est) +
						(8 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 3) - 4 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2))*pow(sig_env(filter_pd[t - 1][n2]) - pd_mu_est, 2)
						)
					);
			}
		}
	}

#pragma omp parallel for reduction(+:phi_grad)
	for (n = 0; n < N; n++) {
		//phi �����_����̔����ɂ���
		phi_grad += smoother_weight[0][n] * (
			-1 / (2 * exp(log_pd_sd_est))*(
				-2 * sig_env(filter_pd[0][n]) * 2 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2)*(pd_0_est - pd_mu_est) +
				2 * pd_mu_est * 2 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2)*(pd_0_est - pd_mu_est) +
				(8 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 3) - 4 * exp(-sig_pd_phi_est) / pow(1 + exp(-sig_pd_phi_est), 2))*pow(pd_0_est - pd_mu_est, 2)
				)
			);
	}

	return phi_grad;

}

double sd_Q_grad(double pd_phi_est, double pd_mu_est, double pd_sd_est, double pd_0_est, double rho_est) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	log_pd_sd_est = log(pd_sd_est);
	sig_pd_phi_est = sig_phi_env(pd_phi_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
	sig_pd_phi_est_tmp = sig_pd_phi_est;
	sig_rho_est_tmp = sig_rho_est;

	phi_grad = 0;
	mu_grad = 0;
	sd_grad = 0;
	zero_grad = 0;
	rho_grad = 0;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:sd_grad)
			for (n2 = 0; n2 < N; n2++) {
				//sd �����ϐ��̎��ɂ��āAsd��ΐ��֐��ŕϊ������l�̔���
				sd_grad += Q_weight[t][n2][n] * (
					-1 + 1 / exp(2 * log_pd_sd_est)*pow(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)), 2.0)
					);
			}
		}
	}
#pragma omp parallel for reduction(+:sd_grad)
	for (n = 0; n < N; n++) {
		//sd �����_����̔����ɂ���
		sd_grad += smoother_weight[0][n] * (
			-1 + 1 / exp(2 * log_pd_sd_est)*pow(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)), 2.0)
			);
	}


	return sd_grad;


}

double mu_Q_grad(double pd_phi_est, double pd_mu_est, double pd_sd_est, double pd_0_est, double rho_est) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	log_pd_sd_est = log(pd_sd_est);
	sig_pd_phi_est = sig_phi_env(pd_phi_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
	sig_pd_phi_est_tmp = sig_pd_phi_est;
	sig_rho_est_tmp = sig_rho_est;

	phi_grad = 0;
	mu_grad = 0;
	sd_grad = 0;
	zero_grad = 0;
	rho_grad = 0;

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:mu_grad)
			for (n2 = 0; n2 < N; n2++) {
				//mu �����ϐ��̎��ɂ���
				mu_grad += Q_weight[t][n2][n] * (
					1 / exp(2 * log_pd_sd_est)*(1 - pd_phi_est)*(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)))
					);
			}
		}
	}

#pragma omp parallel for reduction(+:mu_grad)
	for (n = 0; n < N; n++) {
		//mu �����_����̔����ɂ���
		mu_grad += smoother_weight[0][n] * (
			1 / exp(2 * log_pd_sd_est)*(1 - pd_phi_est)*(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)))
			);
	}
	return mu_grad;

}

double zero_Q_grad(double pd_phi_est, double pd_mu_est, double pd_sd_est, double pd_0_est, double rho_est) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	log_pd_sd_est = log(pd_sd_est);
	sig_pd_phi_est = sig_phi_env(pd_phi_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
	sig_pd_phi_est_tmp = sig_pd_phi_est;
	sig_rho_est_tmp = sig_rho_est;

	phi_grad = 0;
	mu_grad = 0;
	sd_grad = 0;
	zero_grad = 0;
	rho_grad = 0;

#pragma omp parallel for reduction(+:zero_grad)
	for (n = 0; n < N; n++) {
		//pd_0 �����_����̔����ɂ���
		zero_grad += smoother_weight[0][n] * (
			1 / (exp(2 * log_pd_sd_est))*pd_phi_est*(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)))
			);
	}

	return zero_grad;

}

double rho_Q_grad(double pd_phi_est, double pd_mu_est, double pd_sd_est, double pd_0_est, double rho_est) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	log_pd_sd_est = log(pd_sd_est);
	sig_pd_phi_est = sig_phi_env(pd_phi_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
	sig_pd_phi_est_tmp = sig_pd_phi_est;
	sig_rho_est_tmp = sig_rho_est;

	phi_grad = 0;
	mu_grad = 0;
	sd_grad = 0;
	zero_grad = 0;
	rho_grad = 0;

	for (t = 0; t < T; t++) {
#pragma omp parallel for reduction(+:rho_grad)
		//rho �ϑ����ɂ���
		for (n = 0; n < N; n++) {
			rho_grad += smoother_weight[t][n] * (
				1.0 / 2.0 *(-1 -
				(-exp(-sig_rho_est)*(pow(qnorm(DR[t]), 2.0) + pow(qnorm(filter_pd[t][n]), 2.0)) +
					(exp(-sig_rho_est) + 2 * exp(-2 * sig_rho_est)) / sqrt(exp(-sig_rho_est) + exp(-2 * sig_rho_est))*qnorm(filter_pd[t][n])*qnorm(DR[t])
					)
					)
				);
		}
	}

	return rho_grad;
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
	fx = -Q(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
	
	g[0] = -phi_Q_grad(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
	g[1] = -mu_Q_grad(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
	g[2] = -sd_Q_grad(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
	g[3] = -zero_Q_grad(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
	g[4] = -rho_Q_grad(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
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
	printf("  fx = %f, phi = %f, mu = %f, sd = %f, X_0 = %f, rho = %f\n", fx, sig_phi(x[0]), sig(x[1]),exp(x[2]),sig(x[3]),sig(x[4]));
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}



int main(void) {
	int n, t, i, j, k, l;
	double pd_mu_est;
	double pd_phi_est;
	double pd_sd_est;
	double pd_0_est;
	double rho_est;
	/*�t�B���^�����O�̌��ʊi�[*/
	std::vector<double> filter_pd_mean(T);
	/*�������̌��ʊi�[*/
	std::vector<double> smoother_pd_mean(T);

	

	std::vector<double> grad(2);
	double pd_grad = 0;

	/*PD,rho,DR�𔭐�������*/
	pd[0] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd_0) - sig_env(pd_mu)) + pd_sd * dist(mt));
	DR[0] = reject_sample(pd[0], rho);
	for (t = 1; t < T; t++) {
		pd[t] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd[t - 1]) - sig_env(pd_mu)) + pd_sd * dist(mt));
		DR[t] = reject_sample(pd[t], rho);
	}
	

	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(5);
	lbfgs_parameter_t param;



	x[0] = sig_phi_env(r_rand(mt));
	x[1] = sig_env(r_rand(mt) / 10);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	x[2] = log(r_rand(mt) / 10);
	x[3] = sig_env(r_rand(mt) / 10);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	x[4] = sig_env(r_rand(mt) / 5);


	
	double ret;
	/*
	pd_mu_est = sig_env(r_rand(mt) / 10);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	pd_phi_est = r_rand(mt);
	pd_sd_est = r_rand(mt) / 10;
	pd_0_est = sig_env(r_rand(mt) / 10);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	rho_est = r_rand(mt) / 5;
	printf("First parameter  phi_est %f, mu_est %f, sd_est %f,rho_est %f,0_est %f \n\n",
		pd_phi_est, sig(pd_mu_est), pd_sd_est, rho_est, sig(pd_0_est));
		*/
	/*
	pd_mu_est = sig_env(pd_mu);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	pd_phi_est = pd_phi;
	pd_sd_est = pd_sd;
	pd_0_est = sig_env(pd_0);//EM�ł�₱�����̂ŁC�ŏ�����ϊ����Ă���
	rho_est = rho;
		*/
	
	int grad_stop_check = 1;
	


		particle_filter(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]), filter_pd_mean);
		particle_smoother(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]), smoother_pd_mean);
		Q_weight_calc(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
		lbfgs_parameter_init(&param);
		printf("start");
		lbfgs(5, x, &fx, evaluate, progress, NULL, &param);
		printf("phi = %f, mu = %f, sd = %f, X_0 = %f, rho = %f\n", sig_phi(x[0]), sig(x[1]), exp(x[2]), sig(x[3]), sig(x[4]));

	

	FILE *fp;
	if (fopen_s(&fp, "particle_hull.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f\n", t, filter_pd[t][n], filter_weight[t][n], N / 20 * filter_weight[t][n]);

		}
	}
	fclose(fp);

	if (fopen_s(&fp, "X_hull.csv", "w") != 0) {
		return 0;
	}
	for (t = 0; t < T - 1; t++) {
		fprintf(fp, "%d,%f,%f,%f,%f\n", t, pd[t],filter_pd_mean[t], smoother_pd_mean[t], DR[t]);
	}

	fclose(fp);

	

	FILE *gp, *gp2, *gp3;
	gp = _popen(GNUPLOT_PATH, "w");

	//fprintf(gp, "set term postscript eps color\n");
	//fprintf(gp, "set term pdfcairo enhanced size 12in, 9in\n");
	//fprintf(gp, "set output 'particle.pdf'\n");
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
	fprintf(gp, "plot 'particle_hull.csv' using 1:2:4:3 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp);
	fprintf(gp, "replot 'X_hull.csv' using 1:2 with lines linetype 1 lw 4 linecolor rgb '#ff0000 ' title 'Answer PD'\n");
	fflush(gp);
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "replot 'X_hull.csv' using 1:3 with lines linetype 1 lw 4 linecolor rgb '#ffff00 ' title 'Filter'\n");
	fflush(gp);
	fprintf(gp, "replot 'X_hull.csv' using 1:4 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
	fflush(gp);
	//fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'Predict'\n");
	//fflush(gp);



	gp2 = _popen(GNUPLOT_PATH, "w");
	//fprintf(gp2, "set term pdfcairo enhanced size 12in, 9in\n");
	//fprintf(gp2, "set output 'DR.pdf'\n");
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
	fprintf(gp2, "plot 'X_hull.csv' using 1:5 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'DR'\n");
	fflush(gp2);
	//fprintf(gp2, "replot 'X.csv' using 1:6 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'predict DR'\n");
	//fflush(gp2);


	system("pause");
	fprintf(gp, "exit\n");    // gnuplot�̏I��
	_pclose(gp);
	



	//particle_smoother();
	/*q();
	printf("Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n phi_pd_est %f,mean_pd_est %f,sd_sig_pd_est %f\n",
	Now_Q, phi_rho_est, mean_rho_est, sd_sig_rho_est, phi_pd_est, mean_pd_est, sd_sig_pd_est);

	if (1) {
	printf("Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n", Now_Q, phi_rho_est, mean_rho_est, sd_sig_rho_est);

	double pre_pd[T], pre_rho[T];
	double a, b;
	for (t = 0; t < T; t++) {
	a = 0;
	b = 0;
	for (n = 0; n < N; n++) {
	a += pred_pd_all[t][n] * weight_all[t][n];
	b += pred_rho_all[t][n] * weight_all[t][n];
	}
	pre_pd[t] = a;
	pre_rho[t] = b;
	}



	return 0;
	}
	*/
	//	return 0;
	//}



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

/*pd rho�̗\���l��answer�̃v���b�g
double pre_pd[T], pre_rho[T];
double a, b;
for (t = 0; t < T; t++) {
a = 0;
b = 0;
for (n = 0; n < N; n++) {
a += pred_pd_all[t][n] * weight_all[t][n];
b += pred_rho_all[t][n] * weight_all[t][n];
}
pre_pd[t] = a;
pre_rho[t] = b;
}

int i;
FILE *gp;
gp = _popen(GNUPLOT_PATH, "w");
fprintf(gp, "set xrange [0:%d]\n", T);
fprintf(gp, "set yrange [0:0.1]\n");
fprintf(gp, "plot '-' with lines linetype 1 title \"PD\",'-' with lines linetype 2 title \"pre_PD\"\n");
for (i = 1; i < 100; i++) {
fprintf(gp, "%f\t%f\n", i * 1.0, pd[i]);
}
fprintf(gp, "e\n");
for (i = 1; i < 100; i++) {
fprintf(gp, "%f\t%f\n", i * 1.0, pre_pd[i]);
}
fprintf(gp, "e\n");
fflush(gp);

fprintf(gp, "set xrange [0:%d]\n", T);
fprintf(gp, "set yrange [0:0.25]\n");
fprintf(gp, "plot '-' with lines linetype 1 title \"rho\", '-' with lines linetype 2 title \"pre_rho\"\n");
for (i = 1; i < 100; i++) {
fprintf(gp, "%f\t%f\n", i * 1.0, rho[i]);
}
fprintf(gp, "e\n");
for (i = 1; i < 100; i++) {
fprintf(gp, "%f\t%f\n", i * 1.0, pre_rho[i]);
}
fprintf(gp, "e\n");
fflush(gp);

fprintf(gp, "exit\n");	// gnuplot�̏I��
_pclose(gp);

return 0;
*/
