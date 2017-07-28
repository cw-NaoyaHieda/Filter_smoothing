#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include "myfunc.h"
#include "sampling_DR.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359	
#define beta 0.6
#define q_qnorm -2.053749 //q�ɒ������Ƃ��ɁA��0.02
#define rho 0.05
#define X_0 -2.5
#define a_grad 0.0001
#define b_grad 0.5

std::mt19937 mt(100);
std::uniform_real_distribution<double> r_rand(0.0,1.0);


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
void particle_filter(std::vector<double>& DR,double beta_est,double q_qnorm_est,double rho_est,double X_0_est,int N,int T,
	std::vector<std::vector<double>>& state_X_all, std::vector<std::vector<double>>& weight_state_all, std::vector<double>& state_X_mean) {
	int n;
	int t;
	double pred_X_mean_tmp;
	double state_X_mean_tmp;
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
	std::vector<double> post_X(N),post_weight(N);

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
			resample_numbers[n] = resample(cumsum_weight,N, (r_rand(mt) + n - 1) / N);
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
    #pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp)
	for (n = 0; n < N; n++) {
		pred_X_all[0][n] = pred_X[n];
		state_X_all[0][n] = pred_X[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		if (check_resample == 0) {
			weight_state_all[0][n] = weight[n];
		}
		else {
			weight_state_all[0][n] = 1.0 / N;
		}
		pred_X_mean_tmp += pred_X_all[0][n] * 1.0 / N;
		state_X_mean_tmp += state_X_all[0][n] * weight_state_all[0][n];
	}

	pred_X_mean[0] = pred_X_mean_tmp;
	state_X_mean[0] = state_X_mean_tmp;
	/*��������͌J��Ԃ�����*/
	for (t = 2; t < T; t++) {
		/*����O��(����Ӗ����O)���ʎ擾*/
        #pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_X[n] = state_X_all[t - 2][n];
			post_weight[n] = weight_state_all[t - 2][n];
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
				resample_numbers[n] = resample(cumsum_weight,N,r_rand(mt));
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
        #pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp)
		for (n = 0; n < N; n++) {
			pred_X_all[t - 1][n] = pred_X[n];
			state_X_all[t - 1][n] = pred_X[resample_numbers[n]];
			weight_all[t - 1][n] = weight[n];
			if (check_resample == 0) {
				weight_state_all[t - 1][n] = weight[n];
			}
			else {
				weight_state_all[t - 1][n] = 1.0 / N;
			}
			pred_X_mean_tmp += pred_X_all[t - 1][n] * weight_state_all[t - 2][n];
			state_X_mean_tmp += state_X_all[t - 1][n] * weight_state_all[t - 1][n];
		}
		pred_X_mean[t - 1] = pred_X_mean_tmp;
		state_X_mean[t - 1] = state_X_mean_tmp;
	}
}

/*������*/
void particle_smoother(int T,int N, std::vector<std::vector<double>>& weight_state_all, std::vector<std::vector<double>>& state_X_all,double beta_est,
	std::vector<std::vector<double>>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<double>& state_X_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight,resample_check_weight,bunsi_sum,bunbo_sum, state_X_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N), resample_numbers(N);
	/*T���_��weight�͕ς��Ȃ��̂ł��̂܂ܑ��*/
	state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 2][n] = weight_state_all[T - 2][n];
		state_X_all_bffs[T - 2][n] = state_X_all[T - 2][n];
		state_X_all_bffs_mean_tmp += state_X_all_bffs[T - 2][n] * weight_state_all_bffs[T - 2][n];
	}
	state_X_all_bffs_mean[T - 2] = state_X_all_bffs_mean_tmp;
	for (t = T - 3; t > -1; t--) {
		sum_weight = 0;
		resample_check_weight = 0;
		for (n = 0; n < N; n++) {
			bunsi_sum = 0;
			bunbo_sum = 0;
#pragma omp parallel for reduction(+:bunsi_sum) reduction(+:bunbo_sum)
			for (n2 = 0; n2 < N; n2++) {
				/*���q�v�Z*/
				bunsi[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(state_X_all_bffs[t + 1][n2],
						sqrt(beta_est) *  state_X_all[t][n],
						sqrt(1 - beta_est));
				bunsi_sum += bunsi[n][n2];
				/*����v�Z*/
				bunbo[n][n2] = weight_state_all[t][n2] *
					dnorm(state_X_all_bffs[t + 1][n],
						sqrt(beta_est) * state_X_all[t][n2],
						sqrt(1 - beta_est));
				bunbo_sum += bunbo[n][n2];
			}
			weight_state_all_bffs[t][n] = weight_state_all[t][n] * bunsi_sum / bunbo_sum;
			sum_weight += weight_state_all_bffs[t][n];
		}
		if (t == 0) {
			sum_weight;
		}
		/*���K���Ɨݐϑ��Ζޓx�̌v�Z*/
		for (n = 0; n < N; n++) {
			weight_state_all_bffs[t][n] = weight_state_all_bffs[t][n] / sum_weight;
			resample_check_weight += pow(weight_state_all_bffs[t][n], 2);
			if (n != 0) {
				cumsum_weight[n] = weight_state_all_bffs[t][n] + cumsum_weight[n - 1];
			}
			else {
				cumsum_weight[n] = weight_state_all_bffs[t][n];
			}

		}

		/*���T���v�����O���K�v���ǂ������f���������ŕK�v�Ȃ烊�T���v�����O �K�v�Ȃ��ꍇ�͏��Ԃɐ���������*/
		if (1 / resample_check_weight < N / 10) {
			for (n = 0; n < N; n++) {
				resample_numbers[n] = resample(cumsum_weight,N, (r_rand(mt) + n - 1) / N);
			}
			check_resample = 1;
		}
		else {
			for (n = 0; n < N; n++) {
				resample_numbers[n] = n;
			}
			check_resample = 0;
		}
		/*���T���v�����O�̕K�v���ɉ����Č��ʂ̊i�[*/
		state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			if (check_resample == 0) {
				state_X_all_bffs[t][n] = state_X_all[t][n];
				state_X_all_bffs_mean_tmp += state_X_all[t][n] * weight_state_all_bffs[t][n];
			}
			else {
				state_X_all_bffs[t][n] = state_X_all[t][resample_numbers[n]];
				weight_state_all_bffs[t][n] = 1.0 / N;
				state_X_all_bffs_mean_tmp += state_X_all[t][n] * weight_state_all_bffs[t][n];
			}
		}

		state_X_all_bffs_mean[t] = state_X_all_bffs_mean_tmp;

	}

}

/*EM�A���S���Y���ōő剻��������*/
double Q(std::vector<std::vector<double>>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs,double beta_est,double rho_est, double q_qnorm_est,double X_0_est,
	std::vector<double>& DR, int T, int N) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;
	int t, n, n2;
#pragma omp parallel for reduction(+:Q_state) reduction(+:Q_obeserve)
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			for (n2 = 0; n2 < N; n2++) {
				Q_state += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * //weight
					log(
						dnorm(state_X_all_bffs[t][n], sqrt(beta_est)*state_X_all_bffs[t - 1][n2], sqrt(1 - beta_est))//X�̑J�ڊm��
					);
			}
			Q_obeserve += weight_state_all_bffs[t][n] *//weight
				log(
					g_DR_dinamic(DR[t], state_X_all_bffs[t - 1][n], q_qnorm_est, beta_est, rho_est)//�ϑ��̊m��
				);
		}
	}
#pragma omp parallel for reduction(+:first_state)
	for (n = 0; n < N; n++) {
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				dnorm(state_X_all_bffs[0][n], sqrt(beta_est) * X_0_est, sqrt(1 - beta_est))//�������z����̊m��
			);
	}
	return Q_state + Q_obeserve + first_state;
}

/*Q�̍ŋ}�~���@*/
void Q_grad(int grad_stop_check, std::vector<std::vector<double >>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs, double beta_est, double rho_est, double q_qnorm_est, double X_0_est,
	std::vector<double>& DR, int T, int N,
	double beta_grad,double rho_grad,double q_qnorm_grad,double X_0_grad) {
	int t, n, n2, l;
	double Now_Q, q_qnorm_est_tmp, beta_est_tmp, rho_est_tmp, X_0_est_tmp, sig_beta_est, sig_rho_est, sig_beta_est_tmp, sig_rho_est_tmp;
	Now_Q = Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est,
		DR, T, N);
	beta_est_tmp = beta_est;
	rho_est_tmp = rho_est;
	q_qnorm_est_tmp = q_qnorm_est;
	X_0_est_tmp = X_0_est;
	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	sig_beta_est = sig_env(beta_est);
	sig_rho_est = sig_env(rho_est);
	sig_beta_est_tmp = sig_beta_est;
	sig_rho_est_tmp = sig_rho_est;

	beta_grad = 0;
	rho_grad = 0;
	q_qnorm_grad = 0;
	X_0_grad = 0;
#pragma omp parallel for reduction(+:beta_grad) reduction(+:rho_grad) reduction(+:q_qnorm_grad)
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			for (n2 = 0; n2 < N; n2++) {
				//beta �����ϐ��̎��ɂ��āAbeta���V�O���C�h�֐��ŕϊ������l�̔���
				beta_grad += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * exp(sig_beta_est) / 2 * (
					-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2] + pow(state_X_all_bffs[t - 1][n2], 2))) -
					((-exp(-sig_beta_est) *pow(state_X_all_bffs[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2])) +
					1 / (1 + exp(sig_beta_est))
					);
			}

			//���͊ϑ��ϐ��ɂ���
			beta_grad += weight_state_all_bffs[t - 1][n] * (
				exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
				(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) -
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(-(exp(-sig_beta_est + sig_rho_est) / (1 + exp(-sig_beta_est)))*pow(state_X_all_bffs[t - 1][n], 2) + sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2)*q_qnorm_est*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2) * state_X_all_bffs[t - 1][n])
				);

			//�Ō�͏����_����̔����ɂ���
			beta_grad += weight_state_all_bffs[0][n] * exp(sig_beta_est) / 2 * (
				-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[0][n] * X_0_est + pow(X_0_est, 2))) -
				((-exp(-sig_beta_est) * pow(state_X_all_bffs[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] * X_0_est)) +
				1 / (1 + exp(sig_beta_est))
				);

			//rho �ϑ��ϐ��ɂ��� rho���V�O���C�h�֐��ŕϊ������l�̔���
			rho_grad += weight_state_all_bffs[t - 1][n] * (
				-1 / 2 +
				((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) -
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(-(exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - (exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*state_X_all_bffs[t - 1][n]) - DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))
				);




			//q_qnorm �ϑ��ϐ��ɂ���
			q_qnorm_grad += weight_state_all_bffs[t - 1][n] * (
				(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
				((1 + exp(sig_rho_est))*q_qnorm_est - sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n] - DR[t] * sqrt(1 + exp(sig_rho_est)))
				);
		}
	}
#pragma omp parallel for reduction(+:X_0_grad)
	for (n = 0; n < N; n++) {
		//X_0 �����ϐ��ɂ���
		X_0_grad += weight_state_all_bffs[0][n] * (
			exp(sig_beta_est) * (sqrt(1 - exp(-sig_beta_est))*state_X_all_bffs[0][n] - X_0_est)
			);
	}
}

void Q_grad_calc(std::vector<std::vector<double >>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs,
	double& beta_est, double& rho_est, double& q_qnorm_est, double& X_0_est,
	std::vector<double>& DR, int T, int N,
	double& beta_grad, double& rho_grad, double& q_qnorm_grad, double& X_0_grad) {
	int t, n, n2, l, choice;
	double Now_Q, q_qnorm_est_tmp, beta_est_tmp, rho_est_tmp, X_0_est_tmp, sig_beta_est, sig_rho_est, sig_beta_est_tmp, sig_rho_est_tmp;
	double beta_grad_tmp, rho_grad_tmp, q_qnorm_grad_tmp, X_0_grad_tmp;
	Now_Q = Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est,
		DR, T, N);
	beta_est_tmp = beta_est;
	rho_est_tmp = rho_est;
	q_qnorm_est_tmp = q_qnorm_est;
	X_0_est_tmp = X_0_est;
	/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
	sig_beta_est = sig_env(beta_est);
	sig_rho_est = sig_env(rho_est);
	sig_beta_est_tmp = sig_beta_est;
	sig_rho_est_tmp = sig_rho_est;

	beta_grad_tmp = 0;
	rho_grad_tmp = 0;
	q_qnorm_grad_tmp = 0;
	X_0_grad_tmp = 0;
#pragma omp parallel for reduction(+:beta_grad_tmp) reduction(+:rho_grad_tmp) reduction(+:q_qnorm_grad_tmp)
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			for (n2 = 0; n2 < N; n2++) {
				//beta �����ϐ��̎��ɂ��āAbeta���V�O���C�h�֐��ŕϊ������l�̔���
				beta_grad_tmp += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * exp(sig_beta_est) / 2 * (
					-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2] + pow(state_X_all_bffs[t - 1][n2], 2))) -
					((-exp(-sig_beta_est) *pow(state_X_all_bffs[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2])) +
					1 / (1 + exp(sig_beta_est))
					);
			}

			//���͊ϑ��ϐ��ɂ���
			beta_grad_tmp += weight_state_all_bffs[t - 1][n] * (
				exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
				(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(-sig_beta_est + sig_rho_est) / pow((1 + exp(-sig_beta_est)), 2)*pow(state_X_all_bffs[t - 1][n], 2) - sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2)*q_qnorm_est*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2) * state_X_all_bffs[t - 1][n]))
				);

			//�Ō�͏����_����̔����ɂ���
			beta_grad_tmp += weight_state_all_bffs[0][n] * exp(sig_beta_est) / 2 * (
				-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[0][n] * X_0_est + pow(X_0_est, 2))) -
				((-exp(-sig_beta_est) * pow(state_X_all_bffs[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] * X_0_est)) +
				1 / (1 + exp(sig_beta_est))
				);

			//rho �ϑ��ϐ��ɂ��� rho���V�O���C�h�֐��ŕϊ������l�̔���
			rho_grad_tmp += weight_state_all_bffs[t - 1][n] * (
				-1 / 2 +
				((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est * state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - (exp(sig_rho_est) + 2 * exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*state_X_all_bffs[t - 1][n]) - DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))
				);




			//q_qnorm �ϑ��ϐ��ɂ���
			q_qnorm_grad_tmp += weight_state_all_bffs[t - 1][n] * (
				(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
				(-(1 + exp(sig_rho_est))*q_qnorm_est + sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(1 + exp(sig_rho_est)))
				);
		}
	}
#pragma omp parallel for reduction(+:X_0_grad_tmp)
	for (n = 0; n < N; n++) {
		//X_0 �����ϐ��ɂ���
		X_0_grad_tmp += weight_state_all_bffs[0][n] * (
			exp(sig_beta_est) * (sqrt(1 - exp(-sig_beta_est))*state_X_all_bffs[0][n] - X_0_est)
			);
	}
	
	beta_grad = beta_grad_tmp;
	rho_grad = rho_grad_tmp;
	q_qnorm_grad = q_qnorm_grad_tmp;
	X_0_grad = X_0_grad_tmp;
}

int main(void) {
	int n, t, i, j;
	int N = 1000;
	int T = 100;
	int I = 100;
	int J = 5;
	double beta_est;
	double rho_est;
	double q_qnorm_est;
	double X_0_est;
	/*�t�B���^�����O�̌��ʊi�[*/
	std::vector<std::vector<double> > filter_X(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
	std::vector<double> filter_X_mean(T);
	/*�������̌��ʊi�[*/
	std::vector<std::vector<double> > smoother_X(T, std::vector<double>(N));
	std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));
	std::vector<double> smoother_X_mean(T);

	/*Answer�i�[*/
	std::vector<double> X(T);
	std::vector<double> DR(T);

	std::vector<double> grad(4);
	double beta_grad = 0, rho_grad = 0, q_qnorm_grad = 0, X_0_grad = 0;
	
	/*X�����f���ɏ]���ăV�~�����[�V�����p�ɃT���v�����O�A������DR���T���v�����O ���_t��DR�͎��_t-1��X���p�����[�^�ɂ����K���z�ɏ]���̂ŁA��������_�ɒ���*/
	X[0] = sqrt(beta)*X_0 + sqrt(1 - beta) * rnorm(0, 1);
	DR[0] = -100;
	for (t = 1; t < T; t++) {
		X[t] = sqrt(beta)*X[t - 1] + sqrt(1 - beta) * rnorm(0, 1);
		DR[t] = r_DDR(X[t - 1], q_qnorm, rho, beta);
	}

	beta_est = beta;
	rho_est = rho;
	q_qnorm_est = q_qnorm;
	X_0_est = X_0;
	
	
	
	
	FILE *fp;
	if (fopen_s(&fp, "plot_Q_beta_rho.csv", "w") != 0) {
		return 0;
	}

	/*
	Q_grad_calc(smoother_X, smoother_weight, beta_est, rho_est, q_qnorm_est, X_0_est, DR, T, N, beta_grad, rho_grad, q_qnorm_grad, X_0_grad);
	for (i = 1; i < I; i++) {
		fprintf(fp, "%d,%f\n", t, Q(smoother_X, smoother_weight,
			beta_est + pow(0.5, i) * beta_grad, rho_est + pow(0.5, i) * rho_grad,
			q_qnorm_est + pow(0.5, i) * q_qnorm_grad, X_0_est + pow(0.5, i) * X_0_grad, DR, T, N));
	}
	fclose(fp);
	*/


	particle_filter(DR, beta_est, q_qnorm_est, rho_est, X_0_est, N, T, filter_X, filter_weight, filter_X_mean);
	particle_smoother(T, N, filter_weight, filter_X, beta_est, smoother_X, smoother_weight, smoother_X_mean);
	for (i = 1; i < I; i++) {
			fprintf(fp, "%f,%f,%f\n", i / double(I) - 0.0001, j / double(I) - 0.0001,
				Q(smoother_X, smoother_weight, i / double(I) - 0.0001, j / double(I) - 0.0001, q_qnorm_est, X_0_est, DR, T, N));
	}
	
	fclose(fp);
	
	FILE *gp, *gp2, *gp3;
	gp = _popen(GNUPLOT_PATH, "w");

	fprintf(gp, "reset\n");
	fprintf(gp, "set datafile separator ','\n");
	fprintf(gp, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp, "set border lc rgb 'white'\n");
	fprintf(gp, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp, "set key textcolor rgb 'white'\n");
	fprintf(gp, "plot 'plot_Q.csv' using 1:2 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'beta_grad'\n");
	fflush(gp);
	fprintf(gp, "replot 'plot_Q.csv' using 1:3 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'rho_grad'\n");
	fflush(gp);
	
	gp2 = _popen(GNUPLOT_PATH, "w");
	fprintf(gp2, "reset\n");
	fprintf(gp2, "set datafile separator ','\n");
	fprintf(gp2, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp2, "set border lc rgb 'white'\n");
	fprintf(gp2, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp2, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp2, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp2, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp2, "set key textcolor rgb 'white'\n");
	fprintf(gp2, "plot 'plot_Q.csv' using 6:4 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'q_qnorm_grad'\n");
	fflush(gp2);
	
	gp3 = _popen(GNUPLOT_PATH, "w");
	fprintf(gp3, "reset\n");
	fprintf(gp3, "set datafile separator ','\n");
	fprintf(gp3, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp3, "set border lc rgb 'white'\n");
	fprintf(gp3, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp3, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp3, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp3, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp3, "set key textcolor rgb 'white'\n");
	fprintf(gp3, "plot 'plot_Q.csv' using 6:5 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'X_0_grad'\n");
	fflush(gp3);

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