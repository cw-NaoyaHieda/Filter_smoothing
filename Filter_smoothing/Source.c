#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#include<omp.h>
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359	
#define T 100
#define N 1000
#define beta 0.75
#define q_qnorm -2.053749 //q�ɒ������Ƃ��ɁA��0.02
#define rho 0.05
#define X_0 -2.5
#define alpha_grad 0.001
#define beta_grad 0.5
#define rand_seed 1218


/*Answer�i�[*/
double X[T];
double DR[T];

/*�������z����̃T���v�����O�i�[*/
double first_X[N];

/*���_t�̗\���l�i�[*/
double pred_X[N]; //X��Particle
double weight[N]; // weight

/*�S���Ԃ̐���l�i�[�@�t�B���^�����O*/
double pred_X_all[T][N]; //X��Particle  �\���l
double state_X_all[T][N]; //X��Particle �t�B���^�����O
double pred_X_mean[T]; //X�̗\���l
double state_X_mean[T]; //X�̃t�B���^�����O����
double weight_all[T][N]; // weight �\���l
double weight_state_all[T][N]; // weight �t�B���^�����O

/*�S���Ԃ̐���l�i�[�@������*/
double state_X_all_bffs[T][N]; //X��Particle���̂���
double weight_state_all_bffs[T][N]; // weight ���T���v�����O��������
double state_X_all_bffs_mean[T];

/*�p�����[�^�p�ϐ�*/
double beta_est;
double q_qnorm_est;
double rho_est;
double X_0_est;


/*�r���̏����p�ϐ�*/
double sum_weight; //���K�����q(weight�̍��v)
double cumsum_weight[N]; //�ݐϖޓx�@���K��������Ōv�Z��������
double resample_check_weight; //���T���v�����O�̔��f��@���K���ޓx�̓��̍��v
int resample_numbers[N]; //���T���v�����O�������ʂ̔ԍ�
int check_resample; //���T���v�����O�������ǂ����̕ϐ� 0�Ȃ炵�ĂȂ��A1�Ȃ炵�Ă�
int re_n; //���T���v�����O�̎Q�Ɨp
double bunsi[N][N]; //�������̕��q
double bunbo[N][N]; //�������̕���
double bunsi_sum;
double bunbo_sum;
double state_X_all_bffs_mean_tmp;
/*����O�̌���*/
double post_X[N];
double post_weight[N];

/*EM�̌v�Z*/
double Q_state;
double Q_obeserve;
double first_observe;
double first_state;

/*EM���̍ŋ}�~���@*/
double q_qnorm_est_tmp;
double beta_est_tmp;
double rho_est_tmp;
double X_0_est_tmp;
double sig_beta_est;
double sig_rho_est;
double sig_beta_est_tmp;
double sig_rho_est_tmp;
double a = 0;
double b = 0;
double c = 0;
double d = 0;
double Now_Q;
int l;

/*time��Particle��for���p�ϐ�*/
int t;
int n;
int n2;

/*�ŋ}�~���@�̒�~*/
int grad_stop_check;



/*�t�B���^�����O*/
int particle_filter() {
	/*���_1�ł̃t�B���^�����O�J�n*/
	/*�������z����̃T���v�����O���A���̂܂܎��_1�̃T���v�����O*/
	for (n = 0; n < N; n++) {
		/*�������z����@���_0�ƍl����*/
		pred_X[n] = sqrt(beta_est)*X_0_est - sqrt(1 - beta_est) * rnorm(0, 1);
	}

	/*�d�݂̌v�Z*/
	sum_weight = 0;
	resample_check_weight = 0;
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_dinamic(DR[1],pred_X[n],q_qnorm_est,beta_est,rho_est);
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
		for (n = 0; n < N; n++) {
			resample_numbers[n] = resample(N, (Uniform() + n - 1) / N, cumsum_weight);
		}
		check_resample = 1;
	}
	else {
		for (n = 0; n < N; n++) {
			resample_numbers[n] = n;
		}
		check_resample = 0;
	}

	/*���ʂ̊i�[*/
	pred_X_mean[0] = 0;
	state_X_mean[0] = 0;
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
		pred_X_mean[0] += pred_X_all[0][n] * 1.0 / N;
		state_X_mean[0] += state_X_all[0][n] * weight_state_all[0][n];
	}

	/*��������͌J��Ԃ�����*/
	for (t = 2; t < T; t++) {
		/*����O��(����Ӗ����O)���ʎ擾*/
		for (n = 0; n < N; n++) {
			post_X[n] = state_X_all[t - 2][n];
			post_weight[n] = weight_state_all[t - 2][n];
		}
		/*���_t�̃T���v�����O*/
		for (n = 0; n < N; n++) {
			pred_X[n] = sqrt(beta_est)*post_X[n] - sqrt(1 - beta_est)*rnorm(0, 1);
		}

		/*�d�݂̌v�Z*/
		sum_weight = 0.0;
		resample_check_weight = 0.0;
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_dinamic(DR[t],pred_X[n],q_qnorm_est,beta_est,rho_est) * post_weight[n];
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
			for (n = 0; n < N; n++) {
				resample_numbers[n] = resample(N, Uniform(), cumsum_weight);
			}
			check_resample = 1;
		}
		else {
			for (n = 0; n < N; n++) {
				resample_numbers[n] = n;
			}
			check_resample = 0;
		}

		/*���ʂ̊i�[*/
		pred_X_mean[t - 1] = 0;
		state_X_mean[t - 1] = 0;
		for (n = 0; n < N; n++) {
			pred_X_all[t-1][n] = pred_X[n];
			state_X_all[t-1][n] = pred_X[resample_numbers[n]];
			weight_all[t-1][n] = weight[n];
			if (check_resample == 0) {
				weight_state_all[t-1][n] = weight[n];
			}
			else {
				weight_state_all[t-1][n] = 1.0 / N;
			}
			pred_X_mean[t - 1] += pred_X_all[t - 1][n] * weight_state_all[t - 2][n];
			state_X_mean[t - 1] += state_X_all[t - 1][n] * weight_state_all[t - 1][n];
		}

	}
	return 0;
}

/*������*/
int particle_smoother() {
	/*T���_��weight�͕ς��Ȃ��̂ł��̂܂ܑ��*/
	state_X_all_bffs_mean_tmp = 0;
    #pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 2][n] = weight_state_all[T - 2][n];
		state_X_all_bffs[T - 2][n] = state_X_all[T - 2][n];
		state_X_all_bffs_mean_tmp += state_X_all_bffs[T - 2][n] * weight_state_all_bffs[T - 2][n];
	}
	state_X_all_bffs_mean_tmp = state_X_all_bffs_mean[T - 2];
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
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers[n] = resample(N, (Uniform() + n - 1) / N, cumsum_weight);
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
		/*���T���v�����O�̕K�v���ɉ����Č��ʂ̊i�[*/
		state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			if (check_resample == 0) {
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
	return 0;
}

/*Q EM�ōő剻��������*/
double Q() {
	/*�ȉ����֐��ɂ���@Q*/
	Q_state = 0;
	Q_obeserve = 0;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
            #pragma omp parallel for reduction(+:Q_state)
			for (n2 = 0; n2 < N; n2++) {
				Q_state += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * //weight
					log(
						dnorm(state_X_all_bffs[t][n], sqrt(beta_est)*state_X_all_bffs[t - 1][n2], sqrt(1 - beta_est))//X�̑J�ڊm��
					);
			}
			Q_obeserve += weight_state_all_bffs[t][n] *//weight
				log(
					g_DR_dinamic(DR[t], state_X_all_bffs[t-1][n], q_qnorm_est,beta_est, rho_est)//�ϑ��̊m��
				);
		}
	}
	first_state = 0;
    #pragma omp parallel for reduction(+:first_state)
	for (n = 0; n < N; n++) {
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				dnorm(state_X_all_bffs[0][n], sqrt(beta_est) * X_0_est, sqrt(1 - beta_est))//�������z����̊m��
			);
	}
	return Q_state + Q_obeserve + first_state;
}



/*Q�̊m���I���z�@*/
double Q_() {
	int k = 1;
	while (k<1000) {
		Now_Q = Q();
		beta_est_tmp = beta_est;
		rho_est_tmp = rho_est;
		q_qnorm_est_tmp = q_qnorm_est;
		X_0_est_tmp = X_0_est;
		/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
		sig_beta_est = sig_env(beta_est);
		sig_rho_est = sig_env(rho_est);

		a = 0;
		b = 0;
		c = 0;
		d = 0;
		int i;
		for (i = 0; i < 25; i++) {
			t = floor(Uniform() * 100);
			for (n = 0; n < N; n++) {
				//beta �����ϐ��̎��ɂ��āAbeta���V�O���C�h�֐��ŕϊ������l�̔���
				a += weight_state_all_bffs[t][n] * (
					1 / (exp(sig_beta_est)*(2 * pow(1 + exp(-sig_beta_est), 2) * (1 - 1 / (1 + exp(-sig_beta_est))))) +
					(pow(1 / (1 + exp(-sig_beta_est)), 3 / 2) * state_X_all_bffs[t - 1][n] * ((-sqrt(1 / (1 + exp(-sig_beta_est))))*state_X_all_bffs[t - 1][n] + state_X_all_bffs[t][n])) /
					(exp(sig_beta_est)*(2 * (1 - 1 / (1 + exp(-sig_beta_est))))) - pow((-sqrt(1 / (1 + exp(-sig_beta_est))))*state_X_all_bffs[t - 1][n] +
						state_X_all_bffs[t][n], 2) /
						(exp(sig_beta_est)*(2 * pow(1 + exp(-sig_beta_est), 2) * pow(1 - 1 / (1 + exp(-sig_beta_est)), 2))) +
					//���͊ϑ��ϐ��ɂ���
					((1 / 2)*exp(sig_beta_est + sig_rho_est)*(1 + exp(-sig_beta_est))*(exp(-2 * sig_beta_est - sig_rho_est) / pow(1 + exp(-sig_beta_est), 2) -
						exp(-sig_beta_est - sig_rho_est) / (1 + exp(-sig_beta_est))) -
						(exp(sig_rho_est)*sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n] *
						(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
							sqrt(1 - 1 / (1 + exp(-sig_rho_est))))) /
							(2 * sqrt(1 - 1 / (1 + exp(-sig_rho_est)))) +
						(1 / 2)*exp(sig_rho_est)*pow(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
							sqrt(1 - 1 / (1 + exp(-sig_rho_est))), 2) - (1 / 2)*exp(sig_beta_est + sig_rho_est)*(1 + exp(-sig_beta_est))*
						pow(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
							sqrt(1 - 1 / (1 + exp(-sig_rho_est))), 2)
						)
					);
				//rho �ϑ��ϐ��ɂ��� rho���V�O���C�h�֐��ŕϊ������l�̔���
				b += weight_state_all_bffs[t][n] * (
					-(1 / 2) - exp(sig_beta_est + sig_rho_est)*(1 +
						exp(-sig_beta_est))*((sqrt(1 / (1 + exp(-sig_beta_est))) * pow(1 / (1 + exp(-sig_rho_est)), (3 / 2))*state_X_all_bffs[t - 1][n]) /
						(exp(sig_rho_est)*(2 * sqrt(1 - 1 / (1 + exp(-sig_rho_est))))) - (q_qnorm_est -
							sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
							(exp(sig_rho_est)*(2 * pow(1 + exp(-sig_rho_est), 2) * pow(1 - 1 / (1 + exp(-sig_rho_est)), (3 / 2)))))*
							(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
								sqrt(1 - 1 / (1 + exp(-sig_rho_est)))) -
								(1 / 2)*exp(sig_beta_est + sig_rho_est)*(1 +
									exp(-sig_beta_est))*pow(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
										sqrt(1 - 1 / (1 + exp(-sig_rho_est))), 2)
					);
				//q_qnorm �ϑ��ϐ��ɂ���
				c += weight_state_all_bffs[t][n] * (
					(exp(sig_beta_est + sig_rho_est)*(1 +
						exp(-sig_beta_est))*(DR[t] - (q_qnorm_est - sqrt(1 / (1 + exp(-sig_beta_est))) * sqrt(1 / (1 + exp(-sig_rho_est))) * state_X_all_bffs[t - 1][n]) /
							sqrt(1 - 1 / (1 + exp(-sig_rho_est))))) / sqrt(1 - 1 / (1 + exp(-sig_rho_est)))
					);
			}
		}
		for (n = 0; n < N; n++) {
			//X_0 �����ϐ��ɂ���
			d += weight_state_all_bffs[t][n] * (
				(sqrt(1 / (1 + exp(-sig_beta_est))) * ((-sqrt(1 / (1 + exp(-sig_beta_est))))*X_0_est + state_X_all_bffs[0][n])) /
				(1 - 1 / (1 + exp(-sig_beta_est))) 
				);
		}
		sig_beta_est = sig_beta_est + a * alpha_grad / k;
		sig_rho_est = sig_rho_est + b * alpha_grad / k;
		q_qnorm_est = q_qnorm_est + c * alpha_grad / k;
		X_0_est = X_0_est + d * alpha_grad / k;
		beta_est = sig(sig_beta_est);
		rho_est = sig(sig_rho_est);
		
		printf("Old Q %f,Now_Q %f\n,beta_est %f,rho_est %f,q %f X_0_est %f\n\n",
			Now_Q, Q(), beta_est, rho_est, pnorm(q_qnorm_est,0,1), X_0_est);
		k = k + 1;
	}
	return 0;
}

/*Q�̍ŋ}�~���@*/
double Q_grad() {
		Now_Q = Q();
		beta_est_tmp = beta_est;
		rho_est_tmp = rho_est;
		q_qnorm_est_tmp = q_qnorm_est;
		X_0_est_tmp = X_0_est;
		/*beta��rho��[0,1]���񂪂��邽�߁A�_�~�[�ϐ���p����K�v������*/
		sig_beta_est = sig_env(beta_est);
		sig_rho_est = sig_env(rho_est);
		sig_beta_est_tmp = sig_beta_est;
		sig_rho_est_tmp = sig_rho_est;

		a = 0;
		b = 0;
		c = 0;
		d = 0;
		for (t = 1; t < T; t++) {
			for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:a)
				for (n2 = 0; n2 < N; n2++) {
					//beta �����ϐ��̎��ɂ��āAbeta���V�O���C�h�֐��ŕϊ������l�̔���
					a += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * exp(sig_beta_est) / 2 * (
						-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2] + pow(state_X_all_bffs[t - 1][n2], 2))) -
						((-exp(-sig_beta_est) *pow(state_X_all_bffs[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2])) +
						1 / (1 + exp(sig_beta_est))
						);
				}

				//���͊ϑ��ϐ��ɂ���
				a += weight_state_all_bffs[t - 1][n] * (
					exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
					(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
					(pow(DR[t], 2) -
						((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
						2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
					(-(exp(-sig_beta_est + sig_rho_est) / (1 + exp(-sig_beta_est)))*pow(state_X_all_bffs[t - 1][n], 2) + sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2)*q_qnorm_est*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2) * state_X_all_bffs[t - 1][n])
					);

				//�Ō�͏����_����̔����ɂ���
				a += weight_state_all_bffs[0][n] * exp(sig_beta_est) / 2 * (
					-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[0][n] * X_0_est + pow(X_0_est, 2))) -
					((-exp(-sig_beta_est) * pow(state_X_all_bffs[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] * X_0_est)) +
					1 / (1 + exp(sig_beta_est))
					);

				//rho �ϑ��ϐ��ɂ��� rho���V�O���C�h�֐��ŕϊ������l�̔���
				b += weight_state_all_bffs[t - 1][n] * (
					-1 / 2 +
					((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
					(pow(DR[t], 2) -
						((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est) -
						2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
						(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
					(-(exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - (exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*state_X_all_bffs[t - 1][n]) - DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))
					);




				//q_qnorm �ϑ��ϐ��ɂ���
				c += weight_state_all_bffs[t - 1][n] * (
					(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
					((1 + exp(sig_rho_est))*q_qnorm_est - sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n] - DR[t] * sqrt(1 + exp(sig_rho_est)))
					);
			}
		}
            #pragma omp parallel for reduction(+:d)
			for (n = 0; n < N; n++) {
				//X_0 �����ϐ��ɂ���
				d += weight_state_all_bffs[0][n] * (
					exp(sig_beta_est) * (sqrt(1 - exp(-sig_beta_est))*state_X_all_bffs[0][n] - X_0_est)
					);
			}
		int grad_check = 1;
		l = 1;
		printf("\n Score%f a%f b%f c%f d%f \n\n", Q(),a,b,c,d);
		while (grad_check) {
			sig_beta_est = sig_beta_est_tmp;
			sig_rho_est = sig_rho_est_tmp;
			q_qnorm_est = q_qnorm_est_tmp;
			X_0_est = X_0_est_tmp;
			sig_beta_est = sig_beta_est + a * pow(beta_grad,l);
			sig_rho_est = sig_rho_est + b * pow(beta_grad, l);
			q_qnorm_est = q_qnorm_est + c * pow(beta_grad, l);
			X_0_est = X_0_est + d * pow(beta_grad, l);
			beta_est = sig(sig_beta_est);
			rho_est = sig(sig_rho_est);
			if (Now_Q - Q() <= 0){//alpha_grad*pow(beta_grad,l)*pow(Now_Q,2)) {
				grad_check = 0;
			}
			l += 1;
			if (l > 100) {
				grad_stop_check = 0;
				return 0;
			}
		}

		printf("Old Q %f,Now_Q %f\n,beta_est %f,rho_est %f,q %f X_0_est %f\n\n",
			Now_Q, Q(), beta_est, rho_est, pnorm(q_qnorm_est, 0, 1), X_0_est);
		
	return 0;
}


int main(void) {
	/*�������X�L�b�v������*/
	int i;
	for (i = 0; i < rand_seed; i++) {
		Uniform();
	}

	/*X�����f���ɏ]���ăV�~�����[�V�����p�ɃT���v�����O�A������DR���T���v�����O ���_t��DR�͎��_t-1��X���p�����[�^�ɂ����K���z�ɏ]���̂ŁA��������_�ɒ���*/
	X[0] = sqrt(beta)*X_0 + sqrt(1 - beta) * rnorm(0, 1);
	
	for (t = 1; t < T; t++) {
		X[t] = sqrt(beta)*X[t - 1] + sqrt(1 - beta) * rnorm(0, 1);
		DR[t] = r_DDR(X[t-1], q_qnorm, rho, beta);
	}

	beta_est = beta ;
	rho_est = rho;
	q_qnorm_est = q_qnorm ;
	X_0_est = X_0 ;

	grad_stop_check = 1;
	/*
	while (grad_stop_check) {
		particle_filter();
		particle_smoother();
		Q_grad();
	}
	*/
	particle_filter();
	particle_smoother();

	printf("\n\n Score%f \n\n", Q());
	FILE *fp;
	if (fopen_s(&fp, "particle.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 1; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, state_X_all[t][n], weight_state_all[t][n], N/20 * weight_state_all[t][n],weight_state_all_bffs[t][n], N / 20 * weight_state_all_bffs[t][n]);

		}
	}
	fclose(fp);

	if (fopen_s(&fp, "X.csv", "w") != 0) {
		return 0;
	}
	for (t = 1; t < T - 1; t++) {
		fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, X[t], state_X_mean[t], pred_X_mean[t], pnorm(DR[t],0,1), state_X_all_bffs_mean[t]);
	}

	fclose(fp);
	FILE *gp,*gp2;
	gp = _popen(GNUPLOT_PATH, "w");
	
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
	fprintf(gp, "plot 'particle.csv' using 1:2:6:5 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp);
	fprintf(gp, "replot 'X.csv' using 1:2 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'Answer'\n");
	fflush(gp);
	//fprintf(gp, "replot 'X.csv' using 1:3 with lines linetype 1 lw 2.0 linecolor rgb '#ffff00 ' title 'Filter'\n");
	//fflush(gp);
	fprintf(gp, "replot 'X.csv' using 1:6 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
	fflush(gp);
	//fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'Predict'\n");
	//fflush(gp);

	gp2 = _popen(GNUPLOT_PATH, "w");
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
	fprintf(gp2, "plot 'X.csv' using 1:5 with lines linetype 1 lw 3.0 linecolor rgb '#ff0000 ' title 'DR'\n");
	fflush(gp2);



	system("pause");
	fprintf(gp, "exit\n");    // gnuplot�̏I��
	_pclose(gp);
	
	while (1) {

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
		return 0;
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
