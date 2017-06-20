#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define T 100
#define N 100
#define phi_rho 0.95
#define phi_pd 0.95
#define mean_rho 0.1
#define mean_pd 0.04
#define sd_sig_rho 0.01
#define sd_sig_pd 0.01
#define alpha 0.00000001

/*Answer�i�[*/
double pd[T];
double rho[T];
double DR[T];

/*�������z����̃T���v�����O�i�[*/
double first_pd_sig[N];
double first_rho_sig[N];

/*���_t�̗\���l�i�[*/
double pred_pd_sig[N]; //pd��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ�
double pred_rho_sig[N]; //rho��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ�
double pred_pd[N]; //pd��Particle [0,1]�͈̔͂ɒ��������� �\���l
double pred_rho[N]; //rho��Particle [0,1]�͈̔͂ɒ��������́@�\���l
double weight[N]; // weight

/*�S���Ԃ̐���l�i�[�@�t�B���^�����O*/
double pred_pd_all[T][N]; //pd��Particle [0,1]�͈̔͂ɒ��������� 
double pred_rho_all[T][N]; //rho��Particle [0,1]�͈̔͂ɒ���������
double state_pd_sig_all[T][N]; //pd��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ� ���T���v�����O��������
double state_rho_sig_all[T][N]; //rho��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ� ���T���v�����O��������
double state_pd_all[T][N]; //pd��Particle [0,1]�͈̔͂ɒ��������� ���T���v�����O��������
double state_rho_all[T][N]; //rho��Particle [0,1]�͈̔͂ɒ��������� ���T���v�����O��������
double weight_all[T][N]; // weight
double weight_state_all[T][N]; // weight ���T���v�����O��������

/*�S���Ԃ̐���l�i�[�@������*/
double state_pd_sig_all_bffs[T][N]; //pd��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ�
double state_rho_sig_all_bffs[T][N]; //rho��Particle���̂��́@�V�O���C�h�֐��̋t�֐��ŕϊ�
double weight_state_all_bffs[T][N]; // weight ���T���v�����O��������


/*�p�����[�^�p�ϐ�*/
double phi_rho_est;
double phi_pd_est;
double mean_rho_est;
double mean_pd_est;
double sd_sig_rho_est;
double sd_sig_pd_est;


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

/*����O�̌���*/
double post_pd_sig[N];
double post_rho_sig[N];
double post_weight[N];

/*EM�̌v�Z*/
double q_state;
double q_obeserve;
double first_observe;
double first_state;

/*EM���̍ŋ}�~���@*/
double phi_rho_est_tmp;
double phi_pd_est_tmp;
double mean_rho_est_tmp;
double mean_pd_est_tmp;
double sd_sig_rho_est_tmp;
double sd_sig_pd_est_tmp;
double a = 0;
double b = 0;
double c = 0;
double d = 0;
double e = 0;
double f = 0;
double Now_Q;

/*time��Particle��for���p�ϐ�*/
int t;
int n;
int n2;

/*�t�B���^�����O*/
int particle_filter() {
	/*���_1�ł̃t�B���^�����O�J�n*/
	/*�������z����̃T���v�����O���A���̂܂܎��_1�̃T���v�����O*/
	for (n = 0; n < N; n++) {
		/*�������z����@���_0�ƍl����*/
		first_pd_sig[n] = rnorm(sig_env(mean_pd_est), sd_sig_pd_est);
		first_rho_sig[n] = rnorm(sig_env(mean_rho_est), sd_sig_rho_est);
		/*���̌��ʂ���T���v�����O*/
		pred_pd_sig[n] = rnorm(sig_env(mean_pd_est) + phi_pd*(first_pd_sig[n] - sig_env(mean_pd_est)), sd_sig_pd_est);
		pred_rho_sig[n] = rnorm(sig_env(mean_rho_est) + phi_rho*(first_rho_sig[n] - sig_env(mean_rho_est)), sd_sig_rho_est);
		/*pd �� rho�ɕϊ�*/
		pred_pd[n] = sig(pred_pd_sig[n]);
		pred_rho[n] = sig(pred_rho_sig[n]);
	}



	/*�d�݂̌v�Z*/
	sum_weight = 0;
	resample_check_weight = 0;
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_fn(DR[0], pred_pd[n], pred_rho[n]);
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
	for (n = 0; n < N; n++) {
		pred_pd_all[0][n] = pred_pd[n];
		pred_rho_all[0][n] = pred_rho[n];
		state_pd_sig_all[0][n] = pred_pd_sig[resample_numbers[n]];
		state_rho_sig_all[0][n] = pred_rho_sig[resample_numbers[n]];
		state_pd_all[0][n] = pred_pd[resample_numbers[n]];
		state_rho_all[0][n] = pred_rho[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		if (check_resample == 0) {
			weight_state_all[0][n] = weight[n];
		}
		else {
			weight_state_all[0][n] = 1.0 / N;
		}
	}

	/*��������͌J��Ԃ�����*/
	for (t = 1; t < T; t++) {
		/*����O�̌��ʎ擾*/
		for (n = 0; n < N; n++) {
			re_n = resample_numbers[n];
			post_pd_sig[n] = state_pd_sig_all[t - 1][re_n];
			post_rho_sig[n] = state_rho_sig_all[t - 1][re_n];
			post_weight[n] = weight_state_all[t - 1][re_n];
		}
		/*���_t�̃T���v�����O*/
		for (n = 0; n < N; n++) {
			/*���̌��ʂ���T���v�����O*/
			pred_pd_sig[n] = rnorm(sig_env(mean_pd_est) + phi_pd_est*(post_pd_sig[n] - sig_env(mean_pd_est)), sd_sig_pd_est);
			pred_rho_sig[n] = rnorm(sig_env(mean_rho_est) + phi_rho_est*(post_rho_sig[n] - sig_env(mean_rho_est)), sd_sig_rho_est);
			/*pd �� rho�ɕϊ�*/
			pred_pd[n] = sig(pred_pd_sig[n]);
			pred_rho[n] = sig(pred_rho_sig[n]);
		}

		/*�d�݂̌v�Z*/
		sum_weight = 0;
		resample_check_weight = 0;
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_fn(DR[t], pred_pd[n], pred_rho[n]);
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
		for (n = 0; n < N; n++) {
			pred_pd_all[t][n] = pred_pd[n];
			pred_rho_all[t][n] = pred_rho[n];
			state_pd_sig_all[t][n] = pred_pd_sig[resample_numbers[n]];
			state_rho_sig_all[t][n] = pred_rho_sig[resample_numbers[n]];
			state_pd_all[t][n] = pred_pd[resample_numbers[n]];
			state_rho_all[t][n] = pred_rho[resample_numbers[n]];
			weight_all[t][n] = weight[n];
			if (check_resample == 0) {
				weight_state_all[t][n] = weight[n];
			}
			else {
				weight_state_all[t][n] = 1.0 / N;
			}
		}

	}
	return 0;
}

/*������*/
int particle_smoother() {
	/*T���_��weight�͕ς��Ȃ��̂ł��̂܂ܑ��*/
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 1][n] = weight_state_all[T - 1][n];
		state_pd_sig_all_bffs[T - 1][n] = state_pd_sig_all[T - 1][n];
		state_rho_sig_all_bffs[T - 1][n] = state_rho_sig_all[T - 1][n];
	}
	for (t = T - 2; t > -1; t--) {
		sum_weight = 0;
		resample_check_weight = 0;
		for (n = 0; n < N; n++) {
			bunsi_sum = 0;
			bunbo_sum = 0;
			for (n2 = 0; n2 < N; n2++) {
				/*���q�v�Z ���킩��ɂ������ǂ�����*/
				bunsi[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(state_pd_sig_all_bffs[t + 1][n2],
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n] - sig_env(mean_pd_est)),
						sd_sig_pd_est)*
					dnorm(state_rho_sig_all_bffs[t + 1][n2],
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n] - sig_env(mean_pd_est)),
						sd_sig_rho_est);
				bunsi_sum += bunsi[n][n2];
				/*����v�Z ���킩��ɂ������ǂ�����*/
				bunbo[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(state_pd_sig_all_bffs[t + 1][n],
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n2] - sig_env(mean_pd_est)),
						sd_sig_pd_est)*
					dnorm(state_rho_sig_all_bffs[t + 1][n],
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n2] - sig_env(mean_pd_est)),
						sd_sig_rho_est);
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
		/*���T���v�����O�̕K�v���ɉ����Č��ʂ̊i�[*/
		for (n = 0; n < N; n++) {
			if (check_resample == 0) {
				state_pd_sig_all_bffs[t][n] = state_pd_sig_all[t][n];
				state_rho_sig_all_bffs[t][n] = state_rho_sig_all[t][n];
			}
			else {
				state_pd_sig_all_bffs[t][n] = state_pd_sig_all[t][resample_numbers[n]];
				state_rho_sig_all_bffs[t][n] = state_rho_sig_all[t][resample_numbers[n]];
				weight_state_all_bffs[t][n] = 1.0 / N;
			}
		}


	}
	return 0;
}

/*Q EM�ōŏ�����������*/
double Q() {
	/*�ȉ����֐��ɂ���@Q*/
	q_state = 0;
	q_obeserve = 0;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			q_state += weight_state_all_bffs[t][n] *//weight
				log(
					dnorm(state_pd_sig_all_bffs[t][n], sig_env(mean_pd_est) + phi_rho_est * (state_pd_sig_all_bffs[t - 1][n] - sig_env(mean_pd_est)), sd_sig_pd_est)*//pd�̑J�ڊm��
					dnorm(state_rho_sig_all_bffs[t][n], sig_env(mean_rho_est) + phi_rho_est * (state_rho_sig_all_bffs[t - 1][n] - sig_env(mean_rho_est)), sd_sig_rho_est)//rho�̑J�ڊm��
				);
			q_obeserve += weight_state_all_bffs[t][n] *//weight
				log(
					g_DR_fn(DR[t], sig(state_pd_sig_all_bffs[t][n]), sig(state_rho_sig_all_bffs[t][n]))//�ϑ��̊m��
				);
		}
	}
	first_observe = 0;
	first_state = 0;
	for (n = 0; n < N; n++) {
		first_observe += weight_state_all_bffs[0][n] *//weight
			log(
				g_DR_fn(DR[0], sig(state_pd_sig_all_bffs[0][n]), sig(state_rho_sig_all_bffs[0][n]))//�ϑ��̊m��
			);
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				dnorm(state_pd_sig_all_bffs[0][n],sig_env(mean_pd_est),sd_sig_pd_est)*
				dnorm(state_rho_sig_all_bffs[0][n], sig_env(mean_rho_est), sd_sig_rho_est)//�������z����̊m��
			);
	}
	return q_state + q_obeserve + first_observe + first_state;
}

/*Q�̍ŋ}�~��*/
double q() {
	while (1) {
		Now_Q = Q();
		phi_rho_est_tmp = phi_rho_est;
		phi_pd_est_tmp = phi_pd_est;
		mean_rho_est_tmp = mean_rho_est;
		mean_pd_est_tmp = mean_pd_est;
		/*�_�~�[�ϐ���p���邽�� sigma=z^2�Ƃ���*/
		sd_sig_rho_est = sqrt(sd_sig_rho_est);
		sd_sig_pd_est_tmp = sqrt(sd_sig_pd_est);
		sd_sig_rho_est_tmp = sd_sig_rho_est;
		sd_sig_pd_est_tmp = sd_sig_pd_est;
		a = 0;
		b = 0;
		c = 0;
		d = 0;
		e = 0;
		f = 0;

		for (t = 1; t < T; t++) {
			for (n = 0; n < N; n++) {
				a += weight_state_all_bffs[t][n] * (sig_env(mean_rho_est) - state_rho_sig_all_bffs[t - 1][n]) / (2 * pow(sd_sig_rho_est,4));
				b += weight_state_all_bffs[t][n] * (sig_env(mean_pd_est) - state_pd_sig_all_bffs[t - 1][n]) / (2 * pow(sd_sig_pd_est, 4));
				c += weight_state_all_bffs[t][n] * (phi_rho_est - 1) / (2 * pow(sd_sig_rho_est, 4));
				d += weight_state_all_bffs[t][n] * (phi_pd_est - 1) / (2 * pow(sd_sig_pd_est, 4));
				
				e += weight_state_all_bffs[t][n] * 1 / sd_sig_rho_est -
					(state_rho_sig_all_bffs[t][n] - (sig_env(mean_rho_est) + phi_rho_est*(state_rho_sig_all_bffs[t - 1][n] - sig_env(mean_rho_est)))) /
					(2 * pow(sd_sig_rho_est, 2));
				f += weight_state_all_bffs[t][n] * 1 / sd_sig_pd_est -
					(state_pd_sig_all_bffs[t][n] - (sig_env(mean_pd_est) + phi_pd_est*(state_pd_sig_all_bffs[t - 1][n] - sig_env(mean_pd_est)))) /
					(2 * pow(sd_sig_pd_est, 2));
			}
		}

		phi_rho_est = phi_rho_est + a*alpha;
		phi_pd_est = phi_pd_est + b*alpha;
		mean_rho_est = mean_rho_est + sig(c)*alpha;
		mean_pd_est = mean_pd_est + sig(d)*alpha;
		sd_sig_rho_est = sd_sig_rho_est + e*alpha;
		sd_sig_pd_est = sd_sig_pd_est + f*alpha;
		printf("Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n phi_pd_est %f,mean_pd_est %f,sd_sig_pd_est %f\n",
			Q(), phi_rho_est, mean_rho_est, sd_sig_rho_est, phi_pd_est, mean_pd_est, sd_sig_pd_est);
		if (sqrt(pow(a,2)+pow(b,2)+pow(c,2)+pow(d,2)+pow(e,2)+pow(f,2)) < 0.001) {
			phi_rho_est = phi_rho_est_tmp;
			phi_pd_est = phi_pd_est_tmp;
			mean_rho_est = mean_rho_est_tmp;
			mean_pd_est = mean_pd_est_tmp;
			sd_sig_rho_est = sd_sig_rho_est_tmp;
			sd_sig_pd_est = sd_sig_pd_est_tmp;
			return 0;
		}
	}
}

/*Q�̍ŋ}�~�� �������ւ�Ă�*/
double prob_q() {
	while (1) {
		Now_Q = Q();
		phi_rho_est_tmp = phi_rho_est;
		phi_pd_est_tmp = phi_pd_est;
		mean_rho_est_tmp = mean_rho_est;
		mean_pd_est_tmp = mean_pd_est;
		sd_sig_rho_est_tmp = sd_sig_rho_est;
		sd_sig_pd_est_tmp = sd_sig_pd_est;
		a = 0;
		b = 0;
		c = 0;
		d = 0;
		e = 0;
		f = 0;

		for (t = 1; t < T; t++) {
			for (n = 0; n < N; n++) {
				a += weight_state_all_bffs[t][n] * (sig_env(mean_rho_est) - state_rho_sig_all_bffs[t - 1][n]) / sd_sig_rho_est;
				b += weight_state_all_bffs[t][n] * (sig_env(mean_pd_est) - state_pd_sig_all_bffs[t - 1][n]) / sd_sig_rho_est;
				c += weight_state_all_bffs[t][n] * (phi_rho_est - 1) / (2 * sd_sig_rho_est);
				d += weight_state_all_bffs[t][n] * (phi_pd_est - 1) / (2 * sd_sig_pd_est);
				e += weight_state_all_bffs[t][n] * 1 / sd_sig_rho_est -
					(state_rho_sig_all_bffs[t][n] - (sig_env(mean_rho_est) + phi_rho_est*(state_rho_sig_all_bffs[t - 1][n] - sig_env(mean_rho_est)))) /
					(2 * pow(sd_sig_rho_est, 2));
				f += weight_state_all_bffs[t][n] * 1 / sd_sig_pd_est -
					(state_pd_sig_all_bffs[t][n] - (sig_env(mean_pd_est) + phi_pd_est*(state_pd_sig_all_bffs[t - 1][n] - sig_env(mean_pd_est)))) /
					(2 * pow(sd_sig_pd_est, 2));
			}
		}
		int prob_choice;
		prob_choice = (int) (Uniform() * 100) % 6;
		double A[6] = {a,b,c,d,e,f};
		double B[6] = { 0 };

		B[prob_choice] = A[prob_choice];
		phi_rho_est = phi_rho_est -  B[0]*alpha;
		phi_pd_est = phi_pd_est - B[1]*alpha;
		mean_rho_est = mean_rho_est - B[2]*alpha;
		mean_pd_est = mean_pd_est - B[3]*alpha;
		sd_sig_rho_est = sd_sig_rho_est - B[4]*alpha;
		sd_sig_pd_est = sd_sig_pd_est - B[5]*alpha;
		printf("Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n phi_pd_est %f,mean_pd_est %f,sd_sig_pd_est %f\n",
			Q(), phi_rho_est, mean_rho_est, sd_sig_rho_est, phi_pd_est, mean_pd_est, sd_sig_pd_est);
		if (Now_Q > Q()) {
			phi_rho_est = phi_rho_est_tmp;
			phi_pd_est = phi_pd_est_tmp;
			mean_rho_est = mean_rho_est_tmp;
			mean_pd_est = mean_pd_est_tmp;
			sd_sig_rho_est = sd_sig_rho_est_tmp;
			sd_sig_pd_est = sd_sig_pd_est_tmp;
			return 0;
		}
	}
}

int main(void) {
	
	/*PD��rho�����ꂼ��AR���f���ɏ]���ăV�~�����[�V�����p�ɃT���v�����O*/
	AR_sim(T,pd, mean_pd, sd_sig_pd, phi_pd);
	AR_sim(T,rho, mean_rho, sd_sig_rho, phi_rho);
	
	/*���p�@��p���āA�e���_�ł̃p�����[�^����DR�𔭐�*/
	for (t = 0; t < T; t++) {
		DR[t] = reject_sample(pd[t], rho[t]);
	}

	/*�p�����[�^�p�ϐ�*/
	phi_rho_est = phi_rho- 0.05;
	phi_pd_est = phi_pd - 0.05;
	mean_rho_est = mean_rho + 0.05;
	mean_pd_est = mean_pd + 0.05;
	sd_sig_rho_est = sd_sig_rho + 0.01;
	sd_sig_pd_est = sd_sig_pd + 0.01;

	while (1) {
		particle_filter();
		particle_smoother();
		q();
		printf("Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n phi_pd_est %f,mean_pd_est %f,sd_sig_pd_est %f\n",
			Now_Q, phi_rho_est, mean_rho_est, sd_sig_rho_est, phi_pd_est, mean_pd_est, sd_sig_pd_est);
		
		if (Q() > 1000) {
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

			/*��������plot*/
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
		}
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