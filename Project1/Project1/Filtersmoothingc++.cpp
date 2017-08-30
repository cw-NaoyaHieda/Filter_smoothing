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
#define pd_phi 0.85
#define pd_mu 0.05
#define pd_sd 0.1
#define pd_0 0.05
#define rho 0.1
#define a_grad 0.0001
#define b_grad 0.5
#define N 1000
#define T 200
/*フィルタリングの結果格納*/

std::mt19937 mt(120);
std::uniform_real_distribution<double> r_rand(0.0, 1.0);
std::uniform_real_distribution<double> r_rand_phi(0.65,0.95);
// 平均0.0、標準偏差1.0で分布させる
std::normal_distribution<> dist(0.0, 1.0);


std::vector<std::vector<double> > filter_pd(T, std::vector<double>(N));
std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
/*平滑化の結果格納*/
std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));
/*Qの計算のためのwight*/
std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));
/*Answer格納*/
std::vector<double> pd(T);
std::vector<double> DR(T);


/*リサンプリング関数*/
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


/*フィルタリング*/
void particle_filter(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,std::vector<double>& state_pd_mean) {
	int n;
	int t;
	double pred_pd_mean_tmp;
	double state_pd_mean_tmp;
	/*時点tの予測値格納*/
	std::vector<double> pred_pd(N), weight(N); //XのParticle weight

															/*途中の処理用変数*/
	double sum_weight, resample_check_weight; //正規化因子(weightの合計) リサンプリングの判断基準(正規化尤度の二乗の合計)
	std::vector<double> cumsum_weight(N); //累積尤度　正規化した上で計算したもの
	std::vector<int> resample_numbers(N); //リサンプリングした結果の番号
	int check_resample; //リサンプリングしたかどうかの変数 0ならしてない、1ならしてる

						/*全期間の推定値格納*/
	std::vector<std::vector<double>> pred_pd_all(T, std::vector<double>(N)); //XのParticle  予測値 XのParticle フィルタリング
	std::vector<double> pred_pd_mean(T); //Xの予測値,Xのフィルタリング結果
	std::vector<std::vector<double>> weight_all(T, std::vector<double>(N)); // weight 予測値 weight フィルタリング

																			/*一期前の結果*/
	std::vector<double> post_pd(N), post_weight(N);

	/*時点1でのフィルタリング開始*/
	/*初期分布からのサンプリングし、そのまま時点1のサンプリング*/
#pragma omp parallel for
	for (n = 0; n < N; n++) {
		/*初期分布から　時点0と考える*/
		pred_pd[n] = sig(pd_mu_est + pd_phi_est*(pd_0_est - pd_mu_est) + pd_sd_est * dist(mt));
	}

	/*重みの計算*/
	sum_weight = 0;
	resample_check_weight = 0;
#pragma omp parallel for reduction(+:sum_weight)
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_fn(DR[0], pred_pd[n], rho_est);
		sum_weight += weight[n];
	}
	/*重みを正規化しながら、リサンプリング判断用変数の計算と累積尤度の計算*/
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

	/*リサンプリングが必要かどうか判断したうえで必要ならリサンプリング 必要ない場合は順番に数字を入れる*/
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

	/*結果の格納*/
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
	/*こっからは繰り返し処理*/
	for (t = 1; t < T; t++) {
		/*一期前の(ある意味期前)結果取得*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_pd[n] = filter_pd[t - 1][n];
			post_weight[n] = filter_weight[t - 1][n];
		}
		/*時点tのサンプリング*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			pred_pd[n] = sig(pd_mu_est + pd_phi_est*(sig_env(post_pd[n]) - pd_mu_est) + pd_sd_est * dist(mt));
		}

		/*重みの計算*/
		sum_weight = 0.0;
		resample_check_weight = 0.0;
#pragma omp parallel for reduction(+:sum_weight)
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_fn(DR[t], pred_pd[n], rho_est) * post_weight[n];
			sum_weight += weight[n];
		}

		/*重みを正規化しながら、リサンプリング判断用変数の計算と累積尤度の計算*/
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
		/*リサンプリングが必要かどうか判断したうえで必要ならリサンプリング 必要ない場合は順番に数字を入れる*/
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

		/*結果の格納*/
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

/*平滑化*/
void particle_smoother(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,std::vector<double>& state_pd_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight, bunsi_sum, bunbo_sum, state_pd_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T時点のweightは変わらないのでそのまま代入*/
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
				/*分母計算*/
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
			/*分子計算*/
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
		/*正規化と累積相対尤度の計算*/
		for (n = 0; n < N; n++) {
			smoother_weight[t][n] = smoother_weight[t][n] / sum_weight;
			if (n != 0) {
				cumsum_weight[n] = smoother_weight[t][n] + cumsum_weight[n - 1];
			}
			else {
				cumsum_weight[n] = smoother_weight[t][n];
			}
		}

		/*平滑化した推定値を計算*/
		state_pd_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			state_pd_all_bffs_mean_tmp += filter_pd[t][n] * smoother_weight[t][n];
		}
		state_pd_all_bffs_mean[t] = state_pd_all_bffs_mean_tmp;
	}
}

/*Qの計算に必要な新しいwight*/
void Q_weight_calc(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est) {
	int n, n2, t;
	double bunbo;
	for (t = T - 1; t > 0; t--) {
		for (n2 = 0; n2 < N; n2++) {
			bunbo = 0;
#pragma omp parallel for reduction(+:bunbo)
			for (n = 0; n < N; n++) {
				/*分母計算*/
				bunbo += filter_weight[t - 1][n] * dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est);
			}
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				/*分子計算しつつ代入*/
				Q_weight[t][n][n2] = filter_weight[t - 1][n] * smoother_weight[t][n2] *
					dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est) / bunbo;
				//printf("%f\n", Q_weight[t + 1][n][n2]);
			}
		}

	}

}

/*EMアルゴリズムで最大化したい式*/
double Q(double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;
	int t, n, n2;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:Q_state)
			for (n2 = 0; n2 < N; n2++) {
				Q_state += Q_weight[t][n2][n] * //weight
					log(
						std::max(dnorm(sig_env(filter_pd[t][n]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est), pd_sd_est), 0.000000000000001)//Xの遷移確率
					);
			}
		}
	}
	for (t = 1; t < T; t++) {
#pragma omp parallel for reduction(+:Q_obeserve)
		for (n = 0; n < N; n++) {
			Q_obeserve += smoother_weight[t][n] *//weight
				log(
					std::max(g_DR_(DR[t], filter_pd[t][n], rho_est), 0.000000000000001)//観測の確率
				);
		}
	}
#pragma omp parallel for reduction(+:first_state) reduction(+:Q_obeserve)
	for (n = 0; n < N; n++) {
		first_state += smoother_weight[0][n] *//weight
			log(
				dnorm(sig_env(filter_pd[0][n]), pd_mu_est + pd_phi_est*(pd_0_est - pd_mu_est), pd_sd_est)//初期分布からの確率
			);
		Q_obeserve += smoother_weight[0][n] *//weight
			log(
				g_DR_(DR[0], filter_pd[0][n], rho_est)//観測の確率
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
	/*制約があるため、ダミー変数を用いる必要がある*/
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
				//phi 説明変数の式について、phiをシグモイド関数で変換した値の微分
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
		//phi 初期点からの発生について
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
	/*制約があるため、ダミー変数を用いる必要がある*/
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
				//sd 説明変数の式について、sdを対数関数で変換した値の微分
				sd_grad += Q_weight[t][n2][n] * (
					-1 + 1 / exp(2 * log_pd_sd_est)*pow(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)), 2.0)
					);
			}
		}
	}
#pragma omp parallel for reduction(+:sd_grad)
	for (n = 0; n < N; n++) {
		//sd 初期点からの発生について
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
	/*制約があるため、ダミー変数を用いる必要がある*/
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
				//mu 説明変数の式について
				mu_grad += Q_weight[t][n2][n] * (
					1 / exp(2 * log_pd_sd_est)*(1 - pd_phi_est)*(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)))
					);
			}
		}
	}

#pragma omp parallel for reduction(+:mu_grad)
	for (n = 0; n < N; n++) {
		//mu 初期点からの発生について
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
	/*制約があるため、ダミー変数を用いる必要がある*/
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
		//pd_0 初期点からの発生について
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
	/*制約があるため、ダミー変数を用いる必要がある*/
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
		//rho 観測式について
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
	int n, t, i, j, k, l,s;
	double pd_mu_est;
	double pd_phi_est;
	double pd_sd_est;
	double pd_0_est;
	double rho_est;
	/*フィルタリングの結果格納*/
	std::vector<double> filter_pd_mean(T);
	/*平滑化の結果格納*/
	std::vector<double> smoother_pd_mean(T);

	
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(5);
	lbfgs_parameter_t param;



	FILE *fp;
	if (fopen_s(&fp, "partaneter.csv", "w") != 0) {
		return 0;
	}

	
	
	fprintf(fp,"number,Iteration,phi,mu,sd,zero,rho\n");
	fprintf(fp, "-1,-1,%f,%f,%f,%f,%f", pd_phi, sig(pd_mu), pd_sd, sig(pd_0), rho);
	int grad_stop_check;
	for (s = 1; s < 16; s++) {
		/*PD,rho,DRを発生させる*/
		pd[0] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd_0) - sig_env(pd_mu)) + pd_sd * dist(mt));
		DR[0] = reject_sample(pd[0], rho);
		for (t = 1; t < T; t++) {
			pd[t] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd[t - 1]) - sig_env(pd_mu)) + pd_sd * dist(mt));
			DR[t] = reject_sample(pd[t], rho);
		}

		x[0] = sig_phi_env(r_rand_phi(mt));
		x[1] = sig_env(r_rand(mt) / 10);//EMでややこしいので，最初から変換しておく
		x[2] = log(r_rand(mt) / 10 + 0.05);
		x[3] = sig_env(r_rand(mt) / 10);//EMでややこしいので，最初から変換しておく
		x[4] = sig_env(r_rand(mt) / 5);

		printf("%d,0,%f,%f,%f,%f,%f\n", s, sig_phi(x[0]), sig(x[1]), exp(x[2]), sig(x[3]), sig(x[4]));
		fprintf(fp, "%d,0,%f,%f,%f,%f,%f\n", s,sig_phi(x[0]), sig(x[1]), exp(x[2]), sig(x[3]), sig(x[4]));

		grad_stop_check = 1;
		while (grad_stop_check < 16) {
			particle_filter(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]), filter_pd_mean);
			particle_smoother(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]), smoother_pd_mean);
			Q_weight_calc(sig_phi(x[0]), x[1], exp(x[2]), x[3], sig(x[4]));
			lbfgs_parameter_init(&param);
			lbfgs(5, x, &fx, evaluate, progress, NULL, &param);
			printf("%d,%d,%f,%f,%f,%f,%f\n", s, grad_stop_check, sig_phi(x[0]), sig(x[1]), exp(x[2]), sig(x[3]), sig(x[4]));
			fprintf(fp, "%d,%d,%f,%f,%f,%f,%f\n", s, grad_stop_check,sig_phi(x[0]), sig(x[1]), exp(x[2]), sig(x[3]), sig(x[4]));
			grad_stop_check += 1;
		}
	}
	

	fclose(fp);

	

	return 0;
}


