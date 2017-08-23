#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <float.h>
#include <random>
#include "myfunc.h"
#include "sampling_DR.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359
#define pd_phi 0.9
#define pd_mu 0.05
#define pd_sd 0.1
#define pd_0 0.05
#define rho 0.07
#define a_grad 0.0001
#define b_grad 0.5
std::mt19937 mt(100);
std::uniform_real_distribution<double> r_rand(0.0, 1.0);
// 平均0.0、標準偏差1.0で分布させる
std::normal_distribution<> dist(0.0, 1.0);

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
void particle_filter(std::vector<double>& DR, double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,
	int N, int T, std::vector<std::vector<double>>& state_pd_all,
	std::vector<std::vector<double>>& weight_state_all, std::vector<double>& state_pd_mean) {
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
		resample_check_weight += pow(weight[n], 2);
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
		state_pd_all[0][n] = pred_pd[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		if (check_resample == 0) {
			weight_state_all[0][n] = weight[n];
		}
		else {
			weight_state_all[0][n] = 1.0 / N;
		}
		pred_pd_mean_tmp += pred_pd_all[0][n] * 1.0 / N;
		state_pd_mean_tmp += state_pd_all[0][n] * weight_state_all[0][n];
	}

	pred_pd_mean[0] = pred_pd_mean_tmp;
	state_pd_mean[0] = state_pd_mean_tmp;
	/*こっからは繰り返し処理*/
	for (t = 1; t < T; t++) {
		/*一期前の(ある意味期前)結果取得*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_pd[n] = state_pd_all[t - 1][n];
			post_weight[n] = weight_state_all[t - 1][n];
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
			resample_check_weight += pow(weight[n], 2);
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
			state_pd_all[t][n] = pred_pd[resample_numbers[n]];
			weight_all[t][n] = weight[n];
			if (check_resample == 0) {
				weight_state_all[t][n] = weight[n];
			}
			else {
				weight_state_all[t][n] = 1.0 / N;
			}
			pred_pd_mean_tmp += pred_pd_all[t][n] * weight_state_all[t][n];
			state_pd_mean_tmp += state_pd_all[t][n] * weight_state_all[t][n];
		}
		pred_pd_mean[t] = pred_pd_mean_tmp;
		state_pd_mean[t] = state_pd_mean_tmp;
	}
}

/*平滑化*/
void particle_smoother(int T, int N, std::vector<std::vector<double>>& weight_state_all, std::vector<std::vector<double>>& state_pd_all, 
	double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,
	std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<double>& state_pd_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight, bunsi_sum, bunbo_sum, state_pd_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T時点のweightは変わらないのでそのまま代入*/
	state_pd_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 1][n] = weight_state_all[T - 1][n];
		state_pd_all_bffs_mean_tmp += state_pd_all[T - 1][n] * weight_state_all_bffs[T - 1][n];
	}
	state_pd_all_bffs_mean[T - 1] = state_pd_all_bffs_mean_tmp;
	for (t = T - 2; t > -1; t--) {
		sum_weight = 0;
		bunbo_sum = 0;
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:bunbo_sum)
			for (n2 = 0; n2 < N; n2++) {
				/*分母計算*/
				bunbo[n][n2] = weight_state_all[t][n2] *
					dnorm(sig_env(state_pd_all[t + 1][n]),
						pd_mu_est + pd_phi_est*(sig_env(state_pd_all[t][n2]) -pd_mu_est),
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
				bunsi[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(sig_env(state_pd_all[t + 1][n2]),
						pd_mu_est + pd_phi_est*(sig_env(state_pd_all[t][n]) - pd_mu_est),
						pd_sd_est);
				bunsi_sum += bunsi[n][n2];
			}
			weight_state_all_bffs[t][n] = weight_state_all[t][n] * bunsi_sum / bunbo_sum;
			sum_weight += weight_state_all_bffs[t][n];
		}
		/*正規化と累積相対尤度の計算*/
		for (n = 0; n < N; n++) {
			weight_state_all_bffs[t][n] = weight_state_all_bffs[t][n] / sum_weight;
			if (n != 0) {
				cumsum_weight[n] = weight_state_all_bffs[t][n] + cumsum_weight[n - 1];
			}
			else {
				cumsum_weight[n] = weight_state_all_bffs[t][n];
			}
		}

		/*平滑化した推定値を計算*/
		state_pd_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			state_pd_all_bffs_mean_tmp += state_pd_all[t][n] * weight_state_all_bffs[t][n];
		}
		state_pd_all_bffs_mean[t] = state_pd_all_bffs_mean_tmp;
	}
}

/*Qの計算に必要な新しいwight*/
void Q_weight_calc(int T, int N, double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est, std::vector<std::vector<double>>& weight_state_all,
	std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<std::vector<double>>& filter_pd, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	int n, n2, t;
	double bunbo;
	for (t = T - 1; t > 0; t--) {
		for (n2 = 0; n2 < N; n2++) {
			bunbo = 0;
#pragma omp parallel for reduction(+:bunbo)
			for (n = 0; n < N; n++) {
				/*分母計算*/
				bunbo += weight_state_all[t - 1][n] * dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est);
			}
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				/*分子計算しつつ代入*/
				Q_weight[t][n][n2] = weight_state_all[t - 1][n] * weight_state_all_bffs[t][n2] *
					dnorm(sig_env(filter_pd[t][n2]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n]) - pd_mu_est), pd_sd_est) / bunbo;
				//printf("%f\n", Q_weight[t + 1][n][n2]);
			}
		}

	}

}


/*EMアルゴリズムで最大化したい式*/
double Q(std::vector<std::vector<double>>& filter_pd, std::vector<std::vector<double>>& weight_state_all_bffs, double pd_phi_est, double pd_mu_est,double pd_sd_est, double pd_0_est, double rho_est,
	std::vector<double>& DR, int T, int N, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;
	int t, n, n2;
#pragma omp parallel for reduction(+:Q_state) reduction(+:Q_obeserve)
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			for (n2 = 0; n2 < N; n2++) {
				Q_state += Q_weight[t][n2][n] * //weight
					log(
						dnorm(sig_env(filter_pd[t][n]), pd_mu_est + pd_phi_est*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est), pd_sd_est)//Xの遷移確率
					);
			}
			Q_obeserve += weight_state_all_bffs[t][n] *//weight
				log(
					g_DR_(DR[t], filter_pd[t][n], rho_est)//観測の確率
				);
		}
	}
#pragma omp parallel for reduction(+:first_state)
	for (n = 0; n < N; n++) {
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				dnorm(sig_env(filter_pd[0][n]), pd_mu_est + pd_phi_est*(pd_0_est - pd_mu_est), pd_sd_est)//初期分布からの確率
			);
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				g_DR_(DR[0], filter_pd[0][n], rho_est)//観測の確率
			);
	}
	printf("%f %f %f\n", Q_state, Q_obeserve, first_state);
	return Q_state + Q_obeserve + first_state;
}


/*Qの最急降下法*/
void Q_grad(int& grad_stop_check, std::vector<std::vector<double >>& filter_pd, std::vector<std::vector<double>>& weight_state_all_bffs,
	double& pd_phi_est, double& pd_mu_est, double& pd_sd_est, double& pd_0_est, double& rho_est,
	std::vector<double>& DR, int T, int N, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	int t, n, n2, l;
	double Now_Q, New_Q, q_qnorm_est_tmp, pd_phi_est_tmp, pd_mu_est_tmp, pd_sd_est_tmp, pd_0_est_tmp, rho_est_tmp,
		log_pd_sd_est, sig_pd_phi_est, sig_rho_est,
		log_pd_sd_est_tmp, sig_pd_phi_est_tmp, sig_rho_est_tmp;
	double phi_grad, mu_grad, sd_grad, zero_grad, rho_grad;
	Now_Q = Q(filter_pd, weight_state_all_bffs, pd_phi_est, pd_mu_est, pd_sd_est, pd_0_est, rho_est, DR, T, N, Q_weight);
	pd_phi_est_tmp = pd_phi_est;
	pd_mu_est_tmp = pd_mu_est;
	pd_sd_est_tmp = pd_sd_est;
	pd_0_est_tmp = pd_0_est;
	rho_est_tmp = rho_est;
	/*制約があるため、ダミー変数を用いる必要がある*/
	log_pd_sd_est = log(pd_sd_est);
	sig_rho_est = sig_env(rho_est);
	log_pd_sd_est_tmp = log_pd_sd_est;
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
					-1 + 1 / exp(2 * log_pd_sd_est)*pow(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)), 2)
					);
			}
		}
	}
#pragma omp parallel for reduction(+:sd_grad)
	for (n = 0; n < N; n++) {
		//sd 初期点からの発生について
		sd_grad += weight_state_all_bffs[0][n] * (
			-1 + 1 / exp(2 * log_pd_sd_est)*pow(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)), 2)
			);
	}

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:phi_grad)
			for (n2 = 0; n2 < N; n2++) {
				//phi 説明変数の式について、phiを対数関数で変換した値の微分
				phi_grad += Q_weight[t][n2][n] * (
					1 / exp(2 * log_pd_sd_est)*(sig_env(filter_pd[t - 1][n2]) - pd_mu_est)*
					(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)))
					);
			}
		}
	}

#pragma omp parallel for reduction(+:phi_grad)
	for (n = 0; n < N; n++) {
		//phi 初期点からの発生について
		phi_grad += weight_state_all_bffs[0][n] * (
			1 / exp(2 * log_pd_sd_est)*(sig_env(filter_pd[0][n]) - pd_mu_est)*
			(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)))
			);
	}

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:mu_grad)
			for (n2 = 0; n2 < N; n2++) {
				//mu 説明変数の式について、phiを対数関数で変換した値の微分
				mu_grad += Q_weight[t][n2][n] * (
					1 / exp(2 * log_pd_sd_est)*(1-pd_phi_est)*(sig_env(filter_pd[t][n]) - (pd_mu_est + pd_phi_est * (sig_env(filter_pd[t - 1][n2]) - pd_mu_est)))
					);
			}
		}
	}

#pragma omp parallel for reduction(+:mu_grad)
	for (n = 0; n < N; n++) {
		//mu 初期点からの発生について
		mu_grad += weight_state_all_bffs[0][n] * (
			1 / exp(2 * log_pd_sd_est)*(1 - pd_phi_est)*(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)))
			);
	}

#pragma omp parallel for reduction(+:zero_grad)
	for (n = 0; n < N; n++) {
		//pd_0 初期点からの発生について
		zero_grad += weight_state_all_bffs[0][n] * (
			1 / (exp(2 * log_pd_sd_est))*pd_phi_est*(sig_env(filter_pd[0][n]) - (pd_mu_est + pd_phi_est * (pd_0_est - pd_mu_est)))
			);
	}

	for (t = 0; t < T; t++) {
#pragma omp parallel for reduction(+:rho_grad)
		//rho 観測式について
		for (n = 0; n < N; n++) {
			rho_grad += weight_state_all_bffs[t][n] * (
				1.0/2.0 *(-1-
					(-exp(-sig_rho_est)*(pow(qnorm(DR[t]),2) + pow(qnorm(filter_pd[t][n]), 2))+
						(exp(-sig_rho_est) + 2*exp(-2*sig_rho_est)) /sqrt(exp(-sig_rho_est)+ exp(-2*sig_rho_est))*qnorm(filter_pd[t][n])*qnorm(DR[t])
						)
					)
				);
		}
	}


	int grad_check = 1;
	l = 1;
	printf("Q %f,phi_grad %f,mu_grad %f ,sd_grad %f,rho_grad %f,0_grad %f \n\n",
		Now_Q,phi_grad, mu_grad ,sd_grad, rho_grad, zero_grad);
	if (sqrt(pow(phi_grad, 2) + pow(mu_grad,2) + pow(rho_grad, 2) + pow(zero_grad, 2) + pow(sd_grad,2) )< 0.01) {
		grad_stop_check = 0;
		grad_check = 0;
	}

	while (grad_check) {
		pd_phi_est = pd_phi_est_tmp;
		pd_mu_est = pd_mu_est_tmp;
		log_pd_sd_est = log_pd_sd_est_tmp;
		pd_0_est = pd_0_est_tmp;
		sig_rho_est = sig_rho_est_tmp;

		pd_phi_est = pd_phi_est + phi_grad * pow(b_grad, l);
		pd_mu_est = pd_mu_est + mu_grad * pow(b_grad, l);
		log_pd_sd_est = log_pd_sd_est + sd_grad * pow(b_grad, l);
		sig_rho_est = sig_rho_est + rho_grad * pow(b_grad, l);
		pd_0_est = pd_0_est + zero_grad * pow(b_grad, l);

		pd_sd_est = exp(log_pd_sd_est);
		rho_est = sig(sig_rho_est);

		New_Q = Q(filter_pd, weight_state_all_bffs, pd_phi_est, pd_mu_est, pd_sd_est, pd_0_est, rho_est, DR, T, N, Q_weight);
		if (Now_Q - New_Q <= -a_grad * pow(b_grad, l) * (pow(phi_grad * pow(b_grad, l), 2)+ pow(mu_grad * pow(b_grad, l), 2) + pow(rho_grad * pow(b_grad, l), 2) + pow(sd_grad * pow(b_grad, l), 2) + pow(zero_grad * pow(b_grad, l), 2))){
			grad_check = 0;
		}
		l += 1;
		printf("%d ", l);
	}

	printf("\n Old Q %f,Now_Q %f\n,ph_est %f, mu_est %f, sd_est %f,rho_est %f,0_est %f \n\n",
		Now_Q, New_Q, pd_phi_est, pd_mu_est,pd_sd_est, rho_est, sig(pd_0_est));

}



int main(void) {
	int n, t, i, j, k, l;
	int N = 1000;
	int T = 100;
	int I = 1000;
	int J = 5;
	double pd_mu_est;
	double pd_phi_est;
	double pd_sd_est;
	double pd_0_est;
	double rho_est;
	/*フィルタリングの結果格納*/
	std::vector<std::vector<double> > filter_pd(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
	std::vector<double> filter_pd_mean(T);
	/*平滑化の結果格納*/
	std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));
	std::vector<double> smoother_pd_mean(T);
	/*Qの計算のためのwight*/
	std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));

	/*Answer格納*/
	std::vector<double> pd(T);
	std::vector<double> DR(T);

	std::vector<double> grad(2);
	double pd_grad = 0;

	/*PD,rho,DRを発生させる*/
	pd[0] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd_0) - sig_env(pd_mu)) + pd_sd * dist(mt));
	DR[0] = reject_sample(pd[0], rho);
	for (t = 1; t < T; t++) {
		pd[t] = sig(sig_env(pd_mu) + pd_phi*(sig_env(pd[t - 1]) - sig_env(pd_mu)) + pd_sd * dist(mt));
		DR[t] = reject_sample(pd[t], rho);
	}
	

	
	pd_mu_est = sig_env(r_rand(mt));//EMでややこしいので，最初から変換しておく
	pd_phi_est = r_rand(mt);
	pd_sd_est = r_rand(mt);
	pd_0_est = sig_env(r_rand(mt));//EMでややこしいので，最初から変換しておく
	rho_est = r_rand(mt);

	/*
	pd_mu_est = sig_env(pd_mu);//EMでややこしいので，最初から変換しておく
	pd_phi_est = pd_phi;
	pd_sd_est = pd_sd;
	pd_0_est = sig_env(pd_0);//EMでややこしいので，最初から変換しておく
	rho_est = rho;
		*/
	
	int grad_stop_check = 1;
	while (grad_stop_check) {
		particle_filter(DR, pd_phi_est, pd_mu_est, pd_sd_est, pd_0_est, rho_est, N, T, filter_pd, filter_weight, filter_pd_mean);
		particle_smoother(T, N, filter_weight, filter_pd, pd_phi_est, pd_mu_est, pd_sd_est, pd_0_est, rho_est, smoother_weight, smoother_pd_mean);
		Q_weight_calc(T, N, pd_phi_est, pd_mu_est,pd_sd_est, pd_0_est, rho_est, filter_weight, smoother_weight, filter_pd, Q_weight);
		Q_grad(grad_stop_check, filter_pd, smoother_weight, pd_phi_est, pd_mu_est, pd_sd_est, pd_0_est, rho_est, DR, T, N, Q_weight);
	}

	

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
	fprintf(gp, "exit\n");    // gnuplotの終了
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


/*gnuplotを用いてDRの密度線確認
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

fprintf(gp, "exit\n");	// gnuplotの終了
_pclose(gp);

*/

/*pd rhoの予測値のanswerのプロット
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

fprintf(gp, "exit\n");	// gnuplotの終了
_pclose(gp);

return 0;
*/
