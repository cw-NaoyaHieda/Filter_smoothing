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
#define pd_sd 0.1
#define rho_sd 0.1
#define pd_0 0.05
#define rho_0 0.1
#define a_grad 0.0001
#define b_grad 0.5
std::mt19937 mt(2017);
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
void particle_filter(std::vector<double>& DR, double pd_sd_est, double rho_sd_est, double pd_0_est, double rho_0_est,
	int N, int T, std::vector<std::vector<double>>& state_pd_all, std::vector<std::vector<double>>& state_rho_all,
	std::vector<std::vector<double>>& weight_state_pd_all, std::vector<std::vector<double>>& weight_state_rho_all,
	std::vector<double>& state_pd_mean, std::vector<double>& state_rho_mean) {
	int n, n2, t;
	double pred_pd_mean_tmp;
	double pred_rho_mean_tmp;
	double state_pd_mean_tmp;
	double state_rho_mean_tmp;
	/*時点tの予測値格納*/
	std::vector<double> pred_pd(N), pred_rho(N), weight_pd(N), weight_rho(N); //XのParticle weight

															/*途中の処理用変数*/
	double sum_weight_pd, sum_weight_rho, resample_check_weight_pd, resample_check_weight_rho, weight_tmp; //正規化因子(weightの合計) リサンプリングの判断基準(正規化尤度の二乗の合計)
	std::vector<double> cumsum_weight_pd(N), cumsum_weight_rho(N); //累積尤度　正規化した上で計算したもの
	std::vector<int> resample_numbers_pd(N), resample_numbers_rho(N); //リサンプリングした結果の番号
	/*リサンプリングしたかどうか*/
	int check_resample_pd, check_resample_rho;

										  /*全期間の推定値格納*/
	std::vector<std::vector<double>> pred_pd_all(T, std::vector<double>(N)); //XのParticle  予測値 XのParticle フィルタリング
	std::vector<std::vector<double>> pred_rho_all(T, std::vector<double>(N));
	std::vector<double> pred_pd_mean(T); //Xの予測値,Xのフィルタリング結果
	std::vector<double> pred_rho_mean(T);
	std::vector<std::vector<double>> weight_rho_all(T, std::vector<double>(N)); // weight 予測値 weight フィルタリング
	std::vector<std::vector<double>> weight_pd_all(T, std::vector<double>(N)); // weight 予測値 weight フィルタリング
																			/*一期前の結果*/
	std::vector<double> post_pd(N), post_rho(N), post_weight_pd(N), post_weight_rho(N);


	double min;
	double max;
	
	/*時点1でのフィルタリング開始*/
	/*初期分布からのサンプリングし、そのまま時点1のサンプリング*/
#pragma omp parallel for
	for (n = 0; n < N; n++) {
		/*初期分布から　時点0と考える*/
		pred_pd[n] = sig(sig_env(pd_0_est) + pd_sd_est * dist(mt));
		pred_rho[n] = sig(sig_env(rho_0_est) + rho_sd_est * dist(mt));
	}


	sum_weight_pd = 0;

	FILE *fp;
	if (fopen_s(&fp, "weight_tmp_pd.csv", "w") != 0) {
	}


			

	
			/*重みの計算*/
	for (n2 = 0; n2 < N; n2++) {
		weight_tmp = 0;
#pragma omp parallel for reduction(+:weight_tmp)
		for (n = 0; n < N; n++) {
			weight_tmp += log(g_DR_fn(DR[0], pred_pd[n2], pred_rho[n]));
		}
		if (weight_tmp > 700) {
			weight_tmp = 700;
		}
		weight_pd[n2] = exp(weight_tmp / N);
		fprintf(fp, "%f,%f\n", weight_tmp, exp(weight_tmp));
		sum_weight_pd += weight_pd[n2];
	}

	fclose(fp);
	if (fopen_s(&fp, "weight_tmp_rho.csv", "w") != 0) {
	}

	sum_weight_rho = 0;
	for (n2 = 0; n2 < N; n2++) {
		weight_tmp = 0;
#pragma omp parallel for reduction(+:weight_tmp)
		for (n = 0; n < N; n++) {
			weight_tmp += log(g_DR_fn(DR[0], pred_pd[n], pred_rho[n2]));
		}
		if (weight_tmp > 700) {
			weight_tmp = 700;
		}
		weight_rho[n2] = exp(weight_tmp / N);
		fprintf(fp, "%f,%f\n", weight_tmp, exp(weight_tmp));
		sum_weight_rho += weight_rho[n2];
	}


	fclose(fp);
	printf("%f\n%f", sum_weight_pd, sum_weight_rho);

	

	/*重みを正規化しながら、リサンプリング判断用変数の計算と累積尤度の計算*/

	resample_check_weight_pd = 0;
	for (n = 0; n < N; n++) {
		weight_pd[n] = weight_pd[n] / sum_weight_pd;
		resample_check_weight_pd += pow(weight_pd[n], 2);
		if (n != 0) {
			cumsum_weight_pd[n] = weight_pd[n] + cumsum_weight_pd[n - 1];
		}
		else {
			cumsum_weight_pd[n] = weight_pd[n];
		}
	}

	resample_check_weight_rho = 0;
	for (n = 0; n < N; n++) {
		weight_rho[n] = weight_rho[n] / sum_weight_pd;
		resample_check_weight_rho += pow(weight_rho[n], 2);
		if (n != 0) {
			cumsum_weight_rho[n] = weight_rho[n] + cumsum_weight_rho[n - 1];
		}
		else {
			cumsum_weight_rho[n] = weight_rho[n];
		}
	}


	/*リサンプリングが必要かどうか判断したうえで必要ならリサンプリング 必要ない場合は順番に数字を入れる*/
	if (1 / resample_check_weight_pd < N / 10) {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers_pd[n] = resample(cumsum_weight_pd, N, (r_rand(mt) + n - 1) / N);
		}
		check_resample_pd = 1;
	}
	else {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers_pd[n] = n;
		}
		check_resample_pd = 0;
	}

	if (1 / resample_check_weight_rho < N / 10) {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers_rho[n] = resample(cumsum_weight_rho, N, (r_rand(mt) + n - 1) / N);
		}
		check_resample_rho = 1;
	}
	else {
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			resample_numbers_rho[n] = n;
		}
		check_resample_rho = 0;
	}



	/*結果の格納*/
	pred_pd_mean_tmp = 0;
	pred_rho_mean_tmp = 0;
	state_pd_mean_tmp = 0;
	state_rho_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_pd_mean_tmp) reduction(+:state_pd_mean_tmp) reduction(+:pred_rho_mean_tmp) reduction(+:state_rho_mean_tmp)
	for (n = 0; n < N; n++) {
		pred_pd_all[0][n] = pred_pd[n];
		pred_rho_all[0][n] = pred_rho[n];
		state_pd_all[0][n] = pred_pd[resample_numbers_pd[n]];
		state_rho_all[0][n] = pred_rho[resample_numbers_rho[n]];
		weight_pd_all[0][n] = weight_pd[n];
		weight_rho_all[0][n] = weight_rho[n];
		if (check_resample_pd == 0) {
			weight_state_pd_all[0][n] = weight_pd[n];
		}
		else {
			weight_state_pd_all[0][n] = 1.0 / N;
		}
		if (check_resample_pd == 0) {
			weight_state_rho_all[0][n] = weight_rho[n];
		}
		else {
			weight_state_rho_all[0][n] = 1.0 / N;
		}
		pred_pd_mean_tmp += pred_pd_all[0][n] * 1.0 / N;
		state_pd_mean_tmp += state_pd_all[0][n] * weight_state_pd_all[0][n];
		pred_rho_mean_tmp += pred_rho_all[0][n] * 1.0 / N;
		state_rho_mean_tmp += state_rho_all[0][n] * weight_state_rho_all[0][n];
	}

	pred_pd_mean[0] = pred_pd_mean_tmp;
	state_pd_mean[0] = state_pd_mean_tmp;
	pred_rho_mean[0] = pred_rho_mean_tmp;
	state_rho_mean[0] = state_rho_mean_tmp;
	/*こっからは繰り返し処理*/
	for (t = 1; t < T; t++) {
		/*一期前の(ある意味期前)結果取得*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_pd[n] = state_pd_all[t - 1][n];
			post_rho[n] = state_rho_all[t - 1][n];
			post_weight_pd[n] = weight_state_pd_all[t - 1][n];
			post_weight_rho[n] = weight_state_rho_all[t - 1][n];
		}
		/*時点tのサンプリング*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			pred_pd[n] = sig(sig_env(post_pd[n]) + pd_sd_est * dist(mt));
			pred_rho[n] = sig(sig_env(post_rho[n]) + rho_sd_est * dist(mt));
		}

		
		/*重みの計算*/
		sum_weight_pd = 0;
		for (n2 = 0; n2 < N; n2++) {
			weight_tmp = 0;
#pragma omp parallel for reduction(*:weight_tmp)
			for (n = 0; n < N; n++) {
				weight_tmp *= g_DR_fn(DR[t], pred_pd[n2], pred_rho[n]) * post_weight_rho[n];
			}
			weight_pd[n2] = weight_tmp * post_weight_pd[n2];
			sum_weight_pd += weight_pd[n2];
		}

		sum_weight_rho = 0;
		for (n2 = 0; n2 < N; n2++) {
			weight_tmp = 0;
#pragma omp parallel for reduction(*:weight_tmp)
			for (n = 0; n < N; n++) {
				weight_tmp += g_DR_fn(DR[t], pred_pd[n], pred_rho[n2]) * post_weight_pd[n];
			}
			weight_rho[n2] = exp(weight_tmp)* post_weight_rho[n2];
			sum_weight_rho += weight_rho[n2];
		}


		resample_check_weight_pd = 0;
		for (n = 0; n < N; n++) {
			weight_pd[n] = weight_pd[n] / sum_weight_pd;
			resample_check_weight_pd += pow(weight_pd[n], 2);
			if (n != 0) {
				cumsum_weight_pd[n] = weight_pd[n] + cumsum_weight_pd[n - 1];
			}
			else {
				cumsum_weight_pd[n] = weight_pd[n];
			}
		}

		resample_check_weight_rho = 0;
		for (n = 0; n < N; n++) {
			weight_rho[n] = weight_rho[n] / sum_weight_rho;
			resample_check_weight_rho += pow(weight_rho[n], 2);
			if (n != 0) {
				cumsum_weight_rho[n] = weight_rho[n] + cumsum_weight_rho[n - 1];
			}
			else {
				cumsum_weight_rho[n] = weight_rho[n];
			}
		}

		/*リサンプリングが必要かどうか判断したうえで必要ならリサンプリング 必要ない場合は順番に数字を入れる*/
		if (1 / resample_check_weight_pd < N / 10) {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers_pd[n] = resample(cumsum_weight_pd, N, (r_rand(mt) + n - 1) / N);
			}
			check_resample_pd = 1;
		}
		else {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers_pd[n] = n;
			}
			check_resample_pd = 0;
		}

		if (1 / resample_check_weight_rho < N / 10) {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers_rho[n] = resample(cumsum_weight_rho, N, (r_rand(mt) + n - 1) / N);
			}
			check_resample_rho = 1;
		}
		else {
#pragma omp parallel for
			for (n = 0; n < N; n++) {
				resample_numbers_rho[n] = n;
			}
			check_resample_rho = 0;
		}

		/*結果の格納*/
		pred_pd_mean_tmp = 0;
		state_pd_mean_tmp = 0;
		pred_rho_mean_tmp = 0;
		state_rho_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_pd_mean_tmp) reduction(+:pred_rho_mean_tmp) reduction(+:state_pd_mean_tmp) reduction(+:state_rho_mean_tmp)
		for (n = 0; n < N; n++) {
			pred_pd_all[t][n] = pred_pd[n];
			pred_rho_all[t][n] = pred_rho[n];
			state_pd_all[t][n] = pred_pd[resample_numbers_pd[n]];
			state_rho_all[t][n] = pred_rho[resample_numbers_rho[n]];
			weight_pd_all[t][n] = weight_pd[n];
			weight_rho_all[t][n] = weight_rho[n];
			if (check_resample_pd == 0) {
				weight_state_pd_all[t][n] = weight_pd[n];
			}
			else {
				weight_state_pd_all[t][n] = 1.0 / N;
			}
			if (check_resample_rho == 0) {
				weight_state_rho_all[t][n] = weight_rho[n];
			}
			else {
				weight_state_rho_all[t][n] = 1.0 / N;
			}
			pred_pd_mean_tmp += pred_pd_all[t][n] * weight_state_pd_all[t][n];
			state_pd_mean_tmp += state_pd_all[t][n] * weight_state_pd_all[t][n];
			pred_rho_mean_tmp += pred_rho_all[t][n] * weight_state_rho_all[t][n];
			state_rho_mean_tmp += state_rho_all[t][n] * weight_state_rho_all[t][n];
		}
		pred_pd_mean[t] = pred_pd_mean_tmp;
		state_pd_mean[t] = state_pd_mean_tmp;
		pred_rho_mean[t] = pred_rho_mean_tmp;
		state_rho_mean[t] = state_rho_mean_tmp;
	}
}

/*平滑化*/
void particle_smoother(int T, int N, std::vector<std::vector<double>>& weight_state_all, std::vector<std::vector<double>>& state_pd_all, std::vector<std::vector<double>>& state_rho_all,
	double pd_beta_est, double rho_beta_est,
	std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<double>& state_pd_all_bffs_mean, std::vector<double>& state_rho_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight, bunsi_sum, bunbo_sum, state_pd_all_bffs_mean_tmp, state_rho_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T時点のweightは変わらないのでそのまま代入*/
	state_pd_all_bffs_mean_tmp = 0;
	state_rho_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp) reduction(+:state_rho_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 1][n] = weight_state_all[T - 1][n];
		state_pd_all_bffs_mean_tmp += state_pd_all[T - 1][n] * weight_state_all_bffs[T - 1][n];
		state_rho_all_bffs_mean_tmp += state_rho_all[T - 1][n] * weight_state_all_bffs[T - 1][n];
	}
	state_pd_all_bffs_mean[T - 1] = state_pd_all_bffs_mean_tmp;
	state_rho_all_bffs_mean[T - 1] = state_rho_all_bffs_mean_tmp;
	for (t = T - 2; t > -1; t--) {
		sum_weight = 0;
		bunbo_sum = 0;
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:bunbo_sum)
			for (n2 = 0; n2 < N; n2++) {
				/*分母計算*/
				bunbo[n][n2] = weight_state_all[t][n2] *
					dnorm(sig_env(state_pd_all[t + 1][n]),
						sqrt(pd_beta_est) * sig_env(state_pd_all[t][n2]),
						sqrt(1 - pd_beta_est))*
					dnorm(sig_env(state_rho_all[t + 1][n]),
						sqrt(rho_beta_est) * sig_env(state_rho_all[t][n2]),
						sqrt(1 - rho_beta_est));
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
						sqrt(pd_beta_est) *  sig_env(state_pd_all[t][n]),
						sqrt(1 - pd_beta_est))*
						dnorm(sig_env(state_rho_all[t + 1][n2]),
							sqrt(rho_beta_est) *  sig_env(state_rho_all[t][n]),
							sqrt(1 - rho_beta_est));
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
		state_rho_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_pd_all_bffs_mean_tmp) reduction(+:state_rho_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			state_pd_all_bffs_mean_tmp += state_pd_all[t][n] * weight_state_all_bffs[t][n];
			state_rho_all_bffs_mean_tmp += state_rho_all[t][n] * weight_state_all_bffs[t][n];
		}
		state_pd_all_bffs_mean[t] = state_pd_all_bffs_mean_tmp;
		state_rho_all_bffs_mean[t] = state_rho_all_bffs_mean_tmp;
	}
}

/*Qの計算に必要な新しいwight*/
void Q_weight_calc(int T, int N, double beta_est, std::vector<std::vector<double>>& weight_state_all,
	std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<std::vector<double>>& state_X_all, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	int n, n2, t;
	double bunbo;
	for (t = T - 3; t > -1; t--) {
		for (n2 = 0; n2 < N; n2++) {
			bunbo = 0;
#pragma omp parallel for reduction(+:bunbo)
			for (n = 0; n < N; n++) {
				/*分母計算*/
				bunbo += weight_state_all[t][n] * dnorm(state_X_all[t + 1][n2], sqrt(beta_est) * state_X_all[t][n], sqrt(1 - beta_est));
			}

#pragma omp parallel for
			for (n = 0; n < N; n++) {
				/*分子計算しつつ代入*/
				Q_weight[t + 1][n][n2] = weight_state_all[t][n] * weight_state_all_bffs[t + 1][n2] *
					dnorm(state_X_all[t + 1][n2], sqrt(beta_est) * state_X_all[t][n], sqrt(1 - beta_est)) / bunbo;
			}
		}

	}

}


int main(void) {
	int n, t, i, j, k, l;
	int N = 1000;
	int T = 100;
	int I = 1000;
	int J = 5;
	double pd_sd_est;
	double rho_sd_est;
	double pd_0_est;
	double rho_0_est;
	/*フィルタリングの結果格納*/
	std::vector<std::vector<double> > filter_pd(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_rho(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_pd_weight(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_rho_weight(T, std::vector<double>(N));
	std::vector<double> filter_pd_mean(T);
	std::vector<double> filter_rho_mean(T);
	/*平滑化の結果格納*/
	std::vector<std::vector<double> > smoother_pd_weight(T, std::vector<double>(N));
	std::vector<std::vector<double> > smoother_rho_weight(T, std::vector<double>(N));
	std::vector<double> smoother_pd_mean(T);
	std::vector<double> smoother_rho_mean(T);
	/*Qの計算のためのwight*/
	std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));

	/*Answer格納*/
	std::vector<double> pd(T);
	std::vector<double> rho(T);
	std::vector<double> DR(T);

	std::vector<double> grad(2);
	double pd_grad = 0, rho_grad = 0;

	/*PD,rho,DRを発生させる*/
	pd[0] = sig(sig_env(pd_0) + pd_sd * dist(mt));
	rho[0] = sig(sig_env(rho_0) + rho_sd * dist(mt));
	DR[0] = reject_sample(pd[0], rho[0]);
	for (t = 1; t < T; t++) {
		pd[t] = sig(sig_env(pd[t - 1]) + pd_sd * dist(mt));
		rho[t] = sig(sig_env(rho[t - 1]) + rho_sd * dist(mt));
		DR[t] = reject_sample(pd[t], rho[t]);
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

	pd_sd_est = pd_sd;
	rho_sd_est = rho_sd;
	pd_0_est = pd_0;
	rho_0_est = rho_0;
		
	particle_filter(DR, pd_sd_est, rho_sd_est, pd_0_est, rho_0_est, N, T, filter_pd, filter_rho, filter_pd_weight, filter_rho_weight, filter_pd_mean,filter_rho_mean);
	//particle_smoother(T, N, filter_weight, filter_pd, filter_rho, pd_beta_est, rho_beta_est,smoother_weight, smoother_pd_mean, smoother_rho_mean);
	


	/*
	for (j = 1; j < 5; j++) {
	particle_filter(DR, beta_est, q_qnorm_est, rho_est, X_0_est, N, T, filter_X, filter_weight, filter_X_mean);
	particle_smoother(T, N, filter_weight, filter_X, beta_est, smoother_weight, smoother_X_mean);
	Q_weight_calc(T, N, beta_est, filter_weight, smoother_weight, filter_X, Q_weight);
	for (i = 1; i < I; i++) {
	fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", i,
	Q(filter_X, smoother_weight, i / double(I) - 0.0001, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight),
	Q(filter_X, smoother_weight, beta_est, i / double(I) - 0.0001, q_qnorm_est, X_0_est, DR, T, N, Q_weight),
	Q(filter_X, smoother_weight, beta_est, rho_est, (i - 500) / double(100), X_0_est, DR, T, N, Q_weight),
	Q(filter_X, smoother_weight, beta_est, rho_est, q_qnorm_est, (i - 500) / double(100), DR, T, N, Q_weight),
	(i - 500) / double(100),
	Q_grad_beta(filter_X, smoother_weight, i / double(I) - 0.0001, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight),
	Q_grad_rho(filter_X, smoother_weight, beta_est, i / double(I) - 0.0001, q_qnorm_est, X_0_est, DR, T, N, Q_weight),
	Q_grad_q_qnorm(filter_X, smoother_weight, beta_est, rho_est, (i - 500) / double(100), X_0_est, DR, T, N, Q_weight),
	Q_grad_X_0(filter_X, smoother_weight, beta_est, rho_est, q_qnorm_est, (i - 500) / double(100), DR, T, N, Q_weight));
	}
	}
	*/

	FILE *fp;
	if (fopen_s(&fp, "particle_hull.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f,%f,%f,%f\n", t, filter_pd[t][n], filter_rho[t][n],
				filter_pd_weight[t][n], N / 20 * filter_pd_weight[t][n],
				filter_rho_weight[t][n], N / 20 * filter_rho_weight[t][n]);

		}
	}
	fclose(fp);

	if (fopen_s(&fp, "X_hull.csv", "w") != 0) {
		return 0;
	}
	for (t = 0; t < T - 1; t++) {
		fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, pd[t], rho[t],filter_pd_mean[t], filter_rho_mean[t], DR[t]);
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
	fprintf(gp, "plot 'particle_hull.csv' using 1:2:5:4 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp);
	fprintf(gp, "replot 'X_hull.csv' using 1:2 with lines linetype 1 lw 4 linecolor rgb '#ff0000 ' title 'Answer_PD'\n");
	fflush(gp);
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "replot 'X_hull.csv' using 1:4 with lines linetype 1 lw 4 linecolor rgb '#ffff00 ' title 'Filter'\n");
	fflush(gp);
	fprintf(gp, "replot 'X_hull.csv' using 1:6 with lines linetype 1 lw 4 linecolor rgb '#ffffff ' title 'DR'\n");
	fflush(gp);
	//fprintf(gp, "replot 'X_hull.csv' using 1:6 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
	//fflush(gp);
	//fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'Predict'\n");
	//fflush(gp);

	gp3 = _popen(GNUPLOT_PATH, "w");

	//fprintf(gp, "set term postscript eps color\n");
	//fprintf(gp, "set term pdfcairo enhanced size 12in, 9in\n");
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp3, "reset\n");
	fprintf(gp3, "set datafile separator ','\n");
	fprintf(gp3, "set grid lc rgb 'white' lt 2\n");
	fprintf(gp3, "set border lc rgb 'white'\n");
	fprintf(gp3, "set border lc rgb 'white'\n");
	fprintf(gp3, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
	fprintf(gp3, "set palette rgbformulae 22, 13, -31\n");
	fprintf(gp3, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
	fprintf(gp3, "set object 1 rect fc rgb '#333333 ' fillstyle solid 1.0 \n");
	fprintf(gp3, "set key textcolor rgb 'white'\n");
	fprintf(gp3, "set size ratio 1/3\n");
	fprintf(gp3, "plot 'particle_hull.csv' using 1:3:7:6 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp3);
	fprintf(gp3, "replot 'X_hull.csv' using 1:3 with lines linetype 1 lw 4 linecolor rgb '#ff0000 ' title 'Answer_rho'\n");
	fflush(gp3);
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp3, "replot 'X_hull.csv' using 1:5 with lines linetype 1 lw 4 linecolor rgb '#ffff00 ' title 'Filter'\n");
	fflush(gp3);
	//fprintf(gp3, "replot 'X_hull.csv' using 1:7 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
	//fflush(gp3);
	//fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'Predict'\n");
	//fflush(gp);


	


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
