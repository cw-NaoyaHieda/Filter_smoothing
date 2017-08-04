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
#define beta 0.75
#define q_qnorm -2.053749 //qに直したときに、約0.02
#define rho 0.05
#define X_0 -2.5
#define a_grad 0.0001
#define b_grad 0.5

std::mt19937 mt(130);
std::uniform_real_distribution<double> r_rand(0.0, 1.0);
std::uniform_real_distribution<double> r_rand_choice(0.0, 4.0);


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
void particle_filter(std::vector<double>& DR, double beta_est, double q_qnorm_est, double rho_est, double X_0_est, int N, int T,
	std::vector<std::vector<double>>& state_X_all, std::vector<std::vector<double>>& weight_state_all, std::vector<double>& state_X_mean, std::vector<double>& predict_Y_mean) {
	int n;
	int t;
	double pred_X_mean_tmp;
	double state_X_mean_tmp;
	double predict_Y_mean_tmp;
	/*時点tの予測値格納*/
	std::vector<double> pred_X(N), weight(N); //XのParticle weight

	/*途中の処理用変数*/
	double sum_weight, resample_check_weight; //正規化因子(weightの合計) リサンプリングの判断基準(正規化尤度の二乗の合計)
	std::vector<double> cumsum_weight(N); //累積尤度　正規化した上で計算したもの
	std::vector<int> resample_numbers(N); //リサンプリングした結果の番号
	int check_resample; //リサンプリングしたかどうかの変数 0ならしてない、1ならしてる

	/*全期間の推定値格納*/
	std::vector<std::vector<double>> pred_X_all(T, std::vector<double>(N)); //XのParticle  予測値 XのParticle フィルタリング
	std::vector<double> pred_X_mean(T); //Xの予測値,Xのフィルタリング結果
	std::vector<std::vector<double>> weight_all(T, std::vector<double>(N)); // weight 予測値 weight フィルタリング

	/*一期前の結果*/
	std::vector<double> post_X(N), post_weight(N);

	/*時点1でのフィルタリング開始*/
	/*初期分布からのサンプリングし、そのまま時点1のサンプリング*/
#pragma omp parallel for
	for (n = 0; n < N; n++) {
		/*初期分布から　時点0と考える*/
		pred_X[n] = sqrt(beta_est)*X_0_est - sqrt(1 - beta_est) * rnorm(0, 1);
	}

	/*重みの計算*/
	sum_weight = 0;
	resample_check_weight = 0;
#pragma omp parallel for reduction(+:sum_weight)
	for (n = 0; n < N; n++) {
		weight[n] = g_DR_dinamic(DR[1], pred_X[n], q_qnorm_est, beta_est, rho_est);
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
	pred_X_mean_tmp = 0;
	state_X_mean_tmp = 0;
	predict_Y_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp) reduction(+:predict_Y_mean_tmp)
	for (n = 0; n < N; n++) {
		pred_X_all[0][n] = pred_X[n];
		state_X_all[0][n] = pred_X[resample_numbers[n]];
		weight_all[0][n] = weight[n];
		predict_Y_mean_tmp += weight_all[0][n] * r_DDR(state_X_all[0][n], q_qnorm_est, rho_est, beta_est);
		if (check_resample == 0) {
			weight_state_all[0][n] = weight[n];
		}
		else {
			weight_state_all[0][n] = 1.0 / N;
		}
		pred_X_mean_tmp += pred_X_all[0][n] * 1.0 / N;
		state_X_mean_tmp += state_X_all[0][n] * weight_state_all[0][n];
	}
	predict_Y_mean[1] = predict_Y_mean_tmp;

	pred_X_mean[0] = pred_X_mean_tmp;
	state_X_mean[0] = state_X_mean_tmp;
	/*こっからは繰り返し処理*/
	for (t = 2; t < T; t++) {
		/*一期前の(ある意味期前)結果取得*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			post_X[n] = state_X_all[t - 2][n];
			post_weight[n] = weight_state_all[t - 2][n];
		}
		/*時点tのサンプリング*/
#pragma omp parallel for
		for (n = 0; n < N; n++) {
			pred_X[n] = sqrt(beta_est)*post_X[n] - sqrt(1 - beta_est)*rnorm(0, 1);
		}

		/*重みの計算*/
		sum_weight = 0.0;
		resample_check_weight = 0.0;
#pragma omp parallel for reduction(+:sum_weight)
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_dinamic(DR[t], pred_X[n], q_qnorm_est, beta_est, rho_est) * post_weight[n];
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

		/*結果の格納*/
		pred_X_mean_tmp = 0;
		state_X_mean_tmp = 0;
		predict_Y_mean_tmp = 0;
#pragma omp parallel for reduction(+:pred_X_mean_tmp) reduction(+:state_X_mean_tmp) reduction(+:predict_Y_mean_tmp)
		for (n = 0; n < N; n++) {
			pred_X_all[t - 1][n] = pred_X[n];
			state_X_all[t - 1][n] = pred_X[resample_numbers[n]];
			weight_all[t - 1][n] = weight[n];
			predict_Y_mean_tmp += weight_all[t - 1][n] * r_DDR(state_X_all[t - 1][n], q_qnorm_est, rho_est, beta_est);
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
		predict_Y_mean[t] = predict_Y_mean_tmp;
	}
}

/*平滑化*/
void particle_smoother(int T, int N, std::vector<std::vector<double>>& weight_state_all, std::vector<std::vector<double>>& state_X_all, double beta_est,
	std::vector<std::vector<double>>& weight_state_all_bffs, std::vector<double>& state_X_all_bffs_mean) {
	int n, n2, t, check_resample;
	double sum_weight, bunsi_sum, bunbo_sum, state_X_all_bffs_mean_tmp;
	std::vector<std::vector<double>> bunsi(N, std::vector<double>(N)), bunbo(N, std::vector<double>(N));
	std::vector<double> cumsum_weight(N);
	/*T時点のweightは変わらないのでそのまま代入*/
	state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
	for (n = 0; n < N; n++) {
		weight_state_all_bffs[T - 2][n] = weight_state_all[T - 2][n];
		state_X_all_bffs_mean_tmp += state_X_all[T - 2][n] * weight_state_all_bffs[T - 2][n];
	}
	state_X_all_bffs_mean[T - 2] = state_X_all_bffs_mean_tmp;
	for (t = T - 3; t > -1; t--) {
		sum_weight = 0;
		bunbo_sum = 0;
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:bunbo_sum)
			for (n2 = 0; n2 < N; n2++) {
				/*分母計算*/
				bunbo[n][n2] = weight_state_all[t][n2] *
					dnorm(state_X_all[t + 1][n],
						sqrt(beta_est) * state_X_all[t][n2],
						sqrt(1 - beta_est));
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
					dnorm(state_X_all[t + 1][n2],
						sqrt(beta_est) *  state_X_all[t][n],
						sqrt(1 - beta_est));
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
		state_X_all_bffs_mean_tmp = 0;
#pragma omp parallel for reduction(+:state_X_all_bffs_mean_tmp)
		for (n = 0; n < N; n++) {
			state_X_all_bffs_mean_tmp += state_X_all[t][n] * weight_state_all_bffs[t][n];
		}

		state_X_all_bffs_mean[t] = state_X_all_bffs_mean_tmp;

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


/*EMアルゴリズムで最大化したい式*/
double Q(std::vector<std::vector<double>>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs, double beta_est, double rho_est, double q_qnorm_est, double X_0_est,
	std::vector<double>& DR, int T, int N, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	double Q_state = 0, Q_obeserve = 0, first_state = 0;
	int t, n, n2;
#pragma omp parallel for reduction(+:Q_state) reduction(+:Q_obeserve)
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			for (n2 = 0; n2 < N; n2++) {
				Q_state += Q_weight[t][n2][n] * //weight
					log(
						dnorm(state_X_all_bffs[t][n], sqrt(beta_est)*state_X_all_bffs[t - 1][n2], sqrt(1 - beta_est))//Xの遷移確率
					);
			}
			Q_obeserve += weight_state_all_bffs[t - 1][n] *//weight
				log(
					g_DR_dinamic(DR[t], state_X_all_bffs[t - 1][n], q_qnorm_est, beta_est, rho_est)//観測の確率
				);
		}
	}
#pragma omp parallel for reduction(+:first_state)
	for (n = 0; n < N; n++) {
		first_state += weight_state_all_bffs[0][n] *//weight
			log(
				dnorm(state_X_all_bffs[0][n], sqrt(beta_est) * X_0_est, sqrt(1 - beta_est))//初期分布からの確率
			);
	}
	return Q_state + Q_obeserve + first_state;
}


/*Qの最急降下法*/
void Q_grad(int& grad_stop_check,std::vector<std::vector<double >>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs,
	double& beta_est, double& rho_est, double& q_qnorm_est, double& X_0_est,
	std::vector<double>& DR, int T, int N, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	int t, n, n2, l;
	double Now_Q, q_qnorm_est_tmp, beta_est_tmp, rho_est_tmp, X_0_est_tmp, sig_beta_est, sig_rho_est, sig_beta_est_tmp, sig_rho_est_tmp;
	double beta_grad, rho_grad, q_qnorm_grad, X_0_grad;
	Now_Q = Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est,
		 DR, T, N, Q_weight);
	beta_est_tmp = beta_est;
	rho_est_tmp = rho_est;
	q_qnorm_est_tmp = q_qnorm_est;
	X_0_est_tmp = X_0_est;
	/*betaとrhoは[0,1]制約があるため、ダミー変数を用いる必要がある*/
	sig_beta_est = sig_env(beta_est);
	sig_rho_est = sig_env(rho_est);
	sig_beta_est_tmp = sig_beta_est;
	sig_rho_est_tmp = sig_rho_est;

	beta_grad = 0;
	rho_grad = 0;
	q_qnorm_grad = 0;
	X_0_grad = 0;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
#pragma omp parallel for reduction(+:beta_grad)
			for (n2 = 0; n2 < N; n2++) {
				//beta 説明変数の式について、betaをシグモイド関数で変換した値の微分
				beta_grad += Q_weight[t][n2][n] * (
					-exp(sig_beta_est) / 2 * (((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2] + pow(state_X_all_bffs[t - 1][n2], 2))) -
					exp(sig_beta_est) / 2 * ((-exp(-sig_beta_est) *pow(state_X_all_bffs[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2])) +
					exp(sig_beta_est) / (2 + 2 * exp(sig_beta_est))
					);
			}
		}
	}
	for (t = 1; t < T; t++) {
#pragma omp parallel for reduction(+:beta_grad)
		for (n = 0; n < N; n++) {
			//次は観測変数について
			beta_grad += weight_state_all_bffs[t - 1][n] * (
				exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
				(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(-sig_beta_est + sig_rho_est) / pow((1 + exp(-sig_beta_est)), 2)*pow(state_X_all_bffs[t - 1][n], 2) - sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 1.5)*q_qnorm_est*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 1.5) * state_X_all_bffs[t - 1][n]))
				);
		}
	}
#pragma omp parallel for reduction(+:beta_grad)
	for (n = 0; n < N; n++) {
		//最後は初期点からの発生について
		beta_grad += weight_state_all_bffs[0][n] * (
			-exp(sig_beta_est) / 2 * ((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[0][n] * X_0_est + pow(X_0_est, 2)) -
			exp(sig_beta_est) / 2 * ((-exp(-sig_beta_est) * pow(state_X_all_bffs[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] * X_0_est)) +
			exp(sig_beta_est) / (2 + 2 * exp(sig_beta_est))
			);
	}

	for (t = 1; t < T; t++) {
#pragma omp parallel for reduction(+:rho_grad)
		for (n = 0; n < N; n++) {
			rho_grad += weight_state_all_bffs[t - 1][n] * (
				-1.0 / 2.0 +
				((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) -
						2 * sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*q_qnorm_est * state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) -
				(exp(sig_rho_est) + 2 * exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
					DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))
				);
		}
	}

	for (t = 1; t < T; t++) {
#pragma omp parallel for reduction(+:q_qnorm_grad)
		for (n = 0; n < N; n++) {
			q_qnorm_grad += weight_state_all_bffs[t - 1][n] * (
				(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
				(-(1 + exp(sig_rho_est))*q_qnorm_est +
					sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n] +
					DR[t] * sqrt(1 + exp(sig_rho_est)))
				);
		}
	}

#pragma omp parallel for reduction(+:X_0_grad)
	for (n = 0; n < N; n++) {
		//X_0 説明変数について
		X_0_grad += weight_state_all_bffs[0][n] * (
			exp(sig_beta_est) * (sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] - X_0_est)
			);
	}



	int grad_check = 1;
	l = 1;
	printf("beta_grad %f,rho_grad %f,q_grad %f X_0_grad %f\n\n",
		beta_grad, rho_grad, q_qnorm_grad, X_0_grad);
	if (sqrt(pow(beta_grad,2)+pow(rho_grad, 2)+pow(q_qnorm_grad,2)+pow(X_0_grad,2))< 40 ){
		grad_stop_check = 0;
		grad_check = 0;
	}

	while (grad_check) {
		sig_beta_est = sig_beta_est_tmp;
		sig_rho_est = sig_rho_est_tmp;
		q_qnorm_est = q_qnorm_est_tmp;
		X_0_est = X_0_est_tmp;
		sig_beta_est = sig_beta_est + beta_grad * pow(b_grad, l);
		sig_rho_est = sig_rho_est + rho_grad * pow(b_grad, l);
		q_qnorm_est = q_qnorm_est + q_qnorm_grad * pow(b_grad, l);
		X_0_est = X_0_est + X_0_grad * pow(b_grad, l);
		beta_est = sig(sig_beta_est);
		rho_est = sig(sig_rho_est);
		if (Now_Q - Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight) <= -a_grad*pow(b_grad,l)*pow(beta_grad * pow(b_grad, l), 2) + pow(rho_grad * pow(b_grad, l), 2) + pow(q_qnorm_grad * pow(b_grad, l), 2) + pow(X_0_grad * pow(b_grad, l), 2)) {
			grad_check = 0;
		}
		l += 1;
		printf("%d ", l);
	}

	printf("\n Old Q %f,Now_Q %f\n,beta_est %f,rho_est %f,q %f X_0_est %f\n\n",
		Now_Q, Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight), beta_est, rho_est, pnorm(q_qnorm_est, 0, 1), X_0_est);

}


void Q_choice_grad(int& grad_stop_check, std::vector<std::vector<double >>& state_X_all_bffs, std::vector<std::vector<double>>& weight_state_all_bffs,
	double& beta_est, double& rho_est, double& q_qnorm_est, double& X_0_est,
	std::vector<double>& DR, int T, int N, std::vector<std::vector<std::vector<double>>>& Q_weight) {
	int t, n, n2, l, choice;
	double Now_Q, q_qnorm_est_tmp, beta_est_tmp, rho_est_tmp, X_0_est_tmp, sig_beta_est, sig_rho_est, sig_beta_est_tmp, sig_rho_est_tmp;
	double beta_grad, rho_grad, q_qnorm_grad, X_0_grad;
	Now_Q = Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est,
		DR, T, N, Q_weight);
	beta_est_tmp = beta_est;
	rho_est_tmp = rho_est;
	q_qnorm_est_tmp = q_qnorm_est;
	X_0_est_tmp = X_0_est;
	/*betaとrhoは[0,1]制約があるため、ダミー変数を用いる必要がある*/
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
				//beta 説明変数の式について、betaをシグモイド関数で変換した値の微分
				beta_grad += weight_state_all_bffs[t][n] * weight_state_all_bffs[t - 1][n2] * exp(sig_beta_est) / 2 * (
					-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2] + pow(state_X_all_bffs[t - 1][n2], 2))) -
					((-exp(-sig_beta_est) *pow(state_X_all_bffs[t][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[t][n] * state_X_all_bffs[t - 1][n2])) +
					1 / (1 + exp(sig_beta_est))
					);
			}

			//次は観測変数について
			beta_grad += weight_state_all_bffs[t - 1][n] * (
				exp(sig_beta_est) / (2 * (1 + exp(sig_beta_est))) -
				(exp(sig_beta_est) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est*state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(-sig_beta_est + sig_rho_est) / pow((1 + exp(-sig_beta_est)), 2)*pow(state_X_all_bffs[t - 1][n], 2) - sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) * exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2)*q_qnorm_est*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(exp(sig_rho_est))*exp(-sig_beta_est) / pow(1 + exp(-sig_beta_est), 3 / 2) * state_X_all_bffs[t - 1][n]))
				);

			//最後は初期点からの発生について
			beta_grad += weight_state_all_bffs[0][n] * exp(sig_beta_est) / 2 * (
				-(((1 + exp(-sig_beta_est))*pow(state_X_all_bffs[0][n], 2) - 2 * sqrt(1 + exp(-sig_beta_est)) * state_X_all_bffs[0][n] * X_0_est + pow(X_0_est, 2))) -
				((-exp(-sig_beta_est) * pow(state_X_all_bffs[0][n], 2) + exp(-sig_beta_est) / sqrt(1 + exp(-sig_beta_est))*state_X_all_bffs[0][n] * X_0_est)) +
				1 / (1 + exp(sig_beta_est))
				);

			//rho 観測変数について rhoをシグモイド関数で変換した値の微分
			rho_grad += weight_state_all_bffs[t - 1][n] * (
				-1 / 2 +
				((1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				(pow(DR[t], 2) +
					((1 + exp(sig_rho_est))*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - 2 * sqrt(exp(sig_rho_est) + exp(2 * sig_rho_est)) / sqrt(1 + exp(-sig_beta_est))*q_qnorm_est * state_X_all_bffs[t - 1][n]) -
					2 * DR[t] * (sqrt(1 + exp(sig_rho_est))*q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))) -
					(1 + exp(sig_beta_est)) / (2 * exp(sig_rho_est))*
				((exp(sig_rho_est)*pow(q_qnorm_est, 2) + exp(sig_rho_est) / (1 + exp(-sig_beta_est))*pow(state_X_all_bffs[t - 1][n], 2) - (exp(sig_rho_est) + 2 * exp(2 * sig_rho_est)) / sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) * (1 + exp(-sig_beta_est)))*q_qnorm_est*state_X_all_bffs[t - 1][n]) - DR[t] * (exp(sig_rho_est) / sqrt(1 + exp(sig_rho_est)) * q_qnorm_est - sqrt(exp(sig_rho_est) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n]))
				);




			//q_qnorm 観測変数について
			q_qnorm_grad += weight_state_all_bffs[t - 1][n] * (
				(1 + exp(sig_beta_est)) / (exp(sig_rho_est))*
				(-(1 + exp(sig_rho_est))*q_qnorm_est + sqrt((exp(sig_rho_est) + exp(2 * sig_rho_est)) / (1 + exp(-sig_beta_est)))*state_X_all_bffs[t - 1][n] + DR[t] * sqrt(1 + exp(sig_rho_est)))
				);
		}
	}
#pragma omp parallel for reduction(+:X_0_grad)
	for (n = 0; n < N; n++) {
		//X_0 説明変数について
		X_0_grad += weight_state_all_bffs[0][n] * (
			exp(sig_beta_est) * (sqrt(1 - exp(-sig_beta_est))*state_X_all_bffs[0][n] - X_0_est)
			);
	}
	int grad_check = 1;
	l = 1;
	choice = int(r_rand_choice(mt));
	printf("beta_grad %f,rho_grad %f,q_grad %f X_0_grad %f,choice %d\n\n",
		beta_grad, rho_grad, q_qnorm_grad, X_0_grad, choice);
	
	while (grad_check) {
		sig_beta_est = sig_beta_est_tmp;
		sig_rho_est = sig_rho_est_tmp;
		q_qnorm_est = q_qnorm_est_tmp;
		X_0_est = X_0_est_tmp;
		if (choice == 0) {
			sig_beta_est = sig_beta_est + beta_grad * pow(b_grad, l);
		}
		else if (choice == 1) {
			sig_rho_est = sig_rho_est + rho_grad * pow(b_grad, l);
		}
		else if (choice == 2) {
			q_qnorm_est = q_qnorm_est + q_qnorm_grad * pow(b_grad, l);
		}
		else if (choice == 3) {
			X_0_est = X_0_est + X_0_grad * pow(b_grad, l);
		}
		beta_est = sig(sig_beta_est);
		rho_est = sig(sig_rho_est);
		if (Now_Q - Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight) <= -a_grad*pow(b_grad, l)*pow(beta_grad * pow(b_grad, l), 2) + pow(rho_grad * pow(b_grad, l), 2) + pow(q_qnorm_grad * pow(b_grad, l), 2) + pow(X_0_grad * pow(b_grad, l), 2)) {
			grad_check = 0;
		}
		l += 1;
		printf("%d ", l);
		if (l > 65) {
			grad_stop_check = 0;
			grad_check = 0;
		}
	}

	printf("\n Old Q %f,Now_Q %f\n,beta_est %f,rho_est %f,q %f X_0_est %f\n\n",
		Now_Q, Q(state_X_all_bffs, weight_state_all_bffs, beta_est, rho_est, q_qnorm_est, X_0_est, DR, T, N, Q_weight), beta_est, rho_est, pnorm(q_qnorm_est, 0, 1), X_0_est);

}



int main(void) {
	int n,t;
	int N = 1000;
	int T = 100;
	double beta_est;
	double rho_est;
	double q_qnorm_est;
	double X_0_est;
	/*フィルタリングの結果格納*/
	std::vector<std::vector<double> > filter_X(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
	std::vector<double> filter_X_mean(T);
	std::vector<double> predict_Y_mean(T);
	/*平滑化の結果格納*/
	std::vector<std::vector<double> > smoother_weight(T, std::vector<double>(N));
	std::vector<double> smoother_X_mean(T);
	/*Qの計算のためのwight*/
	std::vector<std::vector<std::vector<double>>> Q_weight(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));

	/*Answer格納*/
	std::vector<double> X(T);
	std::vector<double> DR(T);

	
	/*Xをモデルに従ってシミュレーション用にサンプリング、同時にDRもサンプリング 時点tのDRは時点t-1のXをパラメータにもつ正規分布に従うので、一期ずれる点に注意*/
	X[0] = sqrt(beta)*X_0 + sqrt(1 - beta) * rnorm(0, 1);
	DR[0] = -2;
	for (t = 1; t < T; t++) {
		X[t] = sqrt(beta)*X[t - 1] + sqrt(1 - beta) * rnorm(0, 1);
		DR[t] = r_DDR(X[t - 1], q_qnorm, rho, beta);
	}

	beta_est = beta;
	rho_est = rho;
	q_qnorm_est = q_qnorm;
	X_0_est = X_0;
	
	int grad_stop_check = 1;
	while (grad_stop_check) {
		particle_filter(DR, beta_est, q_qnorm_est, rho_est, X_0_est, N, T, filter_X, filter_weight, filter_X_mean, predict_Y_mean);
		particle_smoother(T, N, filter_weight, filter_X, beta_est, smoother_weight, smoother_X_mean);
		Q_weight_calc(T, N, beta_est, filter_weight, smoother_weight, filter_X, Q_weight);
		Q_grad(grad_stop_check, filter_X, smoother_weight, beta_est, rho_est, q_qnorm_est, X_0_est,DR, T, N, Q_weight);
	}
	

	FILE *fp;
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
		fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, X[t], filter_X_mean[t], smoother_X_mean[t], DR[t], predict_Y_mean[t]);
	}

	fclose(fp);
	FILE *gp, *gp2;
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
	fprintf(gp, "plot 'particle.csv' using 1:2:4:3 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
	fflush(gp);
	fprintf(gp, "replot 'X.csv' using 1:2 with lines linetype 1 lw 4 linecolor rgb '#ff0000 ' title 'Answer'\n");
	fflush(gp);
	//fprintf(gp, "set output 'particle.pdf'\n");
	fprintf(gp, "replot 'X.csv' using 1:3 with lines linetype 1 lw 4 linecolor rgb '#ffff00 ' title 'Filter'\n");
	fflush(gp);
	fprintf(gp, "replot 'X.csv' using 1:4 with lines linetype 3 lw 2.0 linecolor rgb 'white ' title 'Smoother'\n");
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
	fprintf(gp2, "replot 'X.csv' using 1:6 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00 ' title 'predict DR'\n");
	fflush(gp2);


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
