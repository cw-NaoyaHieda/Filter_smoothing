#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define T 100
#define N 1000
#define phi_rho 0.95
#define phi_pd 0.95
#define mean_rho 0.1
#define mean_pd 0.04
#define sd_sig_rho 0.1
#define sd_sig_pd 0.1
#define alpha 0.000000000001

/*Answer格納*/
double pd[T];
double rho[T];
double DR[T];

/*初期分布からのサンプリング格納*/
double first_pd_sig[N];
double first_rho_sig[N];

/*時点tの予測値格納*/
double pred_pd_sig[N]; //pdのParticleそのもの　シグモイド関数の逆関数で変換
double pred_rho_sig[N]; //rhoのParticleそのもの　シグモイド関数の逆関数で変換
double pred_pd[N]; //pdのParticle [0,1]の範囲に直したもの 予測値
double pred_rho[N]; //rhoのParticle [0,1]の範囲に直したもの　予測値
double weight[N]; // weight

/*全期間の推定値格納　フィルタリング*/
double pred_pd_all[T][N]; //pdのParticle [0,1]の範囲に直したもの 
double pred_rho_all[T][N]; //rhoのParticle [0,1]の範囲に直したもの
double state_pd_sig_all[T][N]; //pdのParticleそのもの　シグモイド関数の逆関数で変換 リサンプリングしたもの
double state_rho_sig_all[T][N]; //rhoのParticleそのもの　シグモイド関数の逆関数で変換 リサンプリングしたもの
double state_pd_all[T][N]; //pdのParticle [0,1]の範囲に直したもの リサンプリングしたもの
double state_rho_all[T][N]; //rhoのParticle [0,1]の範囲に直したもの リサンプリングしたもの
double weight_all[T][N]; // weight
double weight_state_all[T][N]; // weight リサンプリングしたもの
double state_pd_sig_mean[T]; //pdのフィルタリングの結果をweightで重み付けして平均したもの
double state_rho_sig_mean[T]; //pdのフィルタリングの結果をweightで重み付けして平均したもの

/*全期間の推定値格納　平滑化*/
double state_pd_sig_all_bffs[T][N]; //pdのParticleそのもの　シグモイド関数の逆関数で変換
double state_rho_sig_all_bffs[T][N]; //rhoのParticleそのもの　シグモイド関数の逆関数で変換
double weight_state_all_bffs[T][N]; // weight リサンプリングしたもの


/*パラメータ用変数*/
double phi_rho_est;
double phi_pd_est;
double mean_rho_est;
double mean_pd_est;
double sd_sig_rho_est;
double sd_sig_pd_est;


/*途中の処理用変数*/
double sum_weight; //正規化因子(weightの合計)
double cumsum_weight[N]; //累積尤度　正規化した上で計算したもの
double resample_check_weight; //リサンプリングの判断基準　正規化尤度の二乗の合計
int resample_numbers[N]; //リサンプリングした結果の番号
int check_resample; //リサンプリングしたかどうかの変数 0ならしてない、1ならしてる
int re_n; //リサンプリングの参照用
double bunsi[N][N]; //平滑化の分子
double bunbo[N][N]; //平滑化の分母
double bunsi_sum;
double bunbo_sum;

/*一期前の結果*/
double post_pd_sig[N];
double post_rho_sig[N];
double post_weight[N];

/*EMの計算*/
double q_state;
double q_obeserve;
double first_observe;
double first_state;

/*EM中の最急降下法*/
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

/*timeとParticleのfor文用変数*/
int t;
int n;
int n2;

/*フィルタリング*/
int particle_filter() {
	/*時点1でのフィルタリング開始*/
	/*初期分布からのサンプリングし、そのまま時点1のサンプリング*/
	for (n = 0; n < N; n++) {
		/*初期分布から　時点0と考える*/
		first_pd_sig[n] = rnorm(sig_env(mean_pd_est), sd_sig_pd_est);
		first_rho_sig[n] = rnorm(sig_env(mean_rho_est), sd_sig_rho_est);
		/*その結果からサンプリング*/
		pred_pd_sig[n] = rnorm(sig_env(mean_pd_est) + phi_pd_est*(first_pd_sig[n] - sig_env(mean_pd_est)), sd_sig_pd_est);
		pred_rho_sig[n] = rnorm(sig_env(mean_rho_est) + phi_rho_est*(first_rho_sig[n] - sig_env(mean_rho_est)), sd_sig_rho_est);
		/*pd と rhoに変換*/
		pred_pd[n] = sig(pred_pd_sig[n]);
		pred_rho[n] = sig(pred_rho_sig[n]);
	}

	

	/*重みの計算*/
	sum_weight = 0;
	resample_check_weight = 0;

	for (n = 0; n < N; n++) {
		weight[n] = g_DR_fn(DR[0], pred_pd[n], pred_rho[n]);
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

	/*結果の格納*/
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

	/*こっからは繰り返し処理*/
	for (t = 1; t < T; t++) {
		/*一期前の結果取得*/
		for (n = 0; n < N; n++) {
			re_n = resample_numbers[n];
			post_pd_sig[n] = state_pd_sig_all[t - 1][re_n];
			post_rho_sig[n] = state_rho_sig_all[t - 1][re_n];
			post_weight[n] = weight_state_all[t - 1][re_n];
		}
		/*時点tのサンプリング*/
		for (n = 0; n < N; n++) {
			/*その結果からサンプリング*/
			pred_pd_sig[n] = rnorm(sig_env(mean_pd_est) + phi_pd_est*(post_pd_sig[n] - sig_env(mean_pd_est)), sd_sig_pd_est);
			pred_rho_sig[n] = rnorm(sig_env(mean_rho_est) + phi_rho_est*(post_rho_sig[n] - sig_env(mean_rho_est)), sd_sig_rho_est);
			/*pd と rhoに変換*/
			pred_pd[n] = sig(pred_pd_sig[n]);
			pred_rho[n] = sig(pred_rho_sig[n]);
		}

		/*重みの計算*/
		sum_weight = 0;
		resample_check_weight = 0;
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_fn(DR[t], pred_pd[n], pred_rho[n]) * post_weight[n];
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

		state_pd_sig_mean[t] = 0;
		state_rho_sig_mean[t] = 0;
		/*結果の格納*/
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
			state_pd_sig_mean[t] += state_pd_sig_all[t][n] * weight_state_all[t][n];
			state_rho_sig_mean[t] += state_rho_sig_all[t][n] * weight_state_all[t][n];
		}

	}
	return 0;
}

/*平滑化*/
int particle_smoother() {
	/*T時点のweightは変わらないのでそのまま代入*/
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
				/*分子計算 超わかりにくいけどご勘弁*/
				bunsi[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(state_pd_sig_all_bffs[t + 1][n2],
						sig_env(mean_pd_est) + phi_pd_est * (state_pd_sig_all[t][n] - sig_env(mean_pd_est)),
						sd_sig_pd_est)*
					dnorm(state_rho_sig_all_bffs[t + 1][n2],
						sig_env(mean_rho_est) + phi_rho_est * (state_rho_sig_all[t][n] - sig_env(mean_rho_est)),
						sd_sig_rho_est);
				bunsi_sum += bunsi[n][n2];
				/*分母計算 超わかりにくいけどご勘弁*/
				bunbo[n][n2] = weight_state_all_bffs[t + 1][n2] *
					dnorm(state_pd_sig_all_bffs[t + 1][n],
						sig_env(mean_pd_est) + phi_pd_est * (state_pd_sig_all[t][n2] - sig_env(mean_pd_est)),
						sd_sig_pd_est)*
					dnorm(state_rho_sig_all_bffs[t + 1][n],
						sig_env(mean_rho_est) + phi_rho_est * (state_rho_sig_all[t][n2] - sig_env(mean_rho_est)),
						sd_sig_rho_est);
				bunbo_sum += bunbo[n][n2];
			}
			weight_state_all_bffs[t][n] = weight_state_all[t][n] * bunsi_sum / bunbo_sum;
			sum_weight += weight_state_all_bffs[t][n];
		}
		if (t == 0) {
			sum_weight;
		}
		/*正規化と累積相対尤度の計算*/
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

		/*リサンプリングが必要かどうか判断したうえで必要ならリサンプリング 必要ない場合は順番に数字を入れる*/
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
		/*リサンプリングの必要性に応じて結果の格納*/
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

/*Q EMで最小化したい式*/
double Q() {
	/*以下も関数にする　Q*/
	q_state = 0;
	q_obeserve = 0;
	for (t = 1; t < T; t++) {
		for (n = 0; n < N; n++) {
			q_state += weight_state_all_bffs[t][n] * //weight
				log(
					dnorm(state_pd_sig_all_bffs[t][n], sig_env(mean_pd_est) + phi_pd_est * (state_pd_sig_all_bffs[t - 1][n] - sig_env(mean_pd_est)), sd_sig_pd_est)* //pdの遷移確率
					dnorm(state_rho_sig_all_bffs[t][n], sig_env(mean_rho_est) + phi_rho_est * (state_rho_sig_all_bffs[t - 1][n] - sig_env(mean_rho_est)), sd_sig_rho_est)//rhoの遷移確率
				);
			q_obeserve += weight_state_all_bffs[t][n] * //weight
				log(
					g_DR_fn(DR[t], sig(state_pd_sig_all_bffs[t][n]), sig(state_rho_sig_all_bffs[t][n]))//観測の確率
				);
		}
	}
	first_observe = 0;
	first_state = 0;
	for (n = 0; n < N; n++) {
		first_observe += weight_state_all_bffs[0][n] * //weight
			log(
				g_DR_fn(DR[0], sig(state_pd_sig_all_bffs[0][n]), sig(state_rho_sig_all_bffs[0][n]))//観測の確率
			);
		first_state += weight_state_all_bffs[0][n] * //weight
			log(
				dnorm(state_pd_sig_all_bffs[0][n],sig_env(mean_pd_est),sd_sig_pd_est)*
				dnorm(state_rho_sig_all_bffs[0][n], sig_env(mean_rho_est), sd_sig_rho_est)//初期分布からの確率
			);
	}
	return q_state + q_obeserve + first_observe + first_state;
}

/*Qの最急降下*/
double q() {
	while (1) {
		Now_Q = Q();
		phi_rho_est_tmp = phi_rho_est;
		phi_pd_est_tmp = phi_pd_est;
		mean_rho_est_tmp = mean_rho_est;
		mean_pd_est_tmp = mean_pd_est;
		/*ダミー変数を用いるため sigma=z^2とおく*/
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
				a += weight_state_all_bffs[t][n] * (state_rho_sig_all_bffs[t][n] * state_rho_sig_all_bffs[t - 1][n] - state_rho_sig_all_bffs[t][n] * sig_env(mean_rho_est) -
					phi_rho_est * pow(state_rho_sig_all_bffs[t][n], 2) - phi_rho_est * pow(sig_env(mean_rho_est), 2) - state_rho_sig_all_bffs[t][n] *
					sig_env(mean_rho_est) + pow(sig_env(mean_rho_est), 2)) / pow(sd_sig_rho_est, 4);
				b += weight_state_all_bffs[t][n] * (state_pd_sig_all_bffs[t][n] * state_pd_sig_all_bffs[t - 1][n] - state_pd_sig_all_bffs[t][n] * sig_env(mean_pd_est) -
					phi_pd_est * pow(state_pd_sig_all_bffs[t][n], 2) - phi_pd_est * pow(sig_env(mean_pd_est), 2) - state_pd_sig_all_bffs[t][n] *
					sig_env(mean_pd_est) + pow(sig_env(mean_pd_est), 2)) / pow(sd_sig_pd_est, 4);

				c += weight_state_all_bffs[t][n] * (- phi_rho_est * state_rho_sig_all_bffs[t][n] + state_rho_sig_all_bffs[t][n] - pow(phi_rho_est,2) * mean_rho_est -
					sig_env(mean_rho_est) - phi_rho_est * state_rho_sig_all_bffs[t - 1][n] + 2 * phi_rho_est * mean_rho_est) / pow(sd_sig_rho_est,4);
				d += weight_state_all_bffs[t][n] * (-phi_pd_est * state_pd_sig_all_bffs[t][n] + state_pd_sig_all_bffs[t][n] - pow(phi_pd_est, 2) * mean_pd_est -
					sig_env(mean_pd_est) - phi_pd_est * state_pd_sig_all_bffs[t - 1][n] + 2 * phi_pd_est * mean_pd_est) / pow(sd_sig_pd_est, 4);
				
				e += -2 / sd_sig_rho_est + 2 * pow(state_rho_sig_all_bffs[t][n] - (phi_rho_est * (state_rho_sig_all_bffs[t - 1][n] - sig_env(mean_rho_est)) + sig_env(mean_rho_est)), 2) / pow(sd_sig_rho_est, 5);
				f += -2 / sd_sig_pd_est + 2 * pow(state_pd_sig_all_bffs[t][n] - (phi_pd_est * (state_pd_sig_all_bffs[t - 1][n] - sig_env(mean_pd_est)) + sig_env(mean_pd_est)), 2) / pow(sd_sig_pd_est, 5);
			}
		}

		phi_rho_est = phi_rho_est + a * alpha;
		phi_pd_est = phi_pd_est + b * alpha;
		mean_rho_est = sig(sig_env(mean_rho_est) + c * alpha);
		mean_pd_est = sig(sig_env(mean_pd_est) + d * alpha);
		sd_sig_rho_est = sd_sig_rho_est + e*alpha;
		sd_sig_pd_est = sd_sig_pd_est + f*alpha;
		sd_sig_rho_est = pow(sd_sig_rho_est, 2);
		sd_sig_pd_est = pow(sd_sig_pd_est, 2);
		printf("Old Q %f,Now_Q %f,phi_rho_est %f,mean_rho_est %f,sd_sig_rho_est %f\n phi_pd_est %f,mean_pd_est %f,sd_sig_pd_est %f\n",
			Now_Q, Q(), phi_rho_est, mean_rho_est, sd_sig_rho_est, phi_pd_est, mean_pd_est, sd_sig_pd_est);
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

/*Qの最急降下 ただしへんてこ*/
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
	
	/*PDとrhoをそれぞれARモデルに従ってシミュレーション用にサンプリング*/
	AR_sim(T,pd, mean_pd, sd_sig_pd, phi_pd);
	AR_sim(T,rho, mean_rho, sd_sig_rho, phi_rho);
	pd[0] = 0.04;
	/*棄却法を用いて、各時点でのパラメータからDRを発生*/
	for (t = 0; t < T; t++) {
		DR[t] = reject_sample(pd[t], rho[t]);
	}

	/*パラメータ用変数*/
	phi_rho_est = phi_rho;
	phi_pd_est = phi_pd;
	mean_rho_est = mean_rho;
	mean_pd_est = mean_pd;
	sd_sig_rho_est = sd_sig_rho;
	sd_sig_pd_est = sd_sig_pd;

	while (1) {
		particle_filter();
		/*こっからplot*/
		
		
		FILE *fp,*fp2,*fp3;
		if (fopen_s(&fp, "particle.csv", "w") != 0) {
			return 0;
		}
		
		for (t = 1; t < T; t++) {
			for (n = 1; n < N; n++) {
				fprintf(fp,"%d,%f,%f,%f,%f\n",t,state_pd_all[t][n], state_rho_all[t][n],weight_state_all[t][n],N / 100 * weight_state_all[t][n]);

			}
		}
		fclose(fp);

		if (fopen_s(&fp, "pd.csv", "w") != 0) {
			return 0;
		}
		if (fopen_s(&fp2, "DR.csv", "w") != 0) {
			return 0;
		}
		if (fopen_s(&fp3, "rho.csv", "w") != 0) {
			return 0;
		}
		for (t = 1; t < T; t++) {
			fprintf(fp, "%d,%f,%f\n", t, pd[t], sig(state_pd_sig_mean[t]));
			fprintf(fp2, "%d,%f\n", t, DR[t]);
			fprintf(fp3, "%d,%f,%f\n", t, rho[t], sig(state_rho_sig_mean[t]));
		}

		fclose(fp);
		fclose(fp2);
		fclose(fp3);
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
		fprintf(gp, "set object 1 rect fc rgb '#333333' fillstyle solid 1.0 \n");
		fprintf(gp, "set key textcolor rgb 'white'\n");
		fprintf(gp, "set size ratio 1/3\n");
		fprintf(gp, "plot 'particle.csv' using 1:2:5:4 with circles notitle fs transparent solid 0.65 lw 2.0 pal \n");
		fflush(gp);
		fprintf(gp, "replot 'pd.csv' using 1:2 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00' title 'Answer'\n");
		fflush(gp);
		fprintf(gp, "replot 'pd.csv' using 1:3 with lines linetype 1 lw 3.0 linecolor rgb '#7fffd4' title 'Filter'\n");
		fflush(gp);
		fprintf(gp, "replot 'DR.csv' using 1:2 with lines linetype 1 lw 3.0 linecolor rgb '#f03232' title 'DefaultRate'\n");
		fflush(gp);
		
		gp2 = _popen(GNUPLOT_PATH, "w");
		fprintf(gp2, "reset\n");
		fprintf(gp2, "set datafile separator ','\n");
		fprintf(gp2, "set grid lc rgb 'white' lt 2\n");
		fprintf(gp2, "set border lc rgb 'white'\n");
		fprintf(gp2, "set border lc rgb 'white'\n");
		fprintf(gp2, "set cblabel 'Weight' tc rgb 'white' font ', 30'\n");
		fprintf(gp2, "set palette rgbformulae 22, 13, -31\n");
		fprintf(gp2, "set obj rect behind from screen 0, screen 0 to screen 1, screen 1 \n");
		fprintf(gp2, "set object 1 rect fc rgb '#333333' fillstyle solid 1.0 \n");
		fprintf(gp2, "set key textcolor rgb 'white'\n");
		fprintf(gp2, "set size ratio 1/3\n");
		fprintf(gp2, "plot 'particle.csv' using 1:3:5:4 with circles notitle fs transparent solid 0.85 lw 2.0 pal \n");
		fflush(gp2);
		fprintf(gp2, "replot 'rho.csv' using 1:2 with lines linetype 1 lw 3.0 linecolor rgb '#ffff00' title 'Answer'\n");
		fflush(gp2);
		fprintf(gp2, "replot 'rho.csv' using 1:3 with lines linetype 1 lw 3.0 linecolor rgb '#7fffd4' title 'Filter'\n");
		fflush(gp2);
		system("pause");
		fprintf(gp, "exit\n");
		fprintf(gp2, "exit\n");	// gnuplotの終了
		_pclose(gp);
		_pclose(gp2);

		
		particle_smoother();
		//q();
		return 0;
			
			
	}

	

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