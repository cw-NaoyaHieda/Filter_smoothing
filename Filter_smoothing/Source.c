#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <process.h>
#include <windows.h>
#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define T 100
#define N 100
#define tau_rho 0.95
#define tau_pd 0.95
#define mean_rho 0.1
#define mean_pd 0.04
#define sd_sig_rho 0.05
#define sd_sig_pd 0.05


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

/*全期間の推定値格納　平滑化*/
double state_pd_sig_all_bffs[T][N]; //pdのParticleそのもの　シグモイド関数の逆関数で変換
double state_rho_sig_all_bffs[T][N]; //rhoのParticleそのもの　シグモイド関数の逆関数で変換
double weight_state_all_bffs[T][N]; // weight リサンプリングしたもの


/*パラメータ用変数*/
double tau_rho_est;
double tau_pd_est;
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

/*timeとParticleのfor文用変数*/
int t;
int n;
int n2;


/*マルチスレッド*/
int	thread_id1, thread_id2;
unsigned	dummy;
int	p_pid;



/*フィルタリング*/
int muliti_particle_filter() {
	/*時点1でのフィルタリング開始*/

	/*初期分布からのサンプリングし、そのまま時点1のサンプリング*/
	for (n = 0; n < N; n++) {
		/*初期分布から　時点0と考える*/
		first_pd_sig[n] = rnorm(sig_env(mean_pd_est), sd_sig_pd_est);
		first_rho_sig[n] = rnorm(sig_env(mean_rho_est), sd_sig_rho_est);
		/*その結果からサンプリング*/
		pred_pd_sig[n] = rnorm(mean_pd_est + tau_pd*(first_pd_sig[n] - mean_pd_est), sd_sig_pd_est);
		pred_rho_sig[n] = rnorm(mean_rho_est + tau_rho*(first_rho_sig[n] - mean_rho_est), sd_sig_rho_est);
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
			weight_state_all[0][n] = 1 / N;
		}
	}

	/*こっからは繰り返し処理*/
	for (t = 1; t < T; t++) {
		/*一期前の結果取得*/
		for (n = 0; n < N; n++) {
			re_n = resample_numbers[n];
			post_pd_sig[n] = state_pd_all[t - 1][re_n];
			post_rho_sig[n] = state_rho_all[t - 1][re_n];
			post_weight[n] = weight_state_all[t - 1][re_n];
		}
		/*時点tのサンプリング*/
		for (n = 0; n < N; n++) {
			/*その結果からサンプリング*/
			pred_pd_sig[n] = rnorm(mean_pd_est + tau_pd_est*(post_pd_sig[n] - mean_pd_est), sd_sig_pd_est);
			pred_rho_sig[n] = rnorm(mean_rho_est + tau_rho_est*(post_rho_sig[n] - mean_rho_est), sd_sig_rho_est);
			/*pd と rhoに変換*/
			pred_pd[n] = sig(pred_pd_sig[n]);
			pred_rho[n] = sig(pred_rho_sig[n]);
		}

		/*重みの計算*/
		sum_weight = 0;
		resample_check_weight = 0;
		for (n = 0; n < N; n++) {
			weight[n] = g_DR_fn(DR[t], pred_pd[n], pred_rho[n]);
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
				weight_state_all[t][n] = 1 / N;
			}
		}

	}
	return 0;
}
/*平滑化*/
int muliti_particle_soother() {
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
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n] - sig_env(mean_pd_est)),
						sd_sig_pd_est)*
					dnorm(state_rho_sig_all_bffs[t + 1][n2],
						sig_env(mean_pd_est) + (state_pd_sig_all[t][n] - sig_env(mean_pd_est)),
						sd_sig_rho_est);
				bunsi_sum += bunsi[n][n2];
				/*分母計算 超わかりにくいけどご勘弁*/
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

int main(void) {
	
	thread_id1 = _beginthreadex(NULL, 0, counter, (void *)1, 0, &dummy);
	thread_id2 = _beginthreadex(NULL, 0, counter, (void *)2, 0, &dummy);

	/*PDとrhoをそれぞれARモデルに従ってシミュレーション用にサンプリング*/
	AR_sim(T,pd, mean_pd, sd_sig_pd, tau_pd);
	AR_sim(T,rho, mean_rho, sd_sig_rho, tau_rho);
	
	/*棄却法を用いて、各時点でのパラメータからDRを発生*/
	for (t = 0; t < T; t++) {
		DR[t] = reject_sample(pd[t], rho[t]);
	}

	/*パラメータ用変数*/
	tau_rho_est = tau_rho;
	tau_pd_est = tau_pd;
	mean_rho_est = mean_rho;
	mean_pd_est = mean_pd;
	sd_sig_rho_est = sd_sig_rho;
	sd_sig_pd_est = sd_sig_pd;

	muliti_particle_filter();
	muliti_particle_soother();
	
	

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

