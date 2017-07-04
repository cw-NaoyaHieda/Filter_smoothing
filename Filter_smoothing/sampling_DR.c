/* 関数定義*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myfunc.h"
#define M_PI 3.14159265359	

double *data_vector(int T) {
	double *data_vector = malloc(sizeof(int) * T);
	return data_vector;
}

/*ARモデルからのサンプル 系列の初期値のポインタを受け取っている*/
double* AR_sim(int T,double *y,double mu,double sigma,double phi) {
	/*ARモデルの平均*/
	double sig_mu;
	/*元のデータの系列*/
	double y_;
	/*何番目の処理かのカウント*/
	int i;
	i = 0;
	/*平均をシグモイド関数で変換　全ての系列をシグモイド関数で変換した先でARモデルからサンプリングする*/
	sig_mu = sig_env(mu);
	/*初期値を計算　平均+誤差*/
	*y = sig_mu + rnorm(0, sigma);
	/*系列の最後尾までループ*/
	while (i != T) {
		/*ARモデルに従って次の値を計算する*/
		y_ = sig_mu + phi * (*y - sig_mu) + rnorm(0, sigma);
		/*一個前の値をシグモイド関数で変換*/
		*y = sig(*y);
		/*系列のポインタを一個次に進める*/
		++y;
		/*先ほど計算した値をyに代入*/
		*y = y_;
		/*カウントを進める*/
		++i;
	}
	/*ループが終了した時点で最後の一個が変換されていないはずなので変換*/
	*y = sig(*y);
	return 0;
}

/*棄却法によってDRを発生させる*/
double reject_sample(double pd, double rho) {
	int i;
	double y;
	double prob[10000];
	double max_density = 0;
	double density_range;

	/*現在のパラメータでの確率密度の頂点と、密度が0以上の点の範囲を求める*/
	for (i = 1; i < 9999; i++) {
		prob[i] = g_DR_fn(i / 10000.0, pd, rho);
		if (prob[i] > max_density) {
			max_density = prob[i];
		}
		if (prob[i] > 0) {
			density_range = i / 10000.0;
		}
	}
	
	/*棄却法でDR発生*/
	while (1) {
		y = Uniform() * density_range;
		if (g_DR_fn(y, pd, rho) > max_density*Uniform()) {
			return y;
		}
	}
}

/*サンプリング DynamicDefaultRate*/
double r_DDR(double X_t, double q_qnorm, double rho, double beta) {
	return (q_qnorm - sqrt(rho)*sqrt(beta)*X_t) / sqrt(1 - rho) - sqrt(rho)*sqrt(1 - beta) / sqrt(1 - rho) * rnorm(0, 1);
}

