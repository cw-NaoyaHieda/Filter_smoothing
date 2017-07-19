#include "myfunc.h"
#include "sampling_DR.h"
#include "MT.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#define GNUPLOT_PATH "C:/PROGRA~2/gnuplot/bin/gnuplot.exe"
#define M_PI 3.14159265359	
#define beta 0.75
#define q_qnorm -2.053749 //qに直したときに、約0.02
#define rho 0.05
#define X_0 -2.5
#define alpha_grad 0.001
#define beta_grad 0.5
#define rand_seed 0
#include <iostream>
#include <vector>
#include <tuple>


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
void particle_filter(std::vector<double>& DR,double beta_est,double q_qnorm_est,double rho_est,double X_0_est,int N,int T,
	std::vector<std::vector<double>> state_X_all, std::vector<std::vector<double>> weight_state_all, std::vector<double > state_X_mean) {
	int n;
	int t;
	double pred_X_mean_tmp;
	double state_X_mean_tmp;
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
	std::vector<double> post_X(N),post_weight(N);

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
			resample_numbers[n] = resample(cumsum_weight,N, (Uniform() + n - 1) / N);
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
				resample_numbers[n] = resample(cumsum_weight,N, Uniform());
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



int main(void) {
	int n,t;
	int N = 1000;
	int T = 100;
	double beta_est;
	double rho_est;
	double q_qnorm_est;
	double X_0_est;
	std::vector<std::vector<double> > filter_X(T, std::vector<double>(N));
	std::vector<std::vector<double> > filter_weight(T, std::vector<double>(N));
	std::vector<double> filter_X_mean(N);

	/*Answer格納*/
	std::vector<double> X(T);
	std::vector<double> DR(T);

	/*乱数をスキップさせる*/
	int i;
	for (i = 0; i < rand_seed; i++) {
		Uniform();
	}

	/*Xをモデルに従ってシミュレーション用にサンプリング、同時にDRもサンプリング 時点tのDRは時点t-1のXをパラメータにもつ正規分布に従うので、一期ずれる点に注意*/
	X[0] = sqrt(beta)*X_0 + sqrt(1 - beta) * rnorm(0, 1);

	for (t = 1; t < T; t++) {
		X[t] = sqrt(beta)*X[t - 1] + sqrt(1 - beta) * rnorm(0, 1);
		DR[t] = r_DDR(X[t - 1], q_qnorm, rho, beta);
	}

	beta_est = beta;
	rho_est = rho + 0.3;
	q_qnorm_est = q_qnorm;
	X_0_est = X_0;

	particle_filter(DR,beta_est,q_qnorm_est,rho_est,X_0_est,N,T, filter_X, filter_weight, filter_X_mean);
	
	/*
	printf("\n\n Score%f \n\n", Q());
	FILE *fp;
	if (fopen_s(&fp, "particle.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 1; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, state_X_all[t][n], weight_state_all[t][n], N / 20 * weight_state_all[t][n], weight_state_all_bffs[t][n], N / 20 * weight_state_all_bffs[t][n]);

		}
	}

	/*
	printf("\n\n Score%f \n\n", Q());
	FILE *fp;
	if (fopen_s(&fp, "particle.csv", "w") != 0) {
		return 0;
	}

	for (t = 1; t < T; t++) {
		for (n = 1; n < N; n++) {
			fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, state_X_all[t][n], weight_state_all[t][n], N / 20 * weight_state_all[t][n], weight_state_all_bffs[t][n], N / 20 * weight_state_all_bffs[t][n]);

		}
	}
	fclose(fp);

	if (fopen_s(&fp, "X.csv", "w") != 0) {
		return 0;
	}
	for (t = 1; t < T - 1; t++) {
		fprintf(fp, "%d,%f,%f,%f,%f,%f\n", t, X[t], state_X_mean[t], pred_X_mean[t], pnorm(DR[t], 0, 1), state_X_all_bffs_mean[t]);
	}

	fclose(fp);
	FILE *gp, *gp2;
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
	fprintf(gp, "exit\n");    // gnuplotの終了
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
