#pragma once
#ifndef _MYFUNC_H_
#define _MYFUNC_H_
double sig(double x);
double sig_env(double x);
double Uniform(void);
double sig(double x);
double sig_env(double x);
double rnorm(double mu, double sd);
double dnorm(double x, double mu, double sd);
double pnorm(double q, double mu, double sd);
double qnorm(double q);
double g_DR_fn(double DR, double PD, double rho);
double g_DR_distribution(double DR, double PD, double rho);
double g_DR_(double DR, double PD, double rho);
int resample(int num_of_particle, double x, double *cumsum_weight);
double g_DR_dinamic(double tilde_DR, double X_t_1, double q, double beta, double rho);
#endif // _MYFUNC_H_