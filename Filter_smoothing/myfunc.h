#pragma once
#ifndef _MYFUNC_H_
#define _MYFUNC_H_
double sig(double x);
double sig_env(double x);
double Uniform(void);
double sig(double x);
double sig_env(double x);
double rnorm(double mu, double sigma);
double dnorm(double x, double mu, double sigma);
double pnorm(double q, double mu, double sigma);
double qnorm(double q, double mu, double sigma);
double g_DR_fn(double DR, double PD, double rho);
double g_DR_distribution(double DR, double PD, double rho);
double g_DR_(double DR, double PD, double rho);
int resample(int num_of_particle, double x, double *cumsum_weight);
#endif // _MYFUNC_H_