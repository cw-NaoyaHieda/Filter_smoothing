#pragma once
#ifndef _SAMPLING_DR_H_
#define _SAMPLING_DR_H_
double AR_sim(int T,double *y, double mu, double sigma, double tau);
double reject_sample(double pd, double rho);
double r_DDR(double X_t, double q, double rho, double beta);
#endif // _SAMPLING_DR_H_