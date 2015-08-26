#ifndef _RANDOM_H_
#define _RANDOM_H_

#define _M 2147483647     // modulus of PMMLCG (the default is 2147483647 = 2^31 - 1)
#define _A 16807          // the default is 16807

double Drand();
int IntRand(int max);
long LongRand(long max);
char FlipCoin();
double GaussianRandom(double mean,double stddev);

long SetSeed(long newSeed);

#endif
