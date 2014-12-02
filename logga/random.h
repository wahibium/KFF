#ifndef _RANDOM_H_
#define _RANDOM_H_

#define _M 2147483647     // modulus of PMMLCG (the default is 2147483647 = 2^31 - 1)
#define _A 16807          // the default is 16807

double drand();
int intRand(int max);
long longRand(long max);
char flipCoin();
double gaussianRandom(double mean,double stddev);

long setSeed(long newSeed);

#endif
