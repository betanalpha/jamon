#ifndef _BETA_FLATFUNNEL_

#include <diagConstMetric.h>

class flatFunnel: public diagConstMetric
{
    
public:
    
    explicit flatFunnel(int n);
    ~flatFunnel() {};
    
    double V();
    const VectorXd& gradV();
    
private:
    
};

#define _BETA_FLATFUNNEL_
#endif