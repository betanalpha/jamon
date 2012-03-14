#ifndef _BETA_BSFUNNEL_

#include <diagBkgdBsMetric.h>

class bsFunnel: public diagBkgdBsMetric
{
    
public:
    
    explicit bsFunnel(int n);
    ~bsFunnel() {};
    
    double V();
    const VectorXd& gradV();
    
    double quadT();
    double detT();
    
private:
    
    void fComputeHessianV();
    
};

#define _BETA_BSFUNNEL_
#endif