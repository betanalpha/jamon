#ifndef _BETA_SOFTABSFUNNEL_

#include <softAbsMetric.h>

class softAbsFunnel: public softAbsMetric
{
    
public:
    
    explicit softAbsFunnel(int n);
    ~softAbsFunnel() {};
    
    double V();
    const VectorXd& gradV();
    
private:
    
    void fComputeH();
    void fComputeGradH(int i);
    
};

#define _BETA_SOFTABSFUNNEL_
#endif