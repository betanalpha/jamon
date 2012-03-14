#ifndef _BETA_BSMULTIVARGAUSS_

#include <diagBkgdBsMetric.h>

class bsMultiVarGauss: public diagBkgdBsMetric
{
    
public:
    
    explicit bsMultiVarGauss(VectorXd mu, MatrixXd lambda);
    ~bsMultiVarGauss() {};
    
    double V();
    const VectorXd& gradV();
    
private:
    
    VectorXd mMu;
    MatrixXd mGaussLambda;
    
    void fComputeHessianV();
    
};

#define _BETA_BSMULTIVARGAUSS_
#endif