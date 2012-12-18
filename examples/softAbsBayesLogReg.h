#ifndef _BETA_SOFTABSBAYESLOGREG_

#include <softAbsMetric.h>

class softAbsBayesLogReg: public softAbsMetric
{
    
public:
    
    explicit softAbsBayesLogReg(MatrixXd& data, VectorXd& t, double alpha);
    ~softAbsBayesLogReg() {};
    
    double V();
    const VectorXd& gradV();
    
private:
    
    /// Regression coefficient prior variance
    double mAlpha;
    
    /// References to external data and binary responses
    VectorXd& mT;
    MatrixXd& mData;
    
    /// Internal storage for efficient linear algebra
    VectorXd mS;
    
    void fComputeH();
    
    void fComputeGradH(int i);
    
};

#define _BETA_SOFTABSBAYESLOGREG_
#endif