#ifndef _BETA_FISHERBAYESLOGREG_

#include <denseFisherMetric.h>

class fisherBayesLogReg: public denseFisherMetric
{
    
public:
    
    explicit fisherBayesLogReg(MatrixXd& data, VectorXd& t, double alpha);
    ~fisherBayesLogReg() {};
    
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
    
    void fComputeG();
    
    void fComputeGradG(int i);
    
    
};

#define _BETA_FISHERBAYESLOGREG_
#endif