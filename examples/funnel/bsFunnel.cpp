#include "math.h"

#include "bsFunnel.h"

bsFunnel::bsFunnel(int n): 
diagBkgdBsMetric(n + 1)
{}

double bsFunnel::quadT()
{
    fComputeLambda();
    mAux.noalias() = mLambda.selfadjointView<Eigen::Lower>() * mP;
    return 0.5 * mP.dot(mAux);
}

double bsFunnel::detT()
{
    fComputeLambda();
    return -0.5 * log(mDetLambda);
}

double bsFunnel::V()
{
    
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    return 0.5 * (mDim - 1) * mQ(0) + 0.5 * (sum * exp( -mQ(0) ) + mQ(0) * mQ(0) / 9.0 );
}

const VectorXd& bsFunnel::gradV()
{
    
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    double temp = exp( -mQ(0) );
    
    mGradV = temp * mQ;
    mGradV(0) = 0.5 * (mDim - 1) + mQ(0) / 9.0 - 0.5 * sum * temp;
    
    return mGradV;
    
}

void bsFunnel::fComputeHessianV()
{
    
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    double temp = exp( -mQ(0) );
    
    mHessianV.setIdentity();
    mHessianV *= temp;
    
    mHessianV.col(0) = - temp * mQ;
    mHessianV.row(0) = - temp * mQ.transpose();
    
    mHessianV(0, 0) = 1.0 / 9.0 + 0.5 * sum * temp;
    
}

