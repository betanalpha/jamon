#include "math.h"

#include "softAbsFunnel.h"

softAbsFunnel::softAbsFunnel(int n): 
softAbsMetric(n + 1)
{}

double softAbsFunnel::V()
{
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    return 0.5 * (mDim - 1) * mQ(0) + 0.5 * (sum * exp( -mQ(0) ) + mQ(0) * mQ(0) / 9.0 );
}

const VectorXd& softAbsFunnel::gradV()
{
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    double temp = exp( -mQ(0) );
    
    mGradV = temp * mQ;
    mGradV(0) = 0.5 * (mDim - 1) + mQ(0) / 9.0 - 0.5 * sum * temp;
    
    return mGradV;
}

void softAbsFunnel::fComputeH()
{
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    double temp = exp( -mQ(0) );
    
    mH.setIdentity();
    mH *= temp;
    
    mH.col(0) = - temp * mQ;
    mH.row(0) = - temp * mQ.transpose();
    
    mH(0, 0) = 1.0 / 9.0 + 0.5 * sum * temp;
}

void softAbsFunnel::fComputeGradH(int i)
{
    
    if(i == 0)
    {
        mGradH.block(0, i * mDim, mDim, mDim) = - mH;
        mGradH.block(0, i * mDim, mDim, mDim)(0, 0) += 1.0 / 9.0;
        
        return;
    }
    
    mGradH.block(0, i * mDim, mDim, mDim).setZero();
    
    mGradH.block(0, i * mDim, mDim, mDim)(0, 0) = mQ(i);
    mGradH.block(0, i * mDim, mDim, mDim)(i, 0) = -1.0;
    mGradH.block(0, i * mDim, mDim, mDim)(0, i) = -1.0;
    
    mGradH.block(0, i * mDim, mDim, mDim) *= exp( - mQ(0) );
    
}


