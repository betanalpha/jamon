#include "math.h"

#include "flatFunnel.h"

flatFunnel::flatFunnel(int n): 
diagConstMetric(n + 1)
{}

double flatFunnel::V()
{
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    return 0.5 * (mDim - 1) * mQ(0) + 0.5 * (sum * exp( -mQ(0) ) + mQ(0) * mQ(0) / 9.0 );
}

const VectorXd& flatFunnel::gradV()
{
    double sum = 0;
    for(int i = 1; i < mDim; ++i) sum += mQ(i) * mQ(i);
    
    double temp = exp( -mQ(0) );
    
    mGradV = temp * mQ;
    mGradV(0) = 0.5 * (mDim - 1) + mQ(0) / 9.0 - 0.5 * sum * temp;
    
    return mGradV;
}
