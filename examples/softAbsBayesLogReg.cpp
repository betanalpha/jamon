#include "math.h"

#include "softAbsBayesLogReg.h"

softAbsBayesLogReg::softAbsBayesLogReg(MatrixXd& data, VectorXd& t, double alpha): 
softAbsMetric((int)data.cols()),
mData(data),
mT(t),
mS(VectorXd::Zero((int)data.rows())),
mAlpha(alpha)
{}

double softAbsBayesLogReg::V()
{
    
    mAuxVector.noalias() = mData.transpose() * mT;
    
    double v = - mQ.dot(mAuxVector);
    
    mS = mData * mQ;
    for(int n = 0; n < mS.size(); ++n)
    {
        
        if(mS(n) > 0)
        {
            v += mS(n) + log(1.0 + exp(-mS(n)));
        }
        else
        {
            v += log(1.0 + exp(mS(n)));
        }
    }
    
    v += mQ.squaredNorm() / (2.0 * mAlpha);
    
    return v;
    
}

const VectorXd& softAbsBayesLogReg::gradV()
{
    
    mS = mData * mQ;
    
    double temp = 0;
    for(int n = 0; n < mS.size(); ++n)
    {
        
        if(mS(n) > 0)
        {
            mS(n) = 1.0 / ( 1.0 + exp(-mS(n)) );
        }
        else
        {
            temp = exp(mS(n));
            mS(n) = temp / (temp + 1.0);
        }
        
    }
    
    mGradV.noalias() = mData.transpose() * (-mT + mS);
    
    mGradV += (1.0 / mAlpha) * mQ;
    
    return mGradV;
    
}

void softAbsBayesLogReg::fComputeH()
{
    
    mS = mData * mQ;
    
    double temp = 0;
    for(int n = 0; n < mS.size(); ++n)
    {
        
        if(mS(n) > 0)
        {
            mS(n) = 1.0 / ( 1.0 + exp(-mS(n)) );
        }
        else
        {
            temp = exp(mS(n));
            mS(n) = temp / (temp + 1.0);
        }
        
        mS(n) = mS(n) * (1.0 - mS(n));
        
    }
    
    mH.noalias() = mData.transpose() * mS.asDiagonal() * mData;
    mH.noalias() += (1.0 / mAlpha) * MatrixXd::Identity(mDim, mDim);
    
}

void softAbsBayesLogReg::fComputeGradH(int i)
{
    
    mS = mData * mQ;
    
    double temp = 0;
    for(int n = 0; n < mS.size(); ++n)
    {
        
        if(mS(n) > 0)
        {
            mS(n) = 1.0 / ( 1.0 + exp(-mS(n)) );
        }
        else
        {
            temp = exp(mS(n));
            mS(n) = temp / (temp + 1.0);
        }
        
        mS(n) = mS(n) * (1.0 - mS(n)) * (1.0 - 2.0 * mS(n)) * mData(n, i);
        
    }
    
    mGradH.block(0, i * mDim, mDim, mDim) = mData.transpose() * mS.asDiagonal() * mData;
    
}

