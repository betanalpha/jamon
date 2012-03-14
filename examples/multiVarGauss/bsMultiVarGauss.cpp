#include "bsMultiVarGauss.h"

bsMultiVarGauss::bsMultiVarGauss(VectorXd mu, MatrixXd lambda): 
diagBkgdBsMetric((int)mu.size()),
mMu(mu),
mGaussLambda(lambda)
{}

double bsMultiVarGauss::V()
{
    mAux.noalias() = mGaussLambda.selfadjointView<Eigen::Lower>() * (mQ - mMu);
    return 0.5 * (mQ - mMu).dot(mAux);
}

const VectorXd& bsMultiVarGauss::gradV()
{
    mGradV.noalias() = mGaussLambda.selfadjointView<Eigen::Lower>() * (mQ - mMu);
    return mGradV;
}

void bsMultiVarGauss::fComputeHessianV()
{
    mHessianV = mGaussLambda;
}

