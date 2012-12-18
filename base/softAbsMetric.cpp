#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "softAbsMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

softAbsMetric::softAbsMetric(int dim): 
dynamMetric(dim),
mSoftAbsAlpha(1.0),
mH(MatrixXd::Identity(mDim, mDim)),
mGradH(MatrixXd::Identity(mDim, mDim * mDim)),
mPseudoJ(MatrixXd::Identity(mDim, mDim)),
mAuxMatrixOne(MatrixXd::Identity(mDim, mDim)),
mAuxMatrixTwo(MatrixXd::Identity(mDim, mDim)),
mCacheMatrix(MatrixXd::Identity(mDim, mDim)),
mEigenDeco(mDim),
mSoftAbsLambda(VectorXd::Zero(mDim)),
mSoftAbsLambdaInv(VectorXd::Zero(mDim)),
mQp(VectorXd::Zero(mDim))
{}

double softAbsMetric::T()
{
    fComputeMetric();
    return 0.5 * mQp.dot(mLambdaQp) + 0.5 * mLogDetMetric;
}

double softAbsMetric::tau()
{
    fComputeMetric();
    return 0.5 * mQp.dot(mLambdaQp);
}

void softAbsMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector.noalias() = mEigenDeco.eigenvectors().transpose() * normal;
    mInit.noalias() = mSoftAbsLambdaInv.cwiseProduct(mAuxVector);
    
    double C = -2.0 * mQp.dot(mInit);
    C /= mAuxVector.dot(mInit);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void softAbsMetric::sampleP(Random& random)
{
    
    fComputeMetric();
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) 
    {
        mAuxVector(i) = sqrt(mSoftAbsLambda(i)) * g(random, 0.0, 1.0);
    }
    
    mP.noalias() = mEigenDeco.eigenvectors() * mAuxVector;
    
}

void softAbsMetric::checkEvolution(const double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    // Hessian
    std::cout.precision(6);
    int width = 12;
    int nColumn = 5;

    std::cout << "Potential Hessian (d^{2}V/dq^{i}dq^{j}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Row" 
              << std::setw(width) << std::left << "Column" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta / "
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "(j)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Stepsize^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    fComputeH();
    
    for(int i = 0; i < mDim; ++i)
    {
        
        VectorXd temp = VectorXd::Zero(mDim);        
        
        mQ(i) += epsilon;
        temp += gradV();
        mQ(i) -= 2.0 * epsilon;
        temp -= gradV();
        mQ(i) += epsilon;
        
        temp /= 2.0 * epsilon;
        
        for(int j = 0; j < mDim; ++j)
        {
            
            std::cout << "    "
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << j 
                      << std::setw(width) << std::left << mH(i, j)
                      << std::setw(width) << std::left << temp(j)
                      << std::setw(width) << std::left << (mH(i, j) - temp(j)) / (epsilon * epsilon)
                      << std::endl;
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Gradient of the Hessian
    std::cout.precision(6);
    width = 12;
    nColumn = 6;
    
    std::cout << "Gradient of the Hessian (d^{3}V/dq^{i}dq^{j}dq^{k}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Row" 
              << std::setw(width) << std::left << "Column" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta /"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "(j)"
              << std::setw(width) << std::left << "(k)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Stepsize^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    for(int k = 0; k < mDim; ++k)
    {
        
        mAuxMatrixOne.setZero();        
        
        mQ(k) += epsilon;
        fComputeH();
        mAuxMatrixOne += mH;
        mQ(k) -= 2.0 * epsilon;
        fComputeH();
        mAuxMatrixOne -= mH;
        mQ(k) += epsilon;
        
        mAuxMatrixOne /= 2.0 * epsilon;
        
        fComputeGradH(k);
        
        for(int i = 0; i < mDim; ++i)
        {
            
            for(int j = 0; j < mDim; ++j)
            {
                
                std::cout << "    "
                << std::setw(width) << std::left << k
                << std::setw(width) << std::left << i
                << std::setw(width) << std::left << j 
                << std::setw(width) << std::left << mGradH.block(0, k * mDim, mDim, mDim)(i, j)
                << std::setw(width) << std::left << mAuxMatrixOne(i, j)
                << std::setw(width) << std::left << (mGradH.block(0, k * mDim, mDim, mDim)(i, j) - mAuxMatrixOne(i, j)) / (epsilon * epsilon)
                << std::endl;
                
            }
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Metric
    std::cout.precision(6);
    width = 12;
    nColumn = 6;
    
    std::cout << "Gradient of the metric (dLambda^{jk}/dq^{i}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Row" 
              << std::setw(width) << std::left << "Column" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta /"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "(j)"
              << std::setw(width) << std::left << "(k)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Epsilon^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    fComputeMetric();
    fPrepareSpatialGradients();
    
    for(int k = 0; k < mDim; ++k)
    {
        
        // Approximate metric gradient
        MatrixXd temp = MatrixXd::Zero(mDim, mDim);        
        MatrixXd G = MatrixXd::Zero(mDim, mDim);

        mQ(k) += epsilon;
        fComputeMetric();
        G.noalias() = mEigenDeco.eigenvectors() * mSoftAbsLambda.asDiagonal() * mEigenDeco.eigenvectors().transpose();
        temp += G;
        mQ(k) -= 2.0 * epsilon;
        fComputeMetric();
        G.noalias() = mEigenDeco.eigenvectors() * mSoftAbsLambda.asDiagonal() * mEigenDeco.eigenvectors().transpose();
        temp -= G;
        mQ(k) += epsilon;
        
        temp /= 2.0 * epsilon;

        // Exact metric gradient
        fComputeMetric();
        fComputeGradH(k);
        
        mAuxMatrixOne.noalias() = mGradH.block(0, k * mDim, mDim, mDim) * mEigenDeco.eigenvectors();
        mAuxMatrixTwo.noalias() = mEigenDeco.eigenvectors().transpose() * mAuxMatrixOne;
        mAuxMatrixOne.noalias() = mPseudoJ.cwiseProduct(mAuxMatrixTwo);
        mCacheMatrix.noalias() = mAuxMatrixOne * mEigenDeco.eigenvectors().transpose();
        
        MatrixXd gradG = mEigenDeco.eigenvectors() * mCacheMatrix;
        
        // Compare
        for(int i = 0; i < mDim; ++i)
        {
            
            for(int j = 0; j < mDim; ++j)
            {
                
                std::cout << "    "
                          << std::setw(width) << std::left << k
                          << std::setw(width) << std::left << i
                          << std::setw(width) << std::left << j 
                          << std::setw(width) << std::left << gradG(i, j)
                          << std::setw(width) << std::left << temp(i, j)
                          << std::setw(width) << std::left << (gradG(i, j) - temp(i, j)) / (epsilon * epsilon)
                          << std::endl;
                
            }
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Hamiltonian
    VectorXd gradH = dTaudq();
    gradH += dPhidq();
    
    std::cout << "pDot (-dH/dq^{i}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta /"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Epsilon^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    for(int i = 0; i < mDim; ++i)
    {
        
        // Finite Differences
        double temp = 0.0;        
        
        mQ(i) += epsilon;
        temp -= H();
        mQ(i) -= 2.0 * epsilon;
        temp += H();
        mQ(i) += epsilon;
        
        temp /= 2.0 * epsilon;
        
        // Exact
        double minusGradH = -gradH(i);

        std::cout << "    "
                  << std::setw(width) << std::left << i 
                  << std::setw(width) << std::left << minusGradH
                  << std::setw(width) << std::left << temp
                  << std::setw(width) << std::left << (minusGradH - temp) / (epsilon * epsilon)
                  << std::endl;
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

void softAbsMetric::displayState()
{
    
    baseHamiltonian::displayState();
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 3;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    " 
              << std::setw(width) << std::left << "Row"
              << std::setw(width) << std::left << "Column"
              << std::setw(width) << std::left << "Metric"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    fComputeMetric();
    MatrixXd G = MatrixXd::Zero(mDim, mDim);
    G.noalias() = mEigenDeco.eigenvectors() * mSoftAbsLambda.asDiagonal() * mEigenDeco.eigenvectors().transpose();
    
    for(int i = 0; i < mDim; ++i)
    {
        for(int j = 0; j < mDim; ++j)
        {
            std::cout << "    " 
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << j
                      << std::setw(width) << std::left << G(i, j)
                      << std::endl;
        }
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Compute the metric at the current position, performing
/// an eigen decomposition and computing the log determinant
void softAbsMetric::fComputeMetric()
{
    
    // Compute the Hessian
    fComputeH();
    
    // Compute the eigen decomposition of the Hessian,
    // then perform the soft-abs transformation
    mEigenDeco.compute(mH);

    for(int i = 0; i < mDim; ++i)
    {
        
        const double lambda = mEigenDeco.eigenvalues()(i);
        const double alphaLambda = mSoftAbsAlpha * lambda;
        
        double softAbsLambda = 0;
        if(fabs(alphaLambda) < 1e-4)
        {
            softAbsLambda = (1.0 + (1.0 / 3.0) * alphaLambda * alphaLambda) / mSoftAbsAlpha;
        }
        else if(fabs(alphaLambda) > 18)
        {
            softAbsLambda = fabs(lambda);
        }
        else
        {
            softAbsLambda = lambda / tanh(alphaLambda);
        }
        
        mSoftAbsLambda(i) = softAbsLambda;
        mSoftAbsLambdaInv(i) = 1.0 /softAbsLambda;
    }
    
    // Helpful auxiliary calcs
    mQp.noalias() = mEigenDeco.eigenvectors().transpose() * mP;
    mLambdaQp.noalias() = mSoftAbsLambdaInv.cwiseProduct(mQp);
    
    // Compute the log determinant of the metric
    mLogDetMetric = 0;
    for(int i = 0; i < mDim; ++i) mLogDetMetric += log(mSoftAbsLambda(i));
    
}

/// Compute intermediate values necessary for the spatial gradients dTaudq and dPhidq

void softAbsMetric::fPrepareSpatialGradients()
{
    
    // Compute the discrete Jacobian of the SoftAbs transform
    double delta = 0;
    double lambda = 0;
    double alphaLambda = 0;
    double sdx = 0;
    
    for(int i = 0; i < mDim; ++i)
    {
        
        for(int j = 0; j <= i; ++j)
        {
  
            delta = mEigenDeco.eigenvalues()(i) - mEigenDeco.eigenvalues()(j);
            
            if(fabs(delta) < 1e-10)
            {
                
                lambda = mEigenDeco.eigenvalues()(i);
                alphaLambda = mSoftAbsAlpha * lambda;
                
                if(fabs(alphaLambda) < 1e-4)
                {
                    mPseudoJ(i, j) = (2.0 / 3.0) * alphaLambda * (1.0 - (2.0 / 15.0) * alphaLambda * alphaLambda);
                }
                else if(fabs(alphaLambda) > 18)
                {
                    mPseudoJ(i, j) = lambda > 0 ? 1 : -1;
                }
                else
                {
                    sdx = sinh(mSoftAbsAlpha * lambda) / lambda;
                    mPseudoJ(i, j) = (mSoftAbsLambda(i) - mSoftAbsAlpha / (sdx * sdx) ) / lambda;
                }
                
            }
            else
            {
                mPseudoJ(i, j) = ( mSoftAbsLambda(i) - mSoftAbsLambda(j) ) / delta;
            }
                
        }
        
    }

    // And make sure the gradient has been calculated
    gradV();
    
    // Along with the third-derivative tensor
    for(int i = 0; i < mDim; ++i) fComputeGradH(i);
    
}

void softAbsMetric::fUpdateP()
{
    mQp.noalias() = mEigenDeco.eigenvectors().transpose() * mP;
    mLambdaQp.noalias() = mSoftAbsLambdaInv.cwiseProduct(mQp);
}

VectorXd& softAbsMetric::dTaudp()
{
    mAuxVector.noalias() = (mEigenDeco.eigenvectors() * mLambdaQp);
    return mAuxVector;
}

VectorXd& softAbsMetric::dTaudq()
{
 
    // Cache some handy intermediate values...
    mAuxMatrixOne.noalias() = mLambdaQp.asDiagonal() * mEigenDeco.eigenvectors().transpose();
    mAuxMatrixTwo.noalias() = mPseudoJ.selfadjointView<Eigen::Lower>() * mAuxMatrixOne;
    
    mCacheMatrix.setZero();
    mCacheMatrix.triangularView<Eigen::Lower>() = mAuxMatrixOne.transpose() * mAuxMatrixTwo;

    // Now let's finally compute some gradients
    for(int i = 0; i < mDim; ++i)
    {
        
        mAuxVector(i) = 0;
        
        for(int j = 0; j < mDim; ++j)
        {
            mAuxVector(i) += mGradH.block(0, i * mDim, mDim, mDim).col(j).dot( mCacheMatrix.row(j) 
                          + mCacheMatrix.col(j).transpose() ) - mCacheMatrix(j, j) * mGradH.block(0, i * mDim, mDim, mDim)(j, j);
        }
        
    }
    
    mAuxVector *= -0.5;
    
    return mAuxVector;
    
}

VectorXd& softAbsMetric::dPhidq()
{
    
    // Cache some handy intermediate values...
    mAuxVector = mSoftAbsLambdaInv.cwiseProduct(mPseudoJ.diagonal());
    mAuxMatrixTwo.noalias() = mAuxVector.asDiagonal() * mEigenDeco.eigenvectors().transpose();
    
    mCacheMatrix.setZero();
    mCacheMatrix.triangularView<Eigen::Lower>() = mEigenDeco.eigenvectors() * mAuxMatrixTwo;
    
    // Now let's finally compute some gradients
    for(int i = 0; i < mDim; ++i)
    {
        
        mAuxVector(i) = 0;
        
        for(int j = 0; j < mDim; ++j)
        {
            mAuxVector(i) += mGradH.block(0, i * mDim, mDim, mDim).col(j).dot( mCacheMatrix.row(j) 
                          + mCacheMatrix.col(j).transpose() ) - mCacheMatrix(j, j) * mGradH.block(0, i * mDim, mDim, mDim)(j, j);
        }
        
    }
    
    mAuxVector *= 0.5;
    mAuxVector += mGradV;
    
    return mAuxVector;
    
}
