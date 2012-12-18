#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "diagSoftAbsMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

diagSoftAbsMetric::diagSoftAbsMetric(int dim): 
dynamMetric(dim),
mDiagH(VectorXd::Zero(mDim)),
mGradDiagH(MatrixXd::Identity(mDim, mDim)),
mLambda(VectorXd::Zero(mDim)),
mGradHelper(VectorXd::Zero(mDim)),
mSoftAbsAlpha(1.0)
{}

double diagSoftAbsMetric::T()
{
    fComputeMetric();
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return 0.5 * mP.dot(mAuxVector) + 0.5 * mLogDetMetric;
}

double diagSoftAbsMetric::tau()
{
    fComputeMetric();
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return 0.5 * mP.dot(mAuxVector);
}

void diagSoftAbsMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector.noalias() = mLambda.cwiseProduct(normal);
    
    double C = -2.0 * mP.dot(mAuxVector);
    C /= normal.dot(mAuxVector);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void diagSoftAbsMetric::sampleP(Random& random)
{
    
    fComputeMetric();
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) 
    {
        mP(i) = g(random, 0.0, 1.0) / sqrt(mLambda(i));
    }

}

void diagSoftAbsMetric::checkEvolution(const double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    // Hessian Gradient
    std::cout.precision(6);
    int width = 12;
    int nColumn = 5;
    
    fComputeGradDiagH();
    
    std::cout << "Gradient of the Diagonal Hessian (d^{3}V/dq_{i}dq^{j}dq^{j}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component" 
              << std::setw(width) << std::left << "Row/Column" 
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
    
    for(int i = 0; i < mDim; ++i)
    {
        
        VectorXd temp = VectorXd::Zero(mDim);        
        
        mQ(i) += epsilon;
        fComputeDiagH();
        temp += mDiagH;
        mQ(i) -= 2.0 * epsilon;
        fComputeDiagH();
        temp -= mDiagH;
        mQ(i) += epsilon;
        
        temp /= 2.0 * epsilon;
        
        for(int j = 0; j < mDim; ++j)
        {
            
            std::cout << "    "
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << j 
                      << std::setw(width) << std::left << mGradDiagH(i, j)
                      << std::setw(width) << std::left << temp(j)
                      << std::setw(width) << std::left << (mGradDiagH(i, j) - temp(j)) / (epsilon * epsilon)
                      << std::endl;
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Metric
    std::cout.precision(6);
    width = 12;
    nColumn = 5;
    
    std::cout << "Gradient of the inverse metric (dLambda^{jj}/dq^{i}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Row/Column" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta /"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "(j)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Epsilon^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    fComputeMetric();
    fComputeGradDiagH();
    
    for(int i = 0; i < mDim; ++i)
    {
        
        const double hLambda = mDiagH(i) * mLambda(i);
        double v = 1.0 / mDiagH(i);
        if(fabs(mSoftAbsAlpha * mDiagH(i)) < 20) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mGradDiagH.col(i) *= -mLambda(i) * v;
    }
    
    for(int k = 0; k < mDim; ++k)
    {
        
        // Approximate metric gradient
        VectorXd temp = VectorXd::Zero(mDim);
        
        mQ(k) += epsilon;
        fComputeMetric();
        temp += mLambda;
        mQ(k) -= 2.0 * epsilon;
        fComputeMetric();
        temp -= mLambda;
        mQ(k) += epsilon;
        
        temp /= 2.0 * epsilon;
        
        // Compare
        for(int i = 0; i < mDim; ++i)
        {
    
                std::cout << "    "
                          << std::setw(width) << std::left << k
                          << std::setw(width) << std::left << i
                          << std::setw(width) << std::left << mGradDiagH(k, i)
                          << std::setw(width) << std::left << temp(i)
                          << std::setw(width) << std::left << (mGradDiagH(k, i) - temp(i)) / (epsilon * epsilon)
                          << std::endl;
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Hamiltonian
    gradV();
    mInit = -dTaudq();
    mInit -= dPhidq();
    
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
        double minusGradH = mInit(i);
        
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

void diagSoftAbsMetric::displayState()
{
    
    baseHamiltonian::displayState();
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 2;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    " 
              << std::setw(width) << std::left << "Row/Column"
              << std::setw(width) << std::left << "Inverse Metric"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    

    fComputeMetric();
    
    for(int i = 0; i < mDim; ++i)
    {

            std::cout << "    " 
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << mLambda(i)
                      << std::endl;

    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Compute the metric at the current position, performing
/// an eigen decomposition and computing the log determinant
void diagSoftAbsMetric::fComputeMetric()
{
    
    // Compute the Hessian
    fComputeDiagH();

    for(int i = 0; i < mDim; ++i)
    {
        
        const double lambda = mDiagH(i);
        const double alphaLambda = mSoftAbsAlpha * lambda;
        
        if(fabs(alphaLambda) < 1e-4)
        {
            mLambda(i) = mSoftAbsAlpha * ( 1.0 - (1.0 / 3.0) * alphaLambda * alphaLambda );
        }
        else if(fabs(alphaLambda > 18))
        {
            mLambda(i) = 1 / fabs(lambda);       
        }
        else
        {
            mLambda(i) = tanh(mSoftAbsAlpha * lambda) / lambda;
        }

    }
    
    // Compute the log determinant of the metric
    mLogDetMetric = 0;
    for(int i = 0; i < mDim; ++i) mLogDetMetric -= log(mLambda(i));
    
}

VectorXd& diagSoftAbsMetric::dTaudp()
{
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return mAuxVector;
}

VectorXd& diagSoftAbsMetric::dTaudq()
{
    
    for(int i = 0; i < mDim; ++i)
    {
        
        const double hLambda = mDiagH(i) * mLambda(i);
        double v = 1.0 / mDiagH(i);
        if(fabs(mSoftAbsAlpha * mDiagH(i)) < 18) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mGradHelper(i) = - 0.5 * mLambda(i) * mP(i) * mP(i) * v;
    }
    
    mAuxVector.noalias() = mGradDiagH * mGradHelper;
    
    return mAuxVector;
    
}

VectorXd& diagSoftAbsMetric::dPhidq()
{
    
    for(int i = 0; i < mDim; ++i)
    {
        
        const double hLambda = mDiagH(i) * mLambda(i);
        double v = 1.0 / mDiagH(i);
        if(fabs(mSoftAbsAlpha * mDiagH(i)) < 18) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mGradHelper(i) = 0.5 * v;
    }
    
    mAuxVector.noalias() = mGradDiagH * mGradHelper + mGradV;
    
    return mAuxVector;
    
}


