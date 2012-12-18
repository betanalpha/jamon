#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "diagOuterSoftAbsMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

diagOuterSoftAbsMetric::diagOuterSoftAbsMetric(int dim): 
dynamMetric(dim),
mH(MatrixXd::Identity(mDim, mDim)),
mLambda(VectorXd::Zero(mDim)),
mGradHelper(VectorXd::Zero(mDim)),
mSoftAbsAlpha(1)
{}

double diagOuterSoftAbsMetric::T()
{
    fComputeMetric();
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return 0.5 * mAuxVector.dot(mP) + 0.5 * mLogDetMetric;
}

double diagOuterSoftAbsMetric::tau()
{
    fComputeMetric();
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return 0.5 * mAuxVector.dot(mP);
}

void diagOuterSoftAbsMetric::bounceP(const VectorXd& normal)
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
void diagOuterSoftAbsMetric::sampleP(Random& random)
{
    
    fComputeMetric();
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) 
    {
        mP(i) = g(random, 0.0, 1.0) / sqrt(mLambda(i));
    }
    
}

void diagOuterSoftAbsMetric::checkEvolution(const double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);

    std::cout.precision(6);
    int width = 12;
    int nColumn = 5;
    
    // Hessian
    fComputeH();
    
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
    
    // Analytic gradient
    fComputeMetric();
    fComputeH();
    gradV();

    for(int i = 0; i < mDim; ++i)
    {
        
        const double g2 = mGradV(i) * mGradV(i);
        const double hLambda = g2 * mLambda(i);
        double v = 1.0 / g2;
        if(fabs(mSoftAbsAlpha * g2) < 20) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mH.col(i) *= -2.0 * mLambda(i) * mGradV(i) * v;
        
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
                      << std::setw(width) << std::left << mH(k, i)
                      << std::setw(width) << std::left << temp(i)
                      << std::setw(width) << std::left << (mH(k, i) - temp(i)) / (epsilon * epsilon)
                      << std::endl;
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Hamiltonian
    prepareEvolution();
    mInit = - dTaudq();
    mInit -=  dPhidq();
    
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

void diagOuterSoftAbsMetric::displayState()
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
void diagOuterSoftAbsMetric::fComputeMetric()
{
    
    // Compute the Hessian
    gradV();
    
    for(int i = 0; i < mDim; ++i)
    {
        
        const double lambda = mGradV(i) * mGradV(i);
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

VectorXd& diagOuterSoftAbsMetric::dTaudp()
{
    mAuxVector.noalias() = mLambda.cwiseProduct(mP);
    return mAuxVector;
}

VectorXd& diagOuterSoftAbsMetric::dTaudq()
{

    for(int i = 0; i < mDim; ++i)
    {
        
        const double g2 = mGradV(i) * mGradV(i);
        const double hLambda = g2 * mLambda(i);
        double v = 1.0 / g2;
        if(fabs(mSoftAbsAlpha * g2) < 18) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mGradHelper(i) = - mLambda(i) * mP(i) * mP(i) * mGradV(i) * v;
        
    }
    
    mAuxVector.noalias() = mH * mGradHelper;
    
    return mAuxVector;
    
}

VectorXd& diagOuterSoftAbsMetric::dPhidq()
{

    for(int i = 0; i < mDim; ++i)
    {
        
        const double g2 = mGradV(i) * mGradV(i);
        const double hLambda = g2 * mLambda(i);
        double v = 1.0 / g2;
        if(fabs(mSoftAbsAlpha * g2) < 18) v += mSoftAbsAlpha * (hLambda - 1.0 / hLambda);
        
        mGradHelper(i) = mGradV(i) * v;
        
    }
    
    mAuxVector.noalias() = mH * mGradHelper + mGradV;
    
    return mAuxVector;
    
}

