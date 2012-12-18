#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "denseFisherMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

denseFisherMetric::denseFisherMetric(int dim): 
dynamMetric(dim),
mG(MatrixXd::Identity(mDim, mDim)),
mGradG(MatrixXd::Identity(mDim, mDim)),
mC(VectorXd::Zero(mDim)),
mGL(dim)
{}

double denseFisherMetric::T()
{
    fComputeMetric();
    mAuxVector = mGL.solve(mP);
    return 0.5 * mP.transpose() * mAuxVector + 0.5 * mLogDetMetric;
}

double denseFisherMetric::tau()
{
    fComputeMetric();
    mAuxVector = mGL.solve(mP);
    return 0.5 * mP.transpose() * mAuxVector;
}

void denseFisherMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector = mGL.solve(normal);
    
    double C = -2.0 * mP.dot(mAuxVector);
    C /= normal.dot(mAuxVector);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void denseFisherMetric::sampleP(Random& random)
{
    
    fComputeMetric();
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAuxVector(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mGL.matrixL() * mAuxVector;
    
}

void denseFisherMetric::checkEvolution(const double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    // Metric
    std::cout.precision(6);
    int width = 12;
    int nColumn = 6;
    
    std::cout << "Gradient of the Fisher-Rao metric (dG^{jk}/dq^{i}):" << std::endl;
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
        
        fComputeG();
        fComputeGradG(k);
            
        MatrixXd temp = MatrixXd::Zero(mDim, mDim);        
        
        mQ(k) += epsilon;
        fComputeG();
        temp += mG;
        mQ(k) -= 2.0 * epsilon;
        fComputeG();
        temp -= mG;
        mQ(k ) += epsilon;
        
        temp /= 2.0 * epsilon;
            
        for(int i = 0; i < mDim; ++i)
        {
        
            for(int j = 0; j < mDim; ++j)
            {
                
                std::cout << "    "
                << std::setw(width) << std::left << k
                << std::setw(width) << std::left << i
                << std::setw(width) << std::left << j 
                << std::setw(width) << std::left << mGradG(i, j)
                << std::setw(width) << std::left << temp(i, j)
                << std::setw(width) << std::left << (mGradG(i, j) - temp(i, j)) / (epsilon * epsilon)
                << std::endl;
                
            }
            
        }
    
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Hamiltonian
    fComputeCholeskyG();
    mC = mGL.solve(mP);
    
    gradV();
    
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
              << std::setw(width) << std::left << "Stepsize^{2}"
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
        fComputeGradG(i);
        
        double minusGradH = -0.5 * mGL.solve(mGradG).trace();

        mAuxVector = mGradG * mC;
        minusGradH += 0.5 * mC.dot(mAuxVector);
        
        minusGradH -= mGradV(i);
        
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

void denseFisherMetric::displayState()
{
    
    baseHamiltonian::displayState();
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 3;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    " 
              << std::setw(width) << std::left << "Row"
              << std::setw(width) << std::left << "Column"
              << std::setw(width) << std::left << "Fisher-Rao"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    for(int i = 0; i < mDim; ++i)
    {
        for(int j = 0; j < mDim; ++j)
        {
            std::cout << "    " 
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << j
                      << std::setw(width) << std::left << mG(i, j)
                      << std::endl;
        }
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Compute the Cholesky decomposition of the denseFisher-Rao metric at the current position
/// as well as the determinant of the denseFisher-Rao metric

void denseFisherMetric::fComputeCholeskyG()
{
    
    fComputeG();
    mGL.compute(mG);
    
    mLogDetMetric = 0;
    for(int i = 0; i < mDim; ++i) mLogDetMetric += 2.0 * log(mGL.matrixL()(i, i));

}

VectorXd& denseFisherMetric::dTaudp()
{
    mAuxVector = mGL.solve(mP);
    return mAuxVector;
}

VectorXd& denseFisherMetric::dTaudq()
{
    
    mC = mGL.solve(mP);
    
    for(int i = 0; i < mDim; ++i)
    {
        fComputeGradG(i);
        mAuxVector(i) = 0.5 * mC.transpose() * mGradG * mC;
    }
    
    return mAuxVector;
    
}

VectorXd& denseFisherMetric::dPhidq()
{
    
    for(int i = 0; i < mDim; ++i)
    {
        fComputeGradG(i);
        mAuxVector(i) = mGL.solve(mGradG).trace();
    }
    
    mAuxVector *= 0.5;
    mAuxVector += mGradV;
    
    return mAuxVector;
    
}
