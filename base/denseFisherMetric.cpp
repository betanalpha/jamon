#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "denseFisherMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

denseFisherMetric::denseFisherMetric(int dim): 
denseDynamMetric(dim),
mG(MatrixXd::Identity(mDim, mDim)),
mGradG(MatrixXd::Identity(mDim, mDim)),
mC(VectorXd::Zero(mDim)),
mGL(dim),
mLogDetG(0)
{}

double denseFisherMetric::T()
{
    fComputeCholeskyG();
    mAux = mGL.solve(mP);
    return 0.5 * mP.transpose() * mAux + 0.5 * mLogDetG;
}

void denseFisherMetric::evolveQ(double epsilon)
{
    
    mB = mGL.solve(mP);
    mB = mQ + 0.5 * epsilon * mB;
    
    for(int i = 0; i < mNumFixedPoint; ++i)
    {
        mAux = mGL.solve(mP);
        mAux *= 0.5 * epsilon;
        mQ = mB + mAux;
        fComputeCholeskyG();
    }
    
}

void denseFisherMetric::beginEvolveP(double epsilon)
{
    
    /*
    mC = mP;

    for(int i = 0; i < mNumFixedPoint; ++i)
    {
    
        for(int j = 0; j < mDim; ++j)
        {
            
            fComputeGradG(j);
       
            mB = mGL.solve(mP);
            mAux(j) = 0.5 * mB.transpose() * mGradG * mB;
            
            mAux(j) += - 0.5 * mGL.solve(mGradG).trace();
            
        }
        
        mP = mC + epsilon * (mAux - mGradV);
        
    }
     */
    
    // \hat{F}
    fHatF(epsilon);
    
    // \hat{\mathbb{A}}
    fHatA(epsilon);
    
}

void denseFisherMetric::finishEvolveP(double epsilon)
{
    
    /*
    for(int j = 0; j < mDim; ++j)
    {
        
        fComputeGradG(j);
        
        mB = mGL.solve(mP);
        mAux(j) = 0.5 * mB.transpose() * mGradG * mB;
        
        mAux(j) += - 0.5 * mGL.solve(mGradG).trace();
        
    }
    
    mP += epsilon * (mAux - mGradV);
    */
    
    // \hat{\mathbb{A}}
    fHatA(epsilon);
    
    // \hat{F}
    fHatF(epsilon);
    
}

void denseFisherMetric::bounceP(const VectorXd& normal)
{
    
    fComputeCholeskyG();
    mAux = mGL.solve(normal);
    
    double C = -2.0 * mP.dot(mAux);
    C /= normal.dot(mAux);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void denseFisherMetric::sampleP(Random& random)
{
    
    fComputeCholeskyG();
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAux(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mGL.matrixL() * mAux;
    
}

void denseFisherMetric::checkEvolution(double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    // Metric
    std::cout.precision(6);
    int width = 12;
    int nColumn = 6;
    
    std::cout << "Gradient of the Fisher-Rao metric:" << std::endl;
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
              << std::setw(width) << std::left << ""
              << std::setw(width) << std::left << ""
              << std::setw(width) << std::left << ""
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
    mB = mGL.solve(mP);
    
    gradV();
    
    std::cout << "pDot (-dH/dq_{i}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta /"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << ""
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

        mAux = mGradG * mB;
        minusGradH += 0.5 * mB.dot(mAux);
        
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
    
    mLogDetG = 0;
    for(int i = 0; i < mDim; ++i) mLogDetG += 2.0 * log(mGL.matrixL()(i, i));

}

void denseFisherMetric::fHatA(double epsilon)
{
    
    double alpha = 0;
    double beta = 0;
    double gamma = 0;
    
    for(int i = 0; i < mDim - 1; ++i) 
    {
        
        mB.setZero();
        mB(i) = mP(i);
        
        mC = mGL.solve(mB);
        mB = mGL.solve(mP);
        
        fComputeGradG(i);
        
        mAux = mGradG * mC;
        alpha = mC.dot(mAux);
        
        mAux = mGradG * mB;
        beta = 2.0 * (mC.dot(mAux) - alpha);
        
        gamma = mB.dot(mAux) - alpha - beta;
        
        alpha /= 2.0 * mP(i) * mP(i);
        beta /= 2.0 * mP(i);
        gamma /= 2.0;
        
        // \hat{\gamma}
        mP(i) += 0.25 * epsilon * gamma;
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.25 * epsilon * alpha * mP(i) );
        
        // \hat{\beta}
        mP(i) *= exp( 0.5 * epsilon * beta );
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.25 * epsilon * alpha * mP(i) );
        
        // \hat{\gamma}
        mP(i) += 0.25 * epsilon * gamma;
        
    }
    
    {
        
        int i = mDim - 1;
        
        mB.setZero();
        mB(i) = mP(i);
        
        mC = mGL.solve(mB);
        mB = mGL.solve(mP);
        
        fComputeGradG(i);
        
        mAux = mGradG * mC;
        alpha = mC.dot(mAux);
        
        mAux = mGradG * mB;
        beta = 2.0 * (mC.dot(mAux) - alpha);
        
        gamma = mB.dot(mAux) - alpha - beta;
        
        alpha /= 2.0 * mP(i) * mP(i);
        beta /= 2.0 * mP(i);
        gamma /= 2.0;
        
        // \hat{\gamma}
        mP(i) += 0.5 * epsilon * gamma;
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.5 * epsilon * alpha * mP(i) );
        
        // \hat{\beta}
        mP(i) *= exp( epsilon * beta );
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.5 * epsilon * alpha * mP(i) );
        
        // \hat{\gamma}
        mP(i) += 0.5 * epsilon * gamma;
        
    }
    
    for(int i = mDim - 2; i >= 0; --i) 
    {
        
        mB.setZero();
        mB(i) = mP(i);
        
        mC = mGL.solve(mB);
        mB = mGL.solve(mP);
        
        fComputeGradG(i);
        
        mAux = mGradG * mC;
        alpha = mC.dot(mAux);
        
        mAux = mGradG * mB;
        beta = 2.0 * (mC.dot(mAux) - alpha);
        
        gamma = mB.dot(mAux) - alpha - beta;
        
        alpha /= 2.0 * mP(i) * mP(i);
        beta /= 2.0 * mP(i);
        gamma /= 2.0;
        
        // \hat{\gamma}
        mP(i) += 0.25 * epsilon * gamma;
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.25 * epsilon * alpha * mP(i) );
        
        // \hat{\beta}
        mP(i) *= exp( 0.5 * epsilon * beta );
        
        // \hat{\alpha}
        mP(i) /= (1 - 0.25 * epsilon * alpha * mP(i) );
        
        // \hat{\gamma}
        mP(i) += 0.25 * epsilon * gamma;
        
    }
    
}

void denseFisherMetric::fHatF(double epsilon)
{
    
    for(int i = 0; i < mDim; ++i)
    {
        fComputeGradG(i);
        mAux(i) = - 0.5 * mGL.solve(mGradG).trace();
    }
    
    mP += epsilon * (mAux - mGradV);
    
}
