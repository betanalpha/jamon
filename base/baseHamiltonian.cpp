#include <iostream>
#include <iomanip>

#include "baseHamiltonian.h"

/// Constructor             
/// \param dim Dimension of the target space   
baseHamiltonian::baseHamiltonian(int dim): 

mDim(dim),

mQ(VectorXd::Zero(mDim)),
mStoreQ(VectorXd::Zero(mDim)),
mRejectQ(VectorXd::Zero(mDim)),
mAcceptQ(VectorXd::Zero(mDim)),

mP(VectorXd::Zero(mDim)),
mStoreP(VectorXd::Zero(mDim)),

mN(VectorXd::Zero(mDim)),

mGradV(VectorXd::Zero(mDim)),

mNumAccept(0),
mNumReject(0),

mAcceptRateBar(0),
mEffN(0),
mMovingAlpha(0.5)

{}

/// Update Metroplis accept rate statistics

void baseHamiltonian::updateMetroStats(double a)
{
    
    mNumAccept += a;
    mNumReject += !a;
    
    mAcceptRateBar = mMovingAlpha * mEffN * mAcceptRateBar + a;
    mEffN = mMovingAlpha * mEffN + 1.0;
    mAcceptRateBar /= mEffN;
    
}

/// Clear all convergence diagnostic statistics 

void baseHamiltonian::clearHistory()
{
    
    mNumAccept = 0;
    mNumReject = 0;
    
    mAcceptRateBar = 0;
    mEffN = 0;
    
}

/// Compare the position and momentum evolution
/// implementations with finite differences
/// \param epsilon Size of finite difference step

void baseHamiltonian::checkEvolution(double epsilon)
{

    std::cout.precision(6);
    int width = 12;
    int nColumn = 4;
    
    // Potential energy gradient
    const VectorXd& dVdq = gradV();
    
    std::cout << "Potential Gradient (dV/dq):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta / "
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
        
        double delV = 0;
        
        mQ(i) += epsilon;
        delV += V();
        mQ(i) -= 2.0 * epsilon;
        delV -= V();
        mQ(i) += epsilon;
        
        delV /= 2.0 * epsilon;
        
        std::cout << "    "
                  << std::setw(width) << std::left << i 
                  << std::setw(width) << std::left << dVdq(i)
                  << std::setw(width) << std::left << delV 
                  << std::setw(width) << std::left << (dVdq(i) - delV) / (epsilon * epsilon)
                  << std::endl;
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
        
}

/// Display the current state of the chain

void baseHamiltonian::displayState()
{
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 3;
    
    std::cout << "\tH = " << H() << ", T = " << T() << ", V = " << V() << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    " 
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Position"
              << std::setw(width) << std::left << "Momentum"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    for(int i = 0; i < mDim; ++i)
    {
        std::cout << "    " 
                  << std::setw(width) << std::left << i
                  << std::setw(width) << std::left << mQ(i)
                  << std::setw(width) << std::left << mP(i)
                  << std::endl;
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}
