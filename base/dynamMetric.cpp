#include <iostream>
#include <iomanip>

#include "dynamMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

dynamMetric::dynamMetric(int dim): 
baseHamiltonian(dim),
mAuxVector(VectorXd::Zero(mDim)),
mInit(VectorXd::Zero(mDim)),
mDelta(VectorXd::Zero(mDim)),
mMaxNumFixedPoint(10),
mFixedPointThreshold(1e-8),
mLogDetMetric(0)
{}

void dynamMetric::evolveQ(const double epsilon) { fHatT(epsilon); }

void dynamMetric::beginEvolveP(const double epsilon)
{
    fHatPhi(epsilon);
    fHatTau(epsilon, mMaxNumFixedPoint);
}

void dynamMetric::finishEvolveP(const double epsilon)
{
    fHatTau(epsilon, 1);
    fHatPhi(epsilon);
}

void dynamMetric::fHatT(const double epsilon)
{
    
    mInit.noalias() = mQ + 0.5 * epsilon * dTaudp();
    
    for(int i = 0; i < mMaxNumFixedPoint; ++i)
    {
        
        mDelta.noalias() = mQ;
        mQ.noalias() = mInit + 0.5 * epsilon * dTaudp();
        mDelta -= mQ;
        
        fComputeMetric();
        
        if(mDelta.cwiseAbs().maxCoeff() < mFixedPointThreshold) break;
        
    }
    
    fPrepareSpatialGradients();
    
}

void dynamMetric::fHatTau(const double epsilon, const int numFixedPoint)
{
    
    mInit = mP; 
    for(int i = 0; i < numFixedPoint; ++i) 
    {
        
        mDelta.noalias() = mP;
        mP.noalias() = mInit - epsilon * dTaudq();
        mDelta -= mP;
        
        fUpdateP();
        
        if(mDelta.cwiseAbs().maxCoeff() < mFixedPointThreshold) break;
        
    }
    
}

void dynamMetric::fHatPhi(const double epsilon)
{
    mP -= epsilon * dPhidq();
    fUpdateP();
}
