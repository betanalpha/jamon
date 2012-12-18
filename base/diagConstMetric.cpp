#include "diagConstMetric.h"

#include <RandomLib/NormalDistribution.hpp>

/// Constructor             
/// \param dim Dimension of the target space    

diagConstMetric::diagConstMetric(int dim): 
constMetric(dim),
mMassInv(VectorXd::Ones(mDim))
{}

double diagConstMetric::T()
{
    mAuxVector.noalias() = mMassInv.asDiagonal() * mP;
    return 0.5 * mP.dot(mAuxVector);
}

void diagConstMetric::evolveQ(const double epsilon)
{
    mAuxVector.noalias() = mMassInv.asDiagonal() * mP;
    mQ += epsilon * mAuxVector;
}

void diagConstMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector.noalias() = mMassInv.asDiagonal() * normal;
    double C = -2.0 * mP.dot(mAuxVector);
    C /= normal.dot(mAuxVector);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void diagConstMetric::sampleP(Random& random)
{
    
    RandomLib::NormalDistribution<> g;
    
    for(int i = 0; i < mDim; ++i)
    {
        mP(i) = g(random, 0, sqrt(1.0 / mMassInv(i)));
    }
    
}