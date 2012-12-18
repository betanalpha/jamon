#include "denseConstMetric.h"

#include <RandomLib/NormalDistribution.hpp>

/// Constructor             
/// \param dim Dimension of the target space
/// \param massInv Euclidean metric

denseConstMetric::denseConstMetric(int dim, MatrixXd& massInv): 
constMetric(dim),
mMassL(dim)
{
    
    // Compute the mass matrix, using the mass matrix as temporary storage
    mMassL.compute(massInv.selfadjointView<Eigen::Lower>());
    mMassInv = mMassL.solve(MatrixXd::Identity(mDim, mDim));
    
    // Compute the cholesky decomposition of the background metric
    mMassL.compute(mMassInv);
    
    // Set the inverse mass matrix to its final values
    mMassInv = massInv;
    
}

double denseConstMetric::T()
{
    mAuxVector.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * mP;
    return 0.5 * mP.dot(mAuxVector);
}

void denseConstMetric::evolveQ(const double epsilon)
{
    mAuxVector.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * mP;
    mQ += epsilon * mAuxVector;
}

void denseConstMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * normal;
    double C = -2.0 * mP.dot(mAuxVector);
    C /= normal.dot(mAuxVector);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void denseConstMetric::sampleP(Random& random)
{
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAuxVector(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mMassL.matrixL() * mAuxVector;
    
}

/// Set the inverse mass matrix and recompute the Cholesky decomposition

void denseConstMetric::setInvMass(MatrixXd& massInv)
{

    mMassInv = massInv;
    
    // Compute the mass matrix, using the argument as temporary storage
    mMassL.compute(mMassInv.selfadjointView<Eigen::Lower>());
    massInv = mMassL.solve(MatrixXd::Identity(mDim, mDim));
    
    // Compute the cholesky decomposition of the background metric
    mMassL.compute(massInv);
    
    // Restore initial state of argument
    massInv = mMassInv;
    
}
