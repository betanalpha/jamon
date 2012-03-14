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
    mAux.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * mP;
    return 0.5 * mP.dot(mAux);
}

void denseConstMetric::evolveQ(double epsilon)
{
    mAux.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * mP;
    mQ += epsilon * mAux;
}

void denseConstMetric::bounceP(const VectorXd& normal)
{
    
    mAux.noalias() = mMassInv.selfadjointView<Eigen::Lower>() * normal;
    double C = -2.0 * mP.dot(mAux);
    C /= normal.dot(mAux);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void denseConstMetric::sampleP(Random& random)
{
    
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAux(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mMassL.matrixL() * mAux;
    
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
