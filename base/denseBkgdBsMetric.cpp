#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "denseBkgdBsMetric.h"

/// Constructor             
/// \param dim Dimension of the target space
/// \param backLambda Background metric

denseBkgdBsMetric::denseBkgdBsMetric(int dim, MatrixXd& backLambda): 
denseDynamMetric(dim),
mBackLambda(backLambda),
mSigmaL(dim),
mLambda(MatrixXd::Identity(mDim, mDim)),
mHessianV(MatrixXd::Zero(mDim, mDim)),
mGradLogDetLambda(VectorXd::Zero(mDim)),
mDetLambda(0),
mAlpha(1)
{
    
    // Compute the background metric
    mSigmaL.compute(mBackLambda.selfadjointView<Eigen::Lower>());
    mHessianV = mSigmaL.solve(MatrixXd::Identity(mDim, mDim));
    
    // Compute the cholesky decomposition of the background metric
    mSigmaL.compute(mHessianV);
    
}

double denseBkgdBsMetric::T()
{
    fComputeLambda();
    mAux.noalias() = mLambda.selfadjointView<Eigen::Lower>() * mP;
    return 0.5 * mP.dot(mAux) - 0.5 * log(mDetLambda);
}

void denseBkgdBsMetric::evolveQ(double epsilon)
{
    
    mB = mQ;
    mB.noalias() += 0.5 * epsilon * (mLambda.selfadjointView<Eigen::Lower>() * mP);
    
    for(int i = 0; i < mNumFixedPoint; ++i)
    {
        mAux.noalias() = 0.5 * epsilon * (mLambda.selfadjointView<Eigen::Lower>() * mP);
        mQ = mB + mAux;
        fComputeLambda();
    }
    
    fComputeGradLogDetLambda();
    
}

void denseBkgdBsMetric::beginEvolveP(double epsilon)
{
    
    // \hat{F}
    fHatF(epsilon);
    
    // \hat{\mathbb{A}}
    fHatA(epsilon);
    
}

void denseBkgdBsMetric::finishEvolveP(double epsilon)
{

    // \hat{\mathbb{A}}
    fHatA(epsilon);

    // \hat{F}
    fHatF(epsilon);
    
}

void denseBkgdBsMetric::bounceP(const VectorXd& normal)
{
    
    mAux.noalias() = mLambda.selfadjointView<Eigen::Lower>() * normal;
    double C = -2.0 * mP.dot(mAux);
    C /= normal.dot(mAux);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void denseBkgdBsMetric::sampleP(Random& random)
{

    // The below code would be great, if LLT::rankUpdate were in production...
    /*
    // Store the Cholesky matrix in the Hessian
    mHessianV = mSigmaL.matrixL();
    
    // Compute the rank-one update to the background Cholesky matrix
    mAuxOne = - minusGradV();
    mSigmaL.rankUpdate(mAuxOne);
    
    // Sample the momenta
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAuxOne(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mSigmaL.matrixL() * mAuxOne;
    
    mSigmaL.matrixL() = mHessianV.triangularView<Eigen::Lower>();
    */
    
    // Store the Cholesky matrix in the Hessian
    mHessianV = mSigmaL.matrixL();
    
    // Compute the rank-one update to the background Cholesky matrix,
    // see http://lapmal.epfl.ch/papers/cholupdate.pdf
    gradV();
    
    for(int i = 0; i < mDim; ++i)
    {
        
        double v = mGradV(i);
        double L = mHessianV(i, i);
        double r = sqrt(L * L + v * v);
        
        double c = L / r;
        double s = v / r;
        
        mHessianV(i, i) = r;
        
        for(int j = i + 1; j < mDim; ++j)
        {
            
            double vprime = mGradV(j);
            double Lprime = mHessianV(i, j);
            
            mGradV(j) = c * vprime - s * Lprime;
            mHessianV(i, j) = s * vprime + c * Lprime;
            
        }
    }
    
    mHessianV.transposeInPlace();
    
    // Sample the momenta
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAux(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mHessianV.triangularView<Eigen::Lower>() * mAux;
    
}

/// Set the inverse background metric and recompute the Cholesky decomposition

void denseBkgdBsMetric::setBackLambda(MatrixXd& backLambda)
{
    
    mBackLambda = backLambda;
    
    // Compute the background metric
    mSigmaL.compute(mBackLambda.selfadjointView<Eigen::Lower>());
    mHessianV = mSigmaL.solve(MatrixXd::Identity(mDim, mDim));
    
    // Compute the cholesky decomposition of the background metric
    mSigmaL.compute(mHessianV);
    
}

void denseBkgdBsMetric::checkEvolution(double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 4;
    
    // Hessian
    fComputeHessianV();
    
    std::cout << "Hessian (d^{2}V/dq_{i}dq_{j}):" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Row" 
              << std::setw(width) << std::left << "Column" 
              << std::setw(width) << std::left << "Analytic"
              << std::setw(width) << std::left << "Finite"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << ""
              << std::setw(width) << std::left << ""
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
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
                      << std::setw(width) << std::left << mHessianV(i, j)
                      << std::setw(width) << std::left << temp(j)
                      << std::endl;
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Metric
    std::cout.precision(6);
    width = 12;
    nColumn = 6;
    
    std::cout << "Gradient of the Background-Score metric:" << std::endl;
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
    
    gradV();
    fComputeHessianV();
    
    VectorXd auxOne = VectorXd::Zero(mDim);
    auxOne.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    
    VectorXd auxTwo = VectorXd::Zero(mDim);
    
    for(int k = 0; k < mDim; ++k)
    {
        
        MatrixXd tempOne = MatrixXd::Zero(mDim, mDim);        
        
        mQ(k) += epsilon;
        fComputeLambda();
        tempOne += mLambda;
        mQ(k) -= 2.0 * epsilon;
        fComputeLambda();
        tempOne -= mLambda;
        mQ(k) += epsilon;
        fComputeLambda();
        
        tempOne /= 2.0 * epsilon;
        
        auxTwo.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mHessianV.col(k);
        
        MatrixXd tempTwo = mLambda.selfadjointView<Eigen::Lower>();  
        tempTwo -= mBackLambda.selfadjointView<Eigen::Lower>();
        tempTwo *= 2 * mAlpha * mHessianV.row(k).dot(auxOne);
        
        tempTwo += auxOne * auxTwo.transpose();
        tempTwo += auxTwo * auxOne.transpose();
        
        tempTwo *= - mAlpha * mDetLambda;
        
        for(int i = 0; i < mDim; ++i)
        {
            
            for(int j = 0; j < mDim; ++j)
            {
                
                std::cout << "    "
                          << std::setw(width) << std::left << k
                          << std::setw(width) << std::left << i
                          << std::setw(width) << std::left << j 
                          << std::setw(width) << std::left << tempTwo(i, j)
                          << std::setw(width) << std::left << tempOne(i, j)
                          << std::setw(width) << std::left << (tempTwo(i, j) - tempOne(i, j)) / (epsilon * epsilon)
                          << std::endl;
                
            }
            
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
    // Hamiltonian
    nColumn = 4;
    
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
    
    fComputeLambda();
    fComputeGradLogDetLambda();
    
    VectorXd minusGradH = VectorXd::Zero(mDim);
    minusGradH = 0.5 * mGradLogDetLambda;
    minusGradH -= mGradV;
    
    auxOne.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mP;
    auxTwo.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    
    double C = mP.dot(auxTwo) * mDetLambda;
    auxOne -= C * mAlpha * auxTwo;
    
    for(int i = 0; i < mDim; ++i)
    {
        
        // Finite Differences
        double temp = 0.0;        
        
        mQ(i) += epsilon;
        temp -= H();
        mQ(i) -= 2.0 * epsilon;
        temp += H();
        mQ(i) += epsilon;
        H();
        
        temp /= 2.0 * epsilon;
        
        // Exact
        double delQuad = - 2.0 * mAlpha * C * auxOne.dot(mHessianV.row(i));
        
        // Display
        std::cout << "    "
                  << std::setw(width) << std::left << i 
                  << std::setw(width) << std::left << minusGradH(i) - 0.5 * delQuad
                  << std::setw(width) << std::left << temp
                  << std::setw(width) << std::left << (minusGradH(i) - 0.5 * delQuad - temp) / (epsilon * epsilon)
                  << std::endl;
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

void denseBkgdBsMetric::displayState()
{
    
    baseHamiltonian::displayState();
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 3;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    " 
              << std::setw(width) << std::left << "Row"
              << std::setw(width) << std::left << "Column"
              << std::setw(width) << std::left << "Lambda"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    for(int i = 0; i < mDim; ++i)
    {
        for(int j = 0; j < mDim; ++j)
        {
            std::cout << "    " 
                      << std::setw(width) << std::left << i
                      << std::setw(width) << std::left << j
                      << std::setw(width) << std::left << mLambda(i, j)
                      << std::endl;
        }
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Compute the determinant of the inverse metric
void denseBkgdBsMetric::fComputeDetLambda()
{
    mAux.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    mDetLambda = 1.0 + mAlpha * mGradV.dot(mAux);
    mDetLambda = 1.0 / mDetLambda;
}

/// Compute the inverse metric
void denseBkgdBsMetric::fComputeLambda()
{
    
    // Preliminary calculations
    gradV();
    fComputeDetLambda();
    
    // Compute the inverse of the induced metric
    mAux.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    
    mLambda = mBackLambda;
    mLambda.noalias() -= mAlpha * mDetLambda * mAux * mAux.transpose();
    
}

/// Compute the gradient of the log determinant of the inverse induced metric
void denseBkgdBsMetric::fComputeGradLogDetLambda()
{
    
    fComputeHessianV();
    
    mAux.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    mGradLogDetLambda.noalias() = mHessianV * mAux;
    mGradLogDetLambda *= - 2.0 * mAlpha * mDetLambda;
    
}

void denseBkgdBsMetric::fHatA(double epsilon)
{
    
    double alpha = 0;
    double beta = 0;
    double gamma = 0;
    
    double pDotGradV = 0;
    
    mAux.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mGradV;
    mB.noalias() = mBackLambda.selfadjointView<Eigen::Lower>() * mP;
    
    for(int i = 0; i < mDim - 1; ++i) 
    {
        
        double pOld = mP(i);
        
        pDotGradV = mP.dot(mAux);
        
        mN = - mDetLambda * mAlpha * mAux(i) * mAux;
        mN += mBackLambda.col(i);
        
        alpha = mAux(i) * mN.dot(mHessianV.row(i));
        
        mN = mB - mDetLambda * mAlpha * pDotGradV * mAux;
        
        beta = 2.0 * (mAux(i) * mN.dot(mHessianV.row(i)) - alpha * pOld);
        
        gamma = pDotGradV * mN.dot(mHessianV.row(i)) - beta * pOld - alpha * pOld * pOld;
        
        alpha *= mAlpha * mDetLambda;
        beta *= mAlpha * mDetLambda;
        gamma *= mAlpha * mDetLambda;
        
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
        
        // Necessary rank one updates
        mB += (mP(i) - pOld) * mBackLambda.col(i);
        
    }
    
    {
        
        int i = mDim - 1;
        
        double pOld = mP(i);
        
        pDotGradV = mP.dot(mAux);
        
        mN = - mDetLambda * mAlpha * mAux(i) * mAux;
        mN += mBackLambda.col(i);
        
        alpha = mAux(i) * mN.dot(mHessianV.row(i));
        
        mN = mB - mDetLambda * mAlpha * pDotGradV * mAux;
        
        beta = 2.0 * (mAux(i) * mN.dot(mHessianV.row(i)) - alpha * pOld);
        
        gamma = pDotGradV * mN.dot(mHessianV.row(i)) - beta * pOld - alpha * pOld * pOld;
        
        alpha *= mAlpha * mDetLambda;
        beta *= mAlpha * mDetLambda;
        gamma *= mAlpha * mDetLambda;
        
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
        
        // Necessary rank one updates
        mB += (mP(i) - pOld) * mBackLambda.col(i);
        
    }
    
    for(int i = mDim - 2; i >= 0; --i) 
    {
        
        double pOld = mP(i);
        
        pDotGradV = mP.dot(mAux);
        
        mN = - mDetLambda * mAux(i) * mAux;
        mN += mBackLambda.col(i);
        
        alpha = mAux(i) * mN.dot(mHessianV.row(i));
        
        mN = mB - mDetLambda * mAlpha * pDotGradV * mAux;
        
        beta = 2.0 * (mAux(i) * mN.dot(mHessianV.row(i)) - alpha * pOld);
        
        gamma = pDotGradV * mN.dot(mHessianV.row(i)) - beta * pOld - alpha * pOld * pOld;
        
        alpha *= mAlpha * mDetLambda;
        beta *= mAlpha * mDetLambda;
        gamma *= mAlpha * mDetLambda;
        
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
        
        // Necessary rank one updates
        mB += (mP(i) - pOld) * mBackLambda.col(i);
        
    }
    
}

void denseBkgdBsMetric::fHatF(double epsilon)
{
    mP -= epsilon * ( 0.5 * mGradLogDetLambda + mGradV );
}