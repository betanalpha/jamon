#include <iostream>
#include <iomanip>

#include <RandomLib/NormalDistribution.hpp>

#include "outerSoftAbsMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

outerSoftAbsMetric::outerSoftAbsMetric(int dim): 
dynamMetric(dim),
mH(MatrixXd::Zero(mDim, mDim)),
mSoftAbsAlpha(1.0)
{}

double outerSoftAbsMetric::T()
{
    fComputeMetric();
    return 0.5 * mP.dot(fLambdaDot(mP)) + 0.5 * mLogDetMetric;
}

double outerSoftAbsMetric::tau()
{
    fComputeMetric();
    return 0.5 * mP.dot(fLambdaDot(mP));
}

void outerSoftAbsMetric::bounceP(const VectorXd& normal)
{
    
    mAuxVector.noalias() = fLambdaDot(normal);
    double C = -2.0 * mP.dot(mAuxVector);
    C /= normal.dot(mAuxVector);
    
    mP += C * normal;
    
}

/// Sample the momentum from the conditional distribution
/// \f$ \pi \left( \mathbf{p} | \mathbf{q} \right) \propto 
/// \exp \left( - T \left( \mathbf{p}, \mathbf{q} \right) \right) \f$
/// \param random External RandomLib generator
void outerSoftAbsMetric::sampleP(Random& random)
{
    
    // Store the Cholesky matrix in the Hessian
    mH.setZero();
    
    // Compute the rank-one update to the background Cholesky matrix,
    // see http://lapmal.epfl.ch/papers/cholupdate.pdf
    
    gradV();
    const double gg = mGradV.squaredNorm();
    const double agg = mSoftAbsAlpha * gg;
    
    mGradV *= sqrt( (cosh(agg) - 1) / gg );
    
    for(int i = 0; i < mDim; ++i)
    {
        
        double v = mGradV(i);
        double L = 1.0;
        double r = sqrt(L * L + v * v);
        
        double c = L / r;
        double s = v / r;
        
        mH(i, i) = r;
        
        for(int j = i + 1; j < mDim; ++j)
        {
            
            double vprime = mGradV(j);
            double Lprime = mH(i, j);
            
            mGradV(j) = c * vprime - s * Lprime;
            mH(i, j) = s * vprime + c * Lprime;
            
        }
    }
    
    mH.transposeInPlace();
    
    mH *= sqrt( gg / sinh(agg) );
    
    // Sample the momenta
    RandomLib::NormalDistribution<> g;
    for(int i = 0; i < mDim; ++i) mAuxVector(i) = g(random, 0.0, 1.0);
    
    mP.noalias() = mH.triangularView<Eigen::Lower>() * mAuxVector;
    
}

void outerSoftAbsMetric::checkEvolution(const double epsilon)
{
    
    baseHamiltonian::checkEvolution(epsilon);
    
    std::cout.precision(6);
    int width = 12;
    int nColumn = 5;
    
    // Hessian
    fComputeH();
    
    std::cout << "Potential Hessian (d^{2}V/dq_{i}dq_{j}):" << std::endl;
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
    
    // Hamiltonian
    nColumn = 4;
    
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
    
    gradV();
    fComputeH();
    
    VectorXd minusGradH = - dTaudq();
    minusGradH -= dPhidq();
    
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
        
        // Display
        std::cout << "    "
                  << std::setw(width) << std::left << i 
                  << std::setw(width) << std::left << minusGradH(i)
                  << std::setw(width) << std::left << temp
                  << std::setw(width) << std::left << (minusGradH(i) - temp) / (epsilon * epsilon)
                  << std::endl;
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

void outerSoftAbsMetric::fComputeMetric()
{
    
    gradV();
    
    const double gg = mGradV.squaredNorm();
    
    const double s = sinh(mSoftAbsAlpha * gg);
    const double c = cosh(mSoftAbsAlpha * gg);   
    
    mLogDetMetric = -(double)mDim * log(s / gg) + log(c);
    
}

// Compute the product of the inverse metric and the vector v
VectorXd& outerSoftAbsMetric::fLambdaDot(const VectorXd& v)
{

    const double gg = mGradV.squaredNorm();
    const double gv = mGradV.dot(v);
    
    const double s = sinh(mSoftAbsAlpha * gg);
    const double c = cosh(mSoftAbsAlpha * gg);
    
    mAuxVector.noalias() = (s / gg) * (v + (1 / c - 1) * (gv / gg) * mGradV);
    
    return mAuxVector;
    
}

VectorXd& outerSoftAbsMetric::dTaudp()
{
    mAuxVector = fLambdaDot(mP);
    return mAuxVector;
}

VectorXd& outerSoftAbsMetric::dTaudq()
{
    
    const double gg = mGradV.squaredNorm();
    const double agg = mSoftAbsAlpha * gg;
    const double gp = mGradV.dot(mP);
    const double pp = mP.squaredNorm();
    const double gpDgg = gp / gg;

    const double s = sinh(agg);
    const double c = cosh(agg);
    const double t = s / c;  
    
    const double C1 = agg < 1e-4 ?
                      (1.0 / 3.0) * agg * agg :
                      c - s / agg;
    
    const double C2 = agg < 1e-4 ?
                      1.5 * agg * agg :
                      c - 1.0 / (c * c);
    
    const double C3 = agg < 1e-4 ?
                      - 0.5 * agg * agg :
                      (t - s) / agg;
    
    mAuxVector.noalias()  = (2.0 * mSoftAbsAlpha * (C1 * (pp / gg) - (C2 + 2 * C3) * gpDgg * gpDgg) ) * (mH * mGradV);
    mAuxVector.noalias() += (2.0 * mSoftAbsAlpha * C3 * gpDgg) * (mH * mP);
    
    return mAuxVector;
    
}



/// Compute the gradient of the log determinant of the inverse induced metric
VectorXd& outerSoftAbsMetric::dPhidq()
{

    const double gg = mGradV.squaredNorm();
    const double agg = mSoftAbsAlpha * gg;
    const double t = tanh(agg); 
    
    mAuxVector = mH * mGradV;
    
    if(fabs(agg) < 1e-4)
    {
        mAuxVector *= 2 * ( ((double)mDim / (3.0 * gg)) * agg * agg + mSoftAbsAlpha * t);
    }
    else
    {
        mAuxVector *= 2 * ( ((double)mDim / gg) * (1.0 - agg / t) + mSoftAbsAlpha * t);
    }
    
    mAuxVector += mGradV;
    return mAuxVector;
    
}
