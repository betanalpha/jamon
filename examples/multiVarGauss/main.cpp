#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>

#include "chainBundle.h"
#include "bsMultiVarGauss.h"

using namespace Eigen;
using namespace std;

int main (int argc, const char * argv[])
{

    //////////////////////////////////////////////////
    //       Prepare data and model parameters      //
    //////////////////////////////////////////////////
    
    int N = 10;
    
    RandomLib::Random random;
    random.Reseed();
    
    // Mean
    VectorXd mu = VectorXd::Zero(N);
    
    for(int n = 0; n < N; ++n)
    {
        if(random.Prob<double>(0.5)) mu(n) = +10.0;
        else                         mu(n) = -10.0;
    }
    
    // Covariance
    MatrixXd sigma = MatrixXd::Zero(N, N);
    
    double condNumber = 1000;
    
    for(int n = 0; n < N; ++n)
    {
        sigma(n, n) = (condNumber - 1) * (double)n / (double)(N - 1) + 1;
    }
    
    for(int n = 1; n < N; ++n)
    {
        
        double a = sigma(0, 0);
        double b = sigma(0, n);
        double c = sigma(n, n);
        
        sigma(0, 0) = 0.5 * (a - 2 * b + c);
        sigma(0, n) = 0.5 * (a - c);
        sigma(n, 0) = 0.5 * (a - c);
        sigma(n, n) = 0.5 * (a + 2 * b + c);
        
    }
    
    LLT<MatrixXd> L;
    L.compute(sigma.selfadjointView<Eigen::Lower>());
    
    MatrixXd lambda = L.solve(MatrixXd::Identity(N, N));
    
    //////////////////////////////////////////////////
    //            Build Chains and Sample           //
    //////////////////////////////////////////////////
    
    chainBundle bundle(random);
    bundle.setProbStepSizeJerk(0);
    
    bundle.addChain(new bsMultiVarGauss(mu, lambda));
    
    bundle.seed(-10, 10);
    bundle.setStepSize(5e-1);
    bundle.setNumLeapfrog(250);
    bundle.setWindowSize(25);
    
    // Check implemenation
    bundle.chain(0)->prepareEvolution();
    bundle.chain(0)->sampleP(random);
    //bundle.chain(0)->checkEvolution();
    
    bundle.chain(0)->prepareEvolution();
    //bundle.verboseEvolve(bundle.chain(0), 1e-1);
    
    //bundle.checkIntError(0, 50);
    
    // Warm-Up
    cout << "Warm Up:" << endl;
    for(int i = 0; i < 500; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        bundle.transition(0);
    }
    cout << endl;

    std::ofstream warmOut; warmOut.open("/Users/betan/Desktop/bsMultiVarGaussWarm.txt");

    bundle.computeSummaryStats(1, true, &warmOut);

    warmOut.close();
    
    // Posterior sampling
    bundle.clearSamples();
    
    cout << "Posterior Sampling:" << endl;
    for(int i = 0; i < 5000; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        bundle.transition(0);
    }
    cout << endl;
    
    std::ofstream out; out.open("/Users/betan/Desktop/bsMultiVarGauss.txt");
    
    bundle.computeSummaryStats(1, true, &out);
    
    out.close();
    
    return 0;
    
}

