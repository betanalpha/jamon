#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>

#include "chainBundle.h"
#include "bsFunnel.h"

using namespace Eigen;
using namespace std;

int main (int argc, const char * argv[])
{
    
    //////////////////////////////////////////////////
    //            Build Chains and Sample           //
    //////////////////////////////////////////////////
    
    RandomLib::Random random;
    random.Reseed();
    
    int n = 10;
    
    chainBundle bundle(random);
    bundle.setProbStepSizeJerk(0);
    
    bundle.addChain(new bsFunnel(n));
    
    bundle.seed(-3, 3);
    bundle.setStepSize(1e-2);
    bundle.setNumLeapfrog(200);
    bundle.setWindowSize(25);
    bundle.setNumSubsample(10);
    
    // Check implemenation
    bundle.chain(0)->prepareEvolution();
    bundle.chain(0)->sampleP(random);
    //bundle.chain(0)->checkEvolution();
    
    bundle.chain(0)->prepareEvolution();
    //bundle.verboseEvolve(bundle.chain(0), 1e-2);
    
    //bundle.checkIntError(0, 50);

    // Warm-Up
    cout << "Warm Up:" << endl;
    for(int i = 0; i < 100; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        bundle.transition(0);
    }
    cout << endl;
    
    std::ofstream warmOut; warmOut.open("/Users/betan/Desktop/bsFunnelWarm.txt");
    
    bundle.computeSummaryStats(1, true, &warmOut);
    
    warmOut.close();
    
    // Posterior sampling
    bundle.clearSamples();
    
    cout << "Posterior Sampling:" << endl;
    for(int i = 0; i < 500; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        bundle.transition(0);
    }
    cout << endl;
    
    std::ofstream out; out.open("/Users/betan/Desktop/bsFunnel.txt");
    
    bundle.computeSummaryStats(1, true, &out);
    
    out.close();
    
    return 0;
}

