#include <iostream>
#include <fstream>
#include <time.h>

#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>

#include "flatFunnel.h"
#include "softAbsFunnel.h"
#include "fisherBayesLogReg.h"
#include "softAbsBayesLogReg.h"

#include "chainBundle.hpp"

using namespace std;

int main (int argc, const char * argv[])
{
    
    /************************************************/
    /*                                              */
    /*                 Preliminaries                */
    /*                                              */
    /************************************************/
    
    clock_t start;
    clock_t end;
    double deltaT;
    
    RandomLib::Random random;
    random.Reseed();
    
    std::ofstream trajectory;
    std::ofstream stats;
    
    /************************************************/
    /*                                              */
    /*                    Funnel                    */
    /*                                              */
    /************************************************/
    
    int funnelDim = 50;
    
    VectorXd funnelInitChain(funnelDim + 1);
    for(int i = 0; i < funnelDim + 1; ++i) funnelInitChain(i) = 2 * random.FloatU<double>() - 1;
    
    VectorXd funnelInitTrajectory(funnelDim + 1);
    funnelInitTrajectory(0) = 3.5;
    for(int i = 1; i < funnelDim + 1; ++i) funnelInitTrajectory(i) = 10 * random.FloatU<double>() - 5;
    
    /************************************************/
    /*                  Euclidean HMC               */
    /************************************************/
    
    cout << "Beginning Flat Funnel..." << endl;
    
    chainBundle<flatFunnel> flatFunnelBundle(random);
    flatFunnelBundle.setWindowSize(0);
    flatFunnelBundle.setMaxLagCalc(1000);
    flatFunnelBundle.setMaxLagDisplay(1000);
    
    flatFunnelBundle.addChain(new flatFunnel(funnelDim));
    
    // Generate and then save a Hamiltonian trajectory
    // Display the trajectory with `./plotTrajectory flatFunnelTrajectory.txt` 
    flatFunnelBundle.setStepSize(1e-2);
    flatFunnelBundle.setNumLeapfrog(5000);
    
    flatFunnelBundle.chain(0)->q() = funnelInitTrajectory;
    flatFunnelBundle.chain(0)->sampleP(random);
    flatFunnelBundle.chain(0)->prepareEvolution();
    
    trajectory.open("flatFunnelTrajectory.txt");
    flatFunnelBundle.saveTrajectory(trajectory);
    trajectory.close();
    
    // Generate samples from the distribution
    // Note that no step-size adaption is used
    // Trace plots can be genereated with `./plotTrace flatFunnelStats.txt` 
    flatFunnelBundle.clearSamples();
    flatFunnelBundle.setStepSize(1e-2);
    
    flatFunnelBundle.chain(0)->q() = funnelInitChain;
    flatFunnelBundle.chain(0)->sampleP(random);
    flatFunnelBundle.chain(0)->prepareEvolution();
    
    flatFunnelBundle.setAdaptIntTime(15.0);
    flatFunnelBundle.initAdaptation();
    
    for(int i = 0; i < 1000; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        flatFunnelBundle.transition(0);
    }
    cout << endl;
    
    cout << "Accept probability after burn = " << flatFunnelBundle.chain(0)->acceptRate() << endl;
    
    flatFunnelBundle.clearSamples();
    
    start = clock();
    
    for(int i = 0; i < 10000; ++i) 
    {
        if(i % 100 == 0) cout << "." << flush;
        flatFunnelBundle.transition(0);
    }
    cout << endl;
    
    end = clock();
    
    deltaT = (double)(end - start) / CLOCKS_PER_SEC;
    
    cout << "Accept probability after sampling = " << flatFunnelBundle.chain(0)->acceptRate() << endl;
    
    stats.open("flatFunnelStats.txt");
    flatFunnelBundle.computeSummaryStats(1, false, &stats);
    stats.close();
    
    cout << "Flat Funnel (No adaptation):" << endl;
    cout << "\tDeltaT = " << deltaT << " seconds" << endl;
    cout << "\tminESS = " << flatFunnelBundle.minESS() << endl;
    cout << "\tminESS / DeltaT = " << flatFunnelBundle.minESS() / deltaT << " s^{-1}" << endl;
    cout << endl;
    
    /************************************************/
    /*                 Riemannian HMC               */
    /*            with the SoftAbs Metric           */
    /************************************************/
    
    cout << "Beginning SoftAbs Funnel..." << endl;
    
    chainBundle<softAbsFunnel> softAbsFunnelBundle(random);
    softAbsFunnelBundle.setWindowSize(0);
    softAbsFunnelBundle.setMaxLagCalc(50);
    softAbsFunnelBundle.setMaxLagDisplay(50);
    
    softAbsFunnelBundle.addChain(new softAbsFunnel(funnelDim));
    
    // Generate and then save a Hamiltonian trajectory
    // Display the trajectory with `./plotTrajectory softAbsFunnelTrajectory.txt` 
    softAbsFunnelBundle.setStepSize(1e-2);
    softAbsFunnelBundle.setNumLeapfrog(5000);
    
    softAbsFunnel* chain = (softAbsFunnel*)softAbsFunnelBundle.chain(0);
    chain->setSoftAbsAlpha(1e6);
    
    chain->q() = funnelInitTrajectory;
    chain->sampleP(random);
    chain->prepareEvolution();
    
    trajectory.open("softAbsFunnelTrajectory.txt");
    softAbsFunnelBundle.saveTrajectory(trajectory);
    trajectory.close();
    
    // Generate samples from the distribution
    // Note that step-size adaption is used
    // Trace plots can be genereated with `./plotTrace softAbsFunnelStats.txt` 
    softAbsFunnelBundle.clearSamples();
    softAbsFunnelBundle.setStepSize(1e-1);
    
    softAbsFunnelBundle.chain(0)->q() = funnelInitChain;
    softAbsFunnelBundle.chain(0)->sampleP(random);
    softAbsFunnelBundle.chain(0)->prepareEvolution();
    
    softAbsFunnelBundle.setAdaptIntTime(15.0);
    softAbsFunnelBundle.setAdaptTargetAccept(0.95);
    softAbsFunnelBundle.initAdaptation();
    softAbsFunnelBundle.engageAdaptation();
    
    for(int i = 0; i < 100; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        softAbsFunnelBundle.transition(0);
    }
    cout << endl;
    
    cout << "Accept probability after burn = " << softAbsFunnelBundle.chain(0)->acceptRate() << endl;
    cout << "Stepsize = " << softAbsFunnelBundle.stepSize() << ", nLeapfrog = " << softAbsFunnelBundle.nLeapfrog() << endl;
    
    softAbsFunnelBundle.disengageAdaptation();
    softAbsFunnelBundle.clearSamples();
    
    start = clock();
    
    for(int i = 0; i < 1000; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        softAbsFunnelBundle.transition(0);
    }
    cout << endl;
    
    end = clock();
    
    deltaT = (double)(end - start) / CLOCKS_PER_SEC;
    
    cout << "Accept probability after sampling = " << softAbsFunnelBundle.chain(0)->acceptRate() << endl;
    
    stats.open("softAbsFunnelStats.txt");
    softAbsFunnelBundle.computeSummaryStats(1, false, &stats);
    stats.close();
    
    cout << "SoftAbs Funnel:" << endl;
    cout << "\tDeltaT = " << deltaT << " seconds" << endl;
    cout << "\tminESS = " << softAbsFunnelBundle.minESS() << endl;
    cout << "\tminESS / DeltaT = " << softAbsFunnelBundle.minESS() / deltaT << " s^{-1}" << endl;
    cout << endl;
    
    /************************************************/
    /*                                              */
    /*             Logistic Regression              */
    /*                                              */
    /************************************************/
    
    int nData = 768;
    int nCovar = 8;
    
    MatrixXd data = MatrixXd::Zero(nData, nCovar + 1);
    VectorXd t = VectorXd::Zero(nData);
    
    double alpha = 100.0;
    
    // Input data (http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
    // Note that the data has not been standarized
    
    ifstream inputData;
    inputData.open("pima-indians-diabetes.space.dat");
    
    double buffer = 0.0;
    
    for(int n = 0; n < nData; ++n)
    {
        
        // Covariates
        for(int i = 0; i < nCovar; ++i)
        {
            inputData >> buffer;
            data(n, i) = buffer;
        }
        
        // Intercept
        data(n, nCovar) = 1;
        
        // Predictor
        inputData >> buffer;
        t(n) = buffer;
        
    }
    
    inputData.close();
    
    // Sampling preliminaries
    VectorXd initQ(nCovar + 1);
    initQ(0) = 0.101596;
    initQ(1) = 0.0136317;
    initQ(2) = -0.0194615;
    initQ(3) = 0.00527742;
    initQ(4) = 0.00143356;
    initQ(5) = -0.0304068;
    initQ(6) = -0.496651;
    initQ(7) = -0.0128637;
    initQ(8) = 0.0574961;
    
    int nBurn = 100;
    int nSample = 1000;
    
    int nLagCalc = 100;
    int nLagDisplay = 100;
    
    /************************************************/
    /*                 Riemannian HMC               */
    /*           with the Fisher-Rao Metric         */
    /************************************************/
    
    cout << "Beginning Fisher-Rao Logistic Regression..." << endl;
    
    chainBundle<fisherBayesLogReg> fisherLogRegBundle(random);
    fisherLogRegBundle.setWindowSize(0);
    fisherLogRegBundle.setMaxLagCalc(nLagCalc);
    fisherLogRegBundle.setMaxLagDisplay(nLagDisplay);
    
    fisherLogRegBundle.addChain(new fisherBayesLogReg(data, t, alpha));
    
    fisherLogRegBundle.setStepSize(1e-2);
    
    fisherLogRegBundle.chain(0)->q() = initQ;
    fisherLogRegBundle.chain(0)->sampleP(random);
    fisherLogRegBundle.chain(0)->prepareEvolution();
    
    fisherLogRegBundle.setAdaptIntTime(1.5);
    fisherLogRegBundle.setAdaptTargetAccept(0.35);
    fisherLogRegBundle.setAdaptMaxLeapfrog(100);
    fisherLogRegBundle.initAdaptation();
    fisherLogRegBundle.engageAdaptation();
    
    for(int i = 0; i < nBurn; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        fisherLogRegBundle.transition(0);
    }
    cout << endl;
    
    cout << "Accept probability after burn = " << fisherLogRegBundle.chain(0)->acceptRate() << endl;
    cout << "Stepsize = " << fisherLogRegBundle.stepSize() << ", nLeapfrog = " << fisherLogRegBundle.nLeapfrog() << endl;
    
    fisherLogRegBundle.disengageAdaptation();
    fisherLogRegBundle.clearSamples();
    
    start = clock();
    
    for(int i = 0; i < nSample; ++i) 
    {
        if(i % 100 == 0) cout << "." << flush;
        fisherLogRegBundle.transition(0);
    }
    cout << endl;
    
    end = clock();
    
    deltaT = (double)(end - start) / CLOCKS_PER_SEC;
    
    cout << "Accept probability after sampling = " << fisherLogRegBundle.chain(0)->acceptRate() << endl;
    
    stats.open("fisherLogRegStats.txt");
    fisherLogRegBundle.computeSummaryStats(1, false, &stats);
    stats.close();
    
    cout << "Fisher-Rao Logistic Regression:" << endl;
    cout << "\tDeltaT = " << deltaT << " seconds" << endl;
    cout << "\tminESS = " << fisherLogRegBundle.minESS() << endl;
    cout << "\tminESS / DeltaT = " << fisherLogRegBundle.minESS() / deltaT << " s^{-1}" << endl;
    cout << endl;
    
    /************************************************/
    /*                 Riemannian HMC               */
    /*            with the SoftAbs Metric           */
    /************************************************/
    
    cout << "Beginning SoftAbs Logistic Regression..." << endl;
    
    chainBundle<softAbsBayesLogReg> softAbsLogRegBundle(random);
    softAbsLogRegBundle.setWindowSize(0);
    softAbsLogRegBundle.setMaxLagCalc(nLagCalc);
    softAbsLogRegBundle.setMaxLagDisplay(nLagDisplay);
    
    softAbsLogRegBundle.addChain(new softAbsBayesLogReg(data, t, alpha));
    
    softAbsBayesLogReg* softAbsChain = (softAbsBayesLogReg*)softAbsLogRegBundle.chain(0);
    softAbsChain->setSoftAbsAlpha(1e6);
    
    softAbsLogRegBundle.setStepSize(1e-2);
    
    softAbsLogRegBundle.chain(0)->q() = initQ;
    softAbsLogRegBundle.chain(0)->sampleP(random);
    softAbsLogRegBundle.chain(0)->prepareEvolution();
    
    softAbsLogRegBundle.setAdaptIntTime(1.5);
    softAbsLogRegBundle.setAdaptTargetAccept(0.95);
    softAbsLogRegBundle.initAdaptation();
    softAbsLogRegBundle.engageAdaptation();
    
    for(int i = 0; i < nBurn; ++i) 
    {
        if(i % 10 == 0) cout << "." << flush;
        softAbsLogRegBundle.transition(0);
    }
    cout << endl;
    
    cout << "Accept probability after burn = " << softAbsLogRegBundle.chain(0)->acceptRate() << endl;
    cout << "Stepsize = " << softAbsLogRegBundle.stepSize() << ", nLeapfrog = " << softAbsLogRegBundle.nLeapfrog() << endl;
    
    softAbsLogRegBundle.disengageAdaptation();
    softAbsLogRegBundle.clearSamples();
    
    start = clock();
    
    for(int i = 0; i < nSample; ++i) 
    {
        if(i % 100 == 0) cout << "." << flush;
        softAbsLogRegBundle.transition(0);
    }
    cout << endl;
    
    end = clock();
    
    deltaT = (double)(end - start) / CLOCKS_PER_SEC;
    
    cout << "Accept probability after sampling = " << softAbsLogRegBundle.chain(0)->acceptRate() << endl;
    
    stats.open("softAbsLogRegStats.txt");
    softAbsLogRegBundle.computeSummaryStats(1, false, &stats);
    stats.close();
    
    cout << "softAbs Logistic Regression:" << endl;
    cout << "\tDeltaT = " << deltaT << " seconds" << endl;
    cout << "\tminESS = " << softAbsLogRegBundle.minESS() << endl;
    cout << "\tminESS / DeltaT = " << softAbsLogRegBundle.minESS() / deltaT << " s^{-1}" << endl;
    cout << endl;
    
    return 0;
    
}