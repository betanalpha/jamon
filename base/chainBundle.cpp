// C++ standard libraries
#include "math.h"
#include <iostream>
#include <iomanip>
#include <fstream>

// Local libraries
#include "chainBundle.h"

using std::cout;
using std::endl;
using std::flush;

/// Constructor
/// \param random Pointer to externally instantiated random number generator
/// \see ~chainBundle()

chainBundle::chainBundle(Random& random): mRandom(random)
{
    
    mStoreSamples = true;
    mChar = false;
    mConstraint = true;
    
    mVerbosity = 0;
    
    mNumSubsample = 1;
    mNumLeapfrog = 150;
    mWindowSize = 10;
    
    mStepSize = 0.1;
    mStepSizeJitter = 0.05;
    mStepSizeJerk = 0.1;
    mProbStepSizeJerk = 0.0;
    
    mNumFixedPoint = 7;
    
    mTemperAlpha = 0.0;
    
    mNumSamples = 0;
    
    mMinESS = 0.0;
    
}


/// Destructor
/// \see chainBundle()

chainBundle::~chainBundle()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) delete *mIt;
    mChains.clear();
    
    clearSamples();
}

/// Set reject/accept window size

void chainBundle::setWindowSize(int n)
{
    
    if(n > 0.5 * mNumLeapfrog)
    {
        cout << "chainBundle::setWindowSize() - Reject/Accept window must not be larger"
        << " than half of the total leapfrog steps!" << endl;
        return;
    }
    
    if(n < 2)
    {
        cout << "chainBundle::setWindowSize() - Reject/Accept window must be larger than two"
        << " to avoid proposed windows in violation of the defined constraints!" << endl; 
        return;
    }
    
    mWindowSize = n;
    
}

/// Set valid step size jitter, as a fraction of the stepsize      

void chainBundle::setStepSizeJitter(double j)
{
    
    if(j < 0 || j > 1)
    {
        cout << "chainBundle::setStepSizeJitter() - Step size jitter must lie between 0 and 1!" << endl;
        return;
    }
    
    mStepSizeJitter = j;
    
}

/// Set valid step size jerk, as a fraction of the stepsize    

void chainBundle::setStepSizeJerk(double j)
{
    
    if(j < 0)
    {
        cout << "chainBundle::setStepSizeJerk() - Step size jerk cannot be negative!" << endl;
        return;
    }
    
    mStepSizeJitter = j;
    
}

/// Set valid step size jitter, as a fraction of the stepsize 

void chainBundle::setProbStepSizeJerk(double p)
{
    
    if(p < 0 || p > 1)
    {
        cout << "chainBundle::setProbStepSizeJerk() - Step size jerk probability must lie between 0 and 1!" << endl;
        return;
    }
    
    mProbStepSizeJerk = p;
    
}

/// Set valid number of fixed step iterations for implicit leapfrog updates    

void chainBundle::setNumFixedPoint(int n)
{
    
    if(n < 1)
    {
        cout << "chainBundle::setNumFixedPoint() - There must be at least one fixed point iteration!" << endl;
        return;
    }
    
    mNumFixedPoint = n;
    
}

/// Clear all stored samples

void chainBundle::clearSamples()
{
    mSamples.clear();
    mSampleE.clear();
}

/// Compute summary statistics for the accumulated samples,
/// averaged into batches of size b.  
/// If a pointer to an output file is passed then the file is filled
/// with batch and autocorrelation information ready for parsing
/// with the included gnuplot script, plotTrace.
///
/// Note that the Monte Carlo Standard Error assumes independent batches.
/// /param b Number of samples to be averaged into a single batch
/// /param verbose Flag for verbose diagnostic output
/// /param outputFile Pointer to an external ofstream instance for optional text output

void chainBundle::computeSummaryStats(int b, bool verbose, std::ofstream* outputFile)
{
    
    cout.precision(6);
    int acceptWidth = 14;
    
    if(verbose)
    {
    
        cout << "Displaying summary statistics..." << endl << endl;
        
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "Chain" 
             << std::setw(acceptWidth) << std::left << "Global"
             << std::setw(acceptWidth) << std::left << "Local"
             << std::endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "" 
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::endl;
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        for(int i = 0; i < mChains.size(); ++i)
        {
            cout << "    "
                 << std::setw(acceptWidth / 2) << std::left << i 
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->acceptRate()
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->movingAcceptRate()
                 << endl;
        }
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << endl;
            
    }
    
    int nSamples = (int)mSamples.size();
    int nBatch = nSamples / b;
    int nLag = nBatch / 2;
    
    int nLagDisplay = nLag > 10 ? 10 : nLag;
    nLag = nLag > 25 ? 25 : nLag;
    
    if(verbose)
    {
        cout << "\tUsing " << nBatch << " batches of " << b << " samples"
             << " (" << nSamples % b << " samples clipped)" << endl << endl;
    }
        
    // Loop over samples
    double batchPotential;
    double sumPotential = 0;
    double sumPotential2 = 0;
    
    long dim = chain(0)->dim();
    if(outputFile) *outputFile << "DIM " << dim << endl << endl;
    
    VectorXd batchSample(dim);
    VectorXd sum(VectorXd::Zero(dim));
    VectorXd sum2(VectorXd::Zero(dim));
    
    VectorXd* autoCorr[nLag];
    VectorXd ess(VectorXd::Zero(dim));
    
    vector<VectorXd> batchSamples;
    
    vector<double>::const_iterator eIt = mSampleE.begin();
    vector<VectorXd>::const_iterator it = mSamples.begin();
    
    // Compute batch expectations
    if(outputFile) *outputFile << "BEGINSAMPLES" << endl;
    
    for(int n = 0; n < nBatch; ++n)
    {
        
        batchPotential = 0;
        batchSample.setZero();
        
        // Loop over intra-batch samples
        for(int m = 0; m < b; ++m, ++eIt, ++it)
        {
            
            // Increment sums
            batchPotential += *eIt;
            batchSample += (*it);
            
            // Write to output file, if desired
            if(outputFile)
            {
                *outputFile << n * b + m << "\t" << *eIt << flush;
                for(int i = 0; i < dim; ++i)
                {
                    *outputFile << "\t" << (*it)(i) << flush;
                }
                *outputFile << endl;
            }
        }
        
        // Normalize
        batchPotential /= b;
        sumPotential += batchPotential;
        sumPotential2 += batchPotential * batchPotential;
        
        batchSample /= (double)b;
        batchSamples.push_back(VectorXd(batchSample));
        
        sum += batchSample;
        
        batchSample.cwiseProduct(batchSample);
        sum2 += batchSample;
        
    }
    
    if(outputFile) *outputFile << "ENDSAMPLES" << endl << endl;
    
    // Normalize batch expectations
    sumPotential /= (double)nBatch;
    sumPotential2 = sumPotential2 / (double)nBatch - sumPotential * sumPotential;
    
    sum /= (double)nBatch;
    VectorXd mean2(sum); mean2.cwiseProduct(sum);
    
    sum2 /= (double)nBatch; sum2 -= mean2;
    sum2 *= b / nSamples; // Correct asymptotic batch variance to asymptotic chain variance
    
    // Compute autocorrelations
    VectorXd one(dim);
    VectorXd two(dim);
    
    for(int k = 0; k < nLag; ++k)
    {

        autoCorr[k] = new VectorXd(VectorXd::Zero(dim));
        VectorXd* gamma = autoCorr[k];
        
        vector<VectorXd>::iterator itOne = batchSamples.begin();
        vector<VectorXd>::iterator itTwo = itOne;
        
        for(int j = 0; j < k; ++j) ++itTwo;
        
        for(int n = 0; n < nBatch - k; ++n, ++itOne, ++itTwo)
        {
            one = *itOne - sum;
            two = *itTwo - sum;
            *gamma += one.cwiseProduct(two);
        }
        
        *gamma /= (double)nBatch;

        // Compute autocorrelation
        if(k == 0) continue;
        
        gamma->cwiseQuotient(**autoCorr);
        
    }
    
    (autoCorr[0])->setOnes();
    
    // Compute effective sample size for each variable
    mMinESS = 0;
    
    for(int i = 0; i < dim; ++i)
    {
        
        ess(i) = 0;
        
        double gamma = 0;
        double Gamma = 0;
        double oldGamma = 1;
        
        for(int k = 1; k < nLag; ++k)
        {
            
            gamma = (*(autoCorr[k]))[i];
            
            // Even k, finish computation of Gamma
            if(k % 2 == 0)
            {
                
                Gamma += gamma;
                
                // Convert to initial monotone sequence (IMS)
                if(Gamma > oldGamma) Gamma = oldGamma;
                
                oldGamma = Gamma;
                
                // Terminate if necessary
                if(Gamma < 0) break;
                
                // Increment autocorrelation sum
                ess(i) += Gamma;
                
            }
            // Odd k, begin computation of Gamma
            else
            {
                Gamma = gamma;   
            }
            
            //ess(i) += (1.0 - (double)k / (double)nBatch) * (*(autoCorr[k]))[i];
            
        }
        
        ess(i) = (double)nBatch / (1.0 + 2 * ess(i) );
        
        if(!i) mMinESS = ess(i);
        else   mMinESS = ess(i) < mMinESS ? ess(i) : mMinESS;
        
    }
    
    // Display results
    if(verbose)
    {
        cout << "\tV ~ " << sumPotential << " +/- " << sqrt(sumPotential2) << endl;
        cout << "\tSmallest Effective Sample Size is " << mMinESS << endl << endl;
    }
    
    double whiteNoise = 2.0 / sqrt((double)nBatch);
    
    if(outputFile) 
    {
        
        *outputFile << "WHITENOISE " << whiteNoise << endl << endl;
        *outputFile << "BEGINAUTOCORR" << endl;
        
        for(int k = 0; k < nLag; ++k)
        {
            
            *outputFile << k << flush;
            
            for(int i = 0; i < dim; ++i)
            {
                *outputFile << "\t" << (*(autoCorr[k]))[i] << flush;
            }
            
            *outputFile << endl;
            
        }
        
        *outputFile << "ENDAUTOCORR" << endl;
        
    }
    
    if(verbose)
    {
    
        for(int i = 0; i < dim; ++i)
        {
            
            cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            cout << "\tVariable " << i << endl;
            cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            cout << endl;
            
            cout << "\t\tMean = " << sum[i] << endl;
            cout << "\t\tMonte Carlo Standard Error = " << sqrt(sum2[i]) << endl;
            cout << "\t\tEffective Sample Size = " << ess(i) << endl << endl;
            
            cout << endl << "\t\tAutocorrelation:" << endl;
            cout << "\t\t\tWhite noise correlation = " << whiteNoise << endl << endl;
            
            
            for(int k = 0; k < nLagDisplay; ++k)
            {
                double g = (*(autoCorr[k]))[i];
                cout << "\t\t\tgamma_{" << k << "} = " << g << flush;
                if(g > whiteNoise && k > 0) cout << " (Larger than expected fluctuation from white noise!)" << flush;
                cout << endl;
            }
            
            cout << endl;
            
        }
        
        cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
    
    }
        
    // Clean up
    for(int k = 0; k < nLag; ++k) delete autoCorr[k];
    batchSamples.clear();
    
    return;
    
}

/// Evolve the input Hamiltonian through a time epsilon assuming
/// a splitting of the Hamiltonian of the form
/// \f[ \hat{H} = \frac{\hat{V}}{2} + \frac{\hat{U}}{2} + \hat{T} + \frac{\hat{U}}{2} + \frac{\hat{V}}{2}, \f]
/// where 
/// \f[ \hat{T} = \frac{ \partial H }{\partial p } \frac{ \partial }{ \partial q } \f]
/// and
/// \f[ \hat{V} = - \frac{ \partial H }{\partial q } \frac{ \partial }{ \partial p } \f]
/// are the kinetic and potential Poisson operators, respectively,
/// and \f$\hat{U}\f$ is any constraint potential operator.
/// Note that this implementation assumes that the constraint
/// potential is infinitely large so that \f$ \hat{V} \f$ vanishes
/// when the constraints are violated.
/// \param chain Input Hamiltonian chain
/// \param epsilon Time step

void chainBundle::evolve(baseHamiltonian* chain, double epsilon)
{
    
    // Initial half momentum evolution
    if(mConstraint) chain->beginEvolveP(0.5 * epsilon);
    
    // Full spatial evolution
    chain->evolveQ(epsilon);

    // Check constraint
    mConstraint = fConstraint(chain);
    
    // Final half momentum evolution
    if(mConstraint)  chain->finishEvolveP(0.5 * epsilon);
    else             chain->bounceP(fNormal(chain));
    
}

/// Evolve the input Hamiltonian through a time epsilon, displaying
/// the variation of the Hamiltonian after each update.
/// \see evolve
/// \param chain Input Hamiltonian chain
/// \param epsilon Time step

void chainBundle::verboseEvolve(baseHamiltonian* chain, double epsilon)
{
    
    //cout << "----------------------------------------------------" << endl;
    
    std::cout.precision(6);
    int width = 14;
    int nColumn = 4;

    std::cout << "Verbose Hamiltonian Evolution, Step Size = " << epsilon << ":" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Poisson"
              << std::setw(width) << std::left << "Initial" 
              << std::setw(width) << std::left << "Current"
              << std::setw(width) << std::left << "DeltaH"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Operator"
              << std::setw(width) << std::left << "Hamiltonian" 
              << std::setw(width) << std::left << "Hamiltonian"
              << std::setw(width) << std::left << "/ Stepsize^{2}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    double H0 = chain->H();
    
    // Initial half momentum evolution
    if(mConstraint) chain->beginEvolveP(0.5 * epsilon);
    
    double H1 = chain->H();
    
    std::cout << "    "
              << std::setw(width) << std::left << "hat{V}/2"
              << std::setw(width) << std::left << H0
              << std::setw(width) << std::left << H1
              << std::setw(width) << std::left << (H1 - H0) / (epsilon * epsilon)
              << std::endl;
    
    // Full spatial evolution
    chain->evolveQ(epsilon);
    
    double H2 = chain->H();
    
    std::cout << "    "
              << std::setw(width) << std::left << "hat{T}"
              << std::setw(width) << std::left << H0
              << std::setw(width) << std::left << H2
              << std::setw(width) << std::left << (H2 - H0) / (epsilon * epsilon)
              << std::endl;
    
    // Check constraint
    mConstraint = fConstraint(chain);
    
    // Final half momentum evolution
    if(mConstraint)  chain->finishEvolveP(0.5 * epsilon);
    else             chain->bounceP(fNormal(chain));
    
    double H3 = chain->H();
    
    std::cout << "    "
              << std::setw(width) << std::left << "hat{V}/2"
              << std::setw(width) << std::left << H0
              << std::setw(width) << std::left << H3
              << std::setw(width) << std::left << (H3 - H0) / (epsilon * epsilon)
              << std::endl;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
}

/// Transition all chains currently in the bundle

void chainBundle::transition()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) transition(*mIt);
}

/// Transition the ith chain in the bundle
/// \param i Index of the chain

void chainBundle::transition(int i)
{
    
    if(i < 0 || i >= mChains.size())
    {
        cout << "chainBundle::transition() - Bad chain index " << i << "!" << endl;
        return;
    }
    
    transition(mChains.at(i));
    
    return;
    
}

/// Generate nNumSubsample consecutive transitions from the input 
/// chain using constrained Hamiltonian Monte Carlo, storing 
/// the final state as a sample if desired.
///
/// \param chain Input Markov chain
/// \return Was the final proposal accepted?

bool chainBundle::transition(baseHamiltonian* chain)
{
    
    // Check that the chain was properly seeded
    if(mNumSamples == 0)
    {
        
        mConstraint = fConstraint(chain);
        
        if(!mConstraint)
        {
            cout << "chainBundle::transition() - Initial chain parameters in violation "
            << "of constraints, aborting the transition!" << endl;
            return false;
        }
        
    }
    
    long dim = chain->dim();
    bool accept = true;
    bool nanFlag = false;
    
    for(int n = 0; n < mNumSubsample; ++n)
    {

        // Add random jitter to the step size to avoid closed loops, and
        // an occassional jerk to allow the chain to move through narrow valleys
        double stepSize = mStepSize;
        
        if(mRandom.Prob<double>(mProbStepSizeJerk)) 
        {
            stepSize *= mStepSizeJerk;
        }
        else
        {
            stepSize *= 1.0 + mStepSizeJitter * (2.0 * mRandom.FloatU() - 1.0);
        }

        // Prepare the chain
        chain->prepareEvolution();
        
        // Sample momenta conditioned on the current position of the chain
        chain->sampleP(mRandom);

        //////////////////////////////////////////////////
        //        Sample from the reject window         //
        //////////////////////////////////////////////////
        
        // Immediately accept initial point as current sample
        chain->saveAsRejectSample();
        double logSumRejectProb = -chain->H();
        
        // Backwards evolution
        stepSize *= -1;
        
        chain->saveCurrentPoint();
        int s = mRandom.Integer(mWindowSize);

        for(int t = 0; t < s; ++t)
        {
            
            evolve(chain, stepSize);
            
            // Accept or reject updating the sample, with constraint violations immediately rejected 
            if(mConstraint) 
            {
                double Hplus = chain->H();
                logSumRejectProb = fLogSumExp(logSumRejectProb, -Hplus);
                if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsRejectSample();
            }
            
        }
        
        // Forwards evolution
        stepSize *= -1;
        
        chain->restoreStoredPoint();

        for(int t = 0; t < mWindowSize - s - 1; ++t)
        {
            
            evolve(chain, stepSize);
            
            // Accept or reject updating the sample, with constraint violations immediately rejected 
            if(mConstraint) 
            {
                double Hplus = chain->H();
                logSumRejectProb = fLogSumExp(logSumRejectProb, -Hplus);
                if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsRejectSample();
            }
            
        }
        
        //////////////////////////////////////////////////
        //       Evolve for mNumLeapfrog iterations     //
        //////////////////////////////////////////////////

        if(mVerbosity >= 2)
        {
            cout << "Beginning leapfrog:" << endl;
            cout << endl;
            chain->displayState();
            cout << endl;
        }
        
        if(mTemperAlpha) chain->p() *= sqrt(mTemperAlpha);
        
                
        double m = 0.5 * (mNumLeapfrog - 2 * mWindowSize);
        
        for(unsigned int t = 0; t < mNumLeapfrog - 2 * mWindowSize; ++t)
        {
            
            // Check for any NANs and INFs that have sneaked in, restarting if any are found
            nanFlag = false;
            for(int i = 0; i < dim; ++i) if( chain->p()(i) != chain->p()(i) ) nanFlag |= true;
            for(int i = 0; i < dim; ++i) if(std::isinf(chain->q()(i))) nanFlag |= true;
            if(nanFlag) break;
            
            evolve(chain, stepSize);
            
            if(mTemperAlpha)
            {
                if(t < m)      chain->p() *= sqrt(mTemperAlpha);
                else if(t >= m) chain->p() /= sqrt(mTemperAlpha);
            }
                
        }
        
        // Restart if loop ended with a NAN
        if(nanFlag)
        {
            
            if(mVerbosity >= 1)
            {
                cout << "Restarting leapfrog after encountering a NAN..." << endl;
                cout << endl;
                chain->displayState();
                cout << endl;
            }
            
            chain->restoreStoredPoint();
            n--;
            continue;
        }
        
        if(mVerbosity >= 2)
        {
            cout << "Ending leapfrog:" << endl;
            cout << endl;
            chain->displayState();
            cout << endl;
        }
        
        //////////////////////////////////////////////////
        //        Sample from the accept window         //
        //////////////////////////////////////////////////
        
        // Immediately accept current point, provided the constraint is not violated
        double logSumAcceptProb = 0;
        bool goodAcceptSample = false;
        
        if(mConstraint) 
        {
            chain->saveAsAcceptSample();
            logSumAcceptProb = -chain->H();
            goodAcceptSample = true;
        }
        
        for(int t = 0; t < mWindowSize - 1; ++t)
        {
            
            evolve(chain, stepSize);
            
            // Accept or reject updating the sample, with constraint violations immediately rejected 
            if(mConstraint) 
            {
                
                double Hplus = chain->H();
                
                // Immediately accept if no sample has yet been accepted
                if(!goodAcceptSample)
                {
                    logSumAcceptProb = -Hplus;
                    chain->saveAsAcceptSample();
                    goodAcceptSample = true;
                }
                else
                {
                    logSumAcceptProb = fLogSumExp(logSumAcceptProb, -Hplus);
                    if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsAcceptSample();
                }
                
            }
            
        }
        
        //////////////////////////////////////////////////
        //   Sample between reject and accept windows   //
        //////////////////////////////////////////////////
        
        double metropolisProb = exp(logSumAcceptProb - logSumRejectProb);
        if(!goodAcceptSample) metropolisProb = 0;
        
        if(metropolisProb > 1) accept = true;
        else accept = mRandom.Prob<double>(metropolisProb);

        chain->sampleWindows(accept);

        if(mVerbosity >= 2)
        {
            if(accept) cout << "Accepting proposed sample with" << endl;
            else       cout << "Rejecting proposed sample with" << endl;
            
            cout << "    log(p_{accept window}) = " << logSumAcceptProb << endl;
            cout << "    log(p_{reject window}) = " << logSumRejectProb << endl;
            cout << "    p(accept window) = " << metropolisProb << endl << endl;
            
        }
        
        chain->updateMetroStats((double)accept);
        
    }
    
    // Store sample if desired
    if(mStoreSamples) 
    {
        mSamples.push_back(VectorXd(chain->q()));
        mSampleE.push_back(chain->V());
    }
    
    ++mNumSamples;
    
    return accept;
    
}

/// Check the accumulated error in the symplectic integrator
/// implemented in the ith chain by evolving the chain
/// n time steps, displaying the error every f steps.
/// \param i Index of the chain
/// \param N Number of time steps
/// \param f Frequncy of output

void chainBundle::checkIntError(int i, int N, int f)
{
    
    if(i < 0 || i >= mChains.size())
    {
        cout << "chainBundle::checkIntError() - Bad chain index " << i << "!" << endl;
        return;
    }
    
    // Prepare chain
    mChains.at(i)->prepareEvolution();
    mChains.at(i)->sampleP(mRandom);
 
    mChains.at(i)->displayState();
    
    double initH = mChains.at(i)->H();
    
    // Display Header
    std::cout.precision(6);
    int width = 12;
    int nColumn = 4;
    
    std::cout << "Displaying the integration error of chain " << i << ":" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Step" 
              << std::setw(width) << std::left << "H_{0}"
              << std::setw(width) << std::left << "H_{i}"
              << std::setw(width) << std::left << "H_{0} - H_{i}"
              << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    // Evolve and display results
    for(int n = 0; n < N; ++n)
    {
        
        evolve(mChains.at(i), mStepSize);
        
        if(!(n % f))
        {
            std::cout << "    "
                      << std::setw(width) << std::left << n
                      << std::setw(width) << std::left << initH
                      << std::setw(width) << std::left << mChains.at(i)->H()
                      << std::setw(width) << std::left << initH -  mChains.at(i)->H()
                      << std::endl;
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Seed all chains, using the same bounds for all components
/// \param min Minimum bound for all components
/// \param max Maximum bound for all components

void chainBundle::seed(double min, double max)
{
    for(int i = 0; i < mChains.at(0)->dim(); ++i) seed(i, min, max);
}

/// Seed the ith component of the feature space for all chains
/// \param i The selected component of the feature space
/// \param min Minimum bound for the ith component
/// \param max Maximum bound for the ith component

void chainBundle::seed(int i, double min, double max)
{
    
    if(i < 0 || i >= mChains.at(0)->dim())
    {
        cout << "chainBundle::seed() - Bad parameter index!" << endl;
        return;
    }
    
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) 
    {
        (*mIt)->q()(i) = (max - min) * mRandom.FloatU<double>() + min;
    }
    
}

/// Burn in all chains until the Gelman-Rubin convergence critieria has been satisfied
///
/// Gelman, A.
/// Inference and monitoring convergence,
/// "Markov Chain Monte Carlo in Practice"
/// (1996) Chapman & Hall/CRC, New York
///
/// \param nBurn Number of burn in iterations
/// \param nCheck Number of samples for diagnosing burn in
/// \param minR Minimum Gelman-Rubin statistic
/// \param verbose Set verbose output, defaults to true

void chainBundle::burn(int nBurn, int nCheck, double minR, bool verbose)
{
    
    // Convergence storage
    long dim = mChains.at(0)->dim();
    double m = (double)mChains.size();
    double n = (double)nCheck;
                    
    VectorXd R(dim);   // Gelman-Rubin statistics
    
    VectorXd* chainSum[(int)m];
    for(int i = 0; i < m; ++i) chainSum[i] = new VectorXd(dim);
    VectorXd ensembleSum(dim);
    VectorXd ensembleSum2(dim);
    VectorXd chainSum2(dim);
    
    bool burning = true;
    
    int nPerLine = 5;
    int nLines = (int)dim / nPerLine;
    
    cout.precision(6);
    int width = 12;
    int totalWidth = width + 8;
    int borderWidth = ((int)dim < nPerLine ? (int)dim : nPerLine) * totalWidth + 4;
    
    int acceptWidth = 14;
    
    //////////////////////////////////////////////////
    //              Burn in Markov Chain            //
    //////////////////////////////////////////////////
    
    // Temporarily store samples
    bool sampleFlag = mStoreSamples;
    mStoreSamples = false;
    
    // Burn through initial samples
    if(verbose) cout << endl << "Burn, baby, burn" << flush;
    
    for(int i = 0; i < nBurn; ++i) 
    {
        transition();
        if(verbose) if(i % 10 == 0) cout << "." << flush;
    }
    
    if(verbose) cout << endl << endl;
    if(verbose) cout << "Computing diagnostics:" << endl;
    if(verbose) cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
    
    while(burning)
    {
        
        ensembleSum.setZero();
        ensembleSum2.setZero();
        for(int i = 0; i < m; ++i) (chainSum[i])->setZero();
        chainSum2.setZero();

        // Burn through diagnostic samples
        for(int i = 0; i < nCheck; ++i) 
        {
            
            transition();

            mIt = mChains.begin();
            for(int j = 0; j < m; ++j, ++mIt)
            {
            
                VectorXd& q = (*mIt)->q();
                
                ensembleSum += q;
                ensembleSum2 += q.cwiseAbs2();
                
                *(chainSum[j]) += q;
 
            }
            
        }

        // Compute Gelman-Rubin statistic for each parameter
        double a = (n - 1) / n;
        double b = m / (m - 1);
        
        for(int j = 0; j < m; ++j)
        {
            chainSum2 += (chainSum[j])->cwiseAbs2();
        }
        
        ensembleSum = ensembleSum.cwiseAbs2();
        ensembleSum = ensembleSum.cwiseQuotient(chainSum2);
        ensembleSum /= m;
        ensembleSum = VectorXd::Ones(dim) - ensembleSum;

        ensembleSum2 = ensembleSum2.cwiseQuotient(chainSum2);
        ensembleSum2 *= n;
        ensembleSum2 = VectorXd::Ones(dim) - ensembleSum2;
        
        R = a * (VectorXd::Ones(dim) - b * ensembleSum.cwiseQuotient(ensembleSum2));
        
        burning = false;
        for(int i = 0; i < dim; ++i) burning |= (R[i] > minR);
        
        // Failure output
        if(verbose && burning)
        {
            cout << "    Diagnostic test failed," << endl;
            
            int k = 0;
            for(int i = 0; i < nLines; ++i)
            {
                
                cout << "        " << flush;
                for(int j = 0; j < nPerLine; ++j, ++k)
                {
                    cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
                }
                
                cout << endl;
                
            }
            
            if(dim % nPerLine > 0)
            {
                
                cout << "        " << flush;
                for(int i = 0; i < dim % nPerLine; ++i, ++k)
                {
                    cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
                }
                cout << endl;
                
            }
            
            cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            
        }
        
    }
    
    // Display convergence details if desired
    if(verbose)
    {
        
        cout << "    Markov chains converged with" << endl;
        
        int k = 0;
        for(int i = 0; i < nLines; ++i)
        {
            
            cout << "        " << flush;
            for(int j = 0; j < nPerLine; ++j, ++k)
            {
                cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
            }
            
            cout << endl;
            
        }
        
        if(dim % nPerLine > 0)
        {
            
            cout << "        " << flush;
            for(int i = 0; i < dim % nPerLine; ++i, ++k)
            {
                cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
            }
            cout << endl;
            
        }
        
        cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        cout << endl;
        
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "Chain" 
             << std::setw(acceptWidth) << std::left << "Global"
             << std::setw(acceptWidth) << std::left << "Local"
             << std::endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "" 
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::endl;
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        for(int i = 0; i < mChains.size(); ++i)
        {
            cout << "    "
                 << std::setw(acceptWidth / 2) << std::left << i 
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->acceptRate()
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->movingAcceptRate()
                 << endl;
        }
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << endl;
         
    }
    
    mChar = true;
    
    // Return sample flag
    mStoreSamples = sampleFlag;
    
    return;
    
}

/// Compute the logarithm of the sum of two exponentials,
/// \f$ \log( \exp(a) + \exp(b) ) \f$
/// param a Argument of one exponential
/// param b Argument of the second exponential
/// return logarithm of the summed exponentials

double chainBundle::fLogSumExp(double a, double b)
{
    
    if(a > b) 
    {
        return a + log( 1.0 + exp(b - a) );
    }
    else
    {      
        return b + log( 1.0 + exp(a - b) );
    }
    
}

