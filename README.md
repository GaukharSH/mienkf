# MIEnKF.jl #

This package provides a Julia language implementation of the multi-index ensemble Kalman filtering (MIEnKF) method with the comparison to the ensemble Kalman filtering (EnKF) and multlevel EnKF methods. 

The code was tested in Julia 1.2 and Julia 1.5.2.

> Requirements: Julia 1.2

If you have already installed Julia 1.2, then skip to the **Step 2** under *Installation*. 


## Installation

1. Install [Julia 1.2](https://julialang.org/downloads/) following the installation steps on its website. 
2. Download and save [mienkf.jl](https://github.com/GaukharSH/mienkf.jl) in your desired location.
3. Open a terminal and `cd` to the location of `mienkf.jl`.
4. In the command line type 

    **julia>** `include("mienkf.jl")`
    
    It might need certain packages to install, then you have to run
    
    **julia>** `Pkg.add("TheRequestedPkgName")`


## Usage

1. The code `mienkf.jl` is written for the test problems with Ornstein-Uhlenbeck, Double-well SDEs and Langevin dynamics. In the code script, you can choose one of these problems, set the parameters (See "### DOUBLE WELL PROBLEM" and "### OU PROBLEM" in the code for where to comment/uncomment for the respective problems) or you can test other problems with the  constant diffusion stochastic dynamics by altering the potential function in the drift term. If you have made some changes in `mienkf.jl` file, repeat the **Step 4** under *Installation*.
1*. The code `mienkf_Langevin.jl` is written for the test problem with Langevin dynamics.  In the code script, you can choose the error tolerance for the pseudo-reference solution computed by MIEnKF method. If you have made some changes in `mienkf_Langevin.jl` file, repeat the **Step 4** under *Installation*.
2. Now run the code which computes the reference solution, runtime (wall-clock time) and time-averaged root mean-squared error (RMSE) of EnKF, MLEnKF and MIEnKF (for detailed information see Section 3 of the manuscript).

    **julia>** `testEstimatorRates()`
    
It should produce terminal text output similar to the following:
    
 ```
     Done computing refSol
Enkf [epsilon time error ] : [0.0625 0.792183051 0.015365678292489398]
Enkf [epsilon time error ] : [0.0625 0.046173259 0.015365678292489398]
Enkf [epsilon time error ] : [0.03125 0.332599474 0.007341220271603545]
Enkf [epsilon time error ] : [0.015625 2.7421342395 0.0037548148161475883]
[576, 36, 9, 2]
MLEnkf [epsilon time error ] : [0.0625 1.2738122255 0.020550148440754038]
[576, 36, 9, 2]
MLEnkf [epsilon time error ] : [0.0625 0.1615274865 0.020550148440754038]
[4096, 256, 64, 16, 4]
MLEnkf [epsilon time error ] : [0.03125 1.2479345395 0.007883017239779706]
[25600, 1600, 400, 100, 25, 6]
MLEnkf [epsilon time error ] : [0.015625 8.3037271505 0.004051366415087424]
[108 7 3 1 1; 7 3 1 1 0; 3 1 1 0 0; 1 1 0 0 0; 1 0 0 0 0]
MIEnkf [epsilon time error] : [0.0625 0.524570805 0.020080140922586]
[108 7 3 1 1; 7 3 1 1 0; 3 1 1 0 0; 1 1 0 0 0; 1 0 0 0 0]
MIEnkf [epsilon time error] : [0.0625 0.081718582 0.020080140922586]
[432 26 9 4 2 1; 26 9 4 2 1 0; 9 4 2 1 0 0; 4 2 1 0 0 0; 2 1 0 0 0 0; 1 0 0 0 0 0]
MIEnkf [epsilon time error] : [0.03125 0.3216693635 0.011251203861902056]
[1722 102 36 13 5 2 1; 102 36 13 5 2 1 0; 36 13 5 2 1 0 0; 13 5 2 1 0 0 0; 5 2 1 0 0 0 0; 2 1 0 0 0 0 0; 1 0 0 0 0 0 0]
MIEnkf [epsilon time error] : [0.015625 1.2299740435 0.005954830185727395]
```
    
where for all EnKF, MLEnKF and MIEnKF, the first column shows the tolerance inputs, the  second column refers to runtime and the last one is for RMSE of the mean. The lines above MLEnKF and MIEnKF show the decreasing sequence of Monte Carlo sample sizes used, respectively, at each level/index in MLEnKF/MIEnKF given the epsilon-input value. 
    
The above command also saves the measurement series, the underlying truth, runtime, the abstract cost, the time-averaged RMSE of mean and covariance for all EnKF, MLEnKF and MIEnKF, respectively, in files `enkf$(problemText)_T$(T).jld` , `mlenkf$(problemText)_T$(T).jld` and `mienkf$(problemText)_T$(T).jld`, according to the chosen problem and the simulation length. The files will be saved in the same folder of `mienkf.jl`.
   
3.  In order to plot the results on convergence rates of the methods, run the following:
 
    **julia>** `plotResults("enkf$(problemText)_T$(T).jld","mlenkfOld$(problemText)_T$(T).jld","mlenkf$(problemText)_T$(T).jld")`
    
    according to saved file names. For example, $(problemText)="DoubleWell" and $(T)="20". This provides convergence rates of mean in Figure 2 and covariance in Figure 3, comparing the performance of EnKF, MLEnKF  and MIEnKF methods in terms of runtime vs RMSE. The reference triangle parameters should be adjusted according to each simulation by altering the respective lines in function "plotResults" of `mienkf.jl` and `mienkf_Langevin.jl` files.

4. Note also that the program runs in parallel. The number of workers is set in the beginning of the code by "const parallel_procs = 6;" (where we observe that only 5 are 100% active during long computations). Due to the parallelism, care has been taken to employ non-overlapping random seeds on different parallel processes through the function "Future.randjump()". Parallel computations are executed through the "pmap" function. The estimation of wall-clock runtime is done as follows for EnKF (and analogously for MLEnKF):

```
const parallel_procs = 6;
if nprocs()<parallel_procs
    addprocs(parallel_procs -nprocs());
end
const workers = nprocs()-1;
...
timeEnkf = @elapsed output = pmap(rng -> EnKF(N, P, y, rng), rngSeedVec); #Time it takes for 'workers' many parallel processes 
																		  #to compute all EnKF runs
...
timeEnkf*= workers/numRuns; #this is equal to the average time it takes for one worker/process to compute one EnKF run
```

## Reference

The algorithm implemented in this package is based on the manuscript 

*"Multi-index Ensemble Kalman Filtering"*

by H.Hoel, G.Shaimerdenova and R.Tempone, 2021 (to appear on ArXiv).
