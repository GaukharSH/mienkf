############### Introduction #################

# Tested in Julia 1.2.0 and Julia 1.5.2 

# (thread-locality should not matter for/improve this program as we implement parallel
# random number generation through passing different instances of rng objects
# that are non-overlappingly seeded to the different parallel processes).

# If having problems importing packages, one suggestion
## -- AT, OF COURSE, YOUR OWN RESPONSIBILITY --
# is to wipe clean your julia installation (removing all previously
# imported packages etc.) by calling the following command in a terminal (for Linux and OSX)
# $rm -rf ~/.julia

###########  end introduction ##############

import Pkg
# Add all packages in first run
Pkg.add("PyPlot")
Pkg.add("JLD")
Pkg.add("LaTeXStrings")
Pkg.add("DSP")#digital signal processing for the convolution function
Pkg.add("Dierckx")
Pkg.add("Distributions")
Pkg.add("BenchmarkTools")


using Distributed
const parallel_procs = 11;
if nprocs()<parallel_procs
    addprocs(parallel_procs -nprocs());
end
const workers = nprocs()-1;
@everywhere using Distributed

using SparseArrays
using SuiteSparse
using BenchmarkTools
using Dierckx #Spline1D needs this library
using LinearAlgebra
using LaTeXStrings
using DSP
using Random
using Distributions
using PyPlot
using JLD
using SharedArrays
@everywhere using Random
@everywhere using Distributions

import Future.randjump

### DEFINING THE PROBLEM WITH GLOBAL PARAMETERS BELOW

### DOUBLE WELL Potential
const problemText="Langevin";
@everywhere V(x)      = @. (2/(1+2*x^2) + x^2)/4;
@everywhere dVdx(x)   = @. (-8*x/((1+2*x^2)^2) .+ 2*x)/4;
@everywhere d2Vd2x(x) = @. ((-8+32*x^2+96*x^4)/(1+2*x^2)^4+2)/4;


# Problem parameters (made global and fixed and constant)
@everywhere const sigma=0.5;
@everywhere const Gamma=0.1;
@everywhere const u0=0.;
@everywhere const tau=1.0;
@everywhere const T=10;
@everywhere const nobs = round(Int,T/tau)+1;
@everywhere const Temp=1.0;
@everywhere const kappa=2^(-5)*pi^2;
@everywhere const H=[1.0 0.0];
@everywhere const G=[Gamma];
@everywhere const Id=[1.0 0.0; 0.0 1.0];
#

function testEstimatorRates()

    ### GENERATING ONE UNDERLYING PATH u and measurement series y
    rng = MersenneTwister(43231242);
    u = zeros(nobs); y   = zeros(nobs);
    u,y = generatePathAndObs_Langevin(rng);

    ### REFERENCE SOLUTION for Langevin
    epsRef=2. ^(-11); #error tolerance
    numRunsRef=180; #runs/realizations of reference sol.
    m,c = computePseudoRefSol_Langevin(y, numRunsRef, epsRef, rng);
    
    save("RefSol_$(problemText)_T$(T).jld", "EpsRef", epsRef, "y", y,"u", u, "m", m,"c", c, "sigma", sigma, "Gamma",Gamma, "u0", u0, "Temp", Temp, "kappa", kappa, "H", H)
    
    println("Done computing refSol");
    numRuns = 90; #runs/realizations of enkf and mlenkа

    #generate a sequence of random seeds for parallel computations spaces
    #that is "numRuns" long and each seed is spaced
    #10^20 draws apart in the cycle
    rngSeedVec = let r = rng
        [r; accumulate(randjump, fill(big(10)^20, numRuns), init=r)]
    end;


#######EnKF##############

    epsVec = 2. .^(-[4 4 5 6 7 8 9] );

    # storage of EnKF output
    timeEnkf = zeros(length(epsVec));
    costEnkf = zeros(length(epsVec));
    #mEnkf    = zeros(length(epsVec),nobs,3);
    #cEnkf    = zeros(length(epsVec),nobs,3);
    errMEnkf = zeros(length(epsVec));
    errCEnkf = zeros(length(epsVec));
    
    for i=1:length(epsVec)
        
        N = round(Int, 1/epsVec[i]);
        P = round(Int, 12*(1/epsVec[i])^2);

        timeEnkf[i] += @elapsed output =  pmap(rng ->EnKF_Langevin(N, P, y, rng), rngSeedVec);
        
        for j=1:numRuns

            mTmp = output[j][1];
            cTmp = output[j][2];
            
	    errMEnkf[i]+= norm(mTmp - m)^2/(nobs*numRuns);
            errCEnkf[i]+= norm(cTmp -c)^2/(nobs*numRuns);	

        end    
        
        timeEnkf[i]*= workers/numRuns;
        errMEnkf[i] = sqrt(errMEnkf[i]);
        errCEnkf[i] = sqrt(errCEnkf[i]);
        costEnkf[i] = P*N;
        println("Enkf [epsilon time error ] : ", [epsVec[i] timeEnkf[i] errMEnkf[i]])
        save("enkfLangevin.jld", "u", u, "y", y, "timeEnkf", timeEnkf, "costEnkf", costEnkf, "errMEnkf", errMEnkf, "errCEnkf", errCEnkf);
    end

    # throw away first computation (which due to first-run-JIT-compilation gives an incorrect/too large runtime)
    timeEnkf = timeEnkf[2:end];
    errMEnkf = errMEnkf[2:end];
    errCEnkf = errCEnkf[2:end];
    costEnkf = costEnkf[2:end];
    save("enkf$(problemText)_T$(T).jld", "u", u, "y", y, "timeEnkf", timeEnkf, "costEnkf", costEnkf, "errMEnkf", errMEnkf, "errCEnkf", errCEnkf);



#######MLEnKF#################

    epsVec = 2. .^(-[4 4 5 6 8 9] );
    timeMLEnkf = zeros(length(epsVec));
    costMLEnkf = zeros(length(epsVec));
    mMLEnkf = zeros(length(epsVec),nobs);
    cMLEnkf = zeros(length(epsVec),nobs);
    errMMLEnkf = zeros(length(epsVec));
    errCMLEnkf = zeros(length(epsVec));
   
    for i=1:length(epsVec)
        
        L  = round(Int, log2(1/epsVec[i]))-1;
        Nl = 4*2 .^(0:L);
        Pl = 8*2 .^(0:L);
        Ml = zeros(Int, size(Nl));
      
        #classic Ml
        for l=1:L+1           
           Ml[l] = round(Int, L^2 *epsVec[i]^(-2)/(Nl[l]^2));
        end
        Ml[1] = 4*Ml[1];
        
        timeMLEnkf[i] = @elapsed output = pmap(rng -> MLEnKF(Ml, Nl, Pl, L, y, rng), rngSeedVec);
        for j=1:numRuns
            mTmp = output[j][1];
            cTmp = output[j][2];

	    errMMLEnkf[i] += norm(mTmp - m)^2/(nobs*numRuns);
            errCMLEnkf[i] += norm(cTmp -c)^2/(nobs*numRuns);	

        end    
        timeMLEnkf[i]*= workers/numRuns;
        errMMLEnkf[i] = sqrt(errMMLEnkf[i]);
        errCMLEnkf[i] = sqrt(errCMLEnkf[i]);
        costMLEnkf[i] = sum(Pl.*Ml.*Nl);
        println(Ml)
        println("MLEnkf [epsilon time error ] : ", [epsVec[i] timeMLEnkf[i] errMMLEnkf[i]])
        save("mlenkf.jld", "u", u, "y", y, "timeMLEnkf", timeMLEnkf, "costMLEnkf",
             costMLEnkf, "errMMLEnkf", errMMLEnkf, "errCMLEnkf", errCMLEnkf);
        
    end

    timeMLEnkf = timeMLEnkf[2:end];
    errMMLEnkf = errMMLEnkf[2:end];
    errCMLEnkf = errCMLEnkf[2:end];
    costMLEnkf = costMLEnkf[2:end];
    
    save("mlenkf$(problemText)_T$(T).jld", "u", u, "y", y, "timeMLEnkf", timeMLEnkf, "costMLEnkf",
         costMLEnkf, "errMMLEnkf", errMMLEnkf, "errCMLEnkf", errCMLEnkf);


########MIEnKF####################

    epsVec = 2. .^(-[4 4 5 6 7 8 9 10] );
    timeMIEnkf = zeros(length(epsVec));  
    costMIEnkf = zeros(length(epsVec));
    mMIEnkf = zeros(length(epsVec),nobs);
    cMIEnkf = zeros(length(epsVec),nobs);
    errMMIEnkf = zeros(length(epsVec));
    errCMIEnkf = zeros(length(epsVec));
   

    for i=1:length(epsVec)

        Lstar = round(Int, log2(1/epsVec[i]))-1;
        L=round(Int,Lstar+log2(Lstar))-1;
        Nl  = 4*2 .^(0:L);
        Pl  = 20*2 .^(0:L);
        Ml = zeros(Int, L+1,L+1);
    
        for l1=1:L+1  
            for l2=1:L+2-l1
               Ml[l1,l2] = ceil(Int, 50*epsVec[i]^(-2)*Nl[l1]^(-3/2)*Pl[l2]^(-3/2));
            end
        end
    Ml[1,1] = 6Ml[1,1];
         
      timeMIEnkf[i] += @elapsed output =  pmap(rng -> MIEnKF_Langevin(Ml, Nl, Pl, L, y, rng), rngSeedVec);
     
        for j=1:numRuns

            mTmp = output[j][1];
            cTmp = output[j][2];

	    errMMIEnkf[i] += norm(mTmp - m)^2/(nobs*numRuns);
            errCMIEnkf[i] += norm(cTmp -c)^2/(nobs*numRuns);	

        end    
        timeMIEnkf[i]*= workers/numRuns;
        errMMIEnkf[i] = sqrt(errMMIEnkf[i]);
        errCMIEnkf[i] = sqrt(errCMIEnkf[i]);
        costMIEnkf[i] = sum(Ml.*(Nl*Pl'));
        println(Ml)
        println("MIEnkf [epsilon time error] : ", [epsVec[i] timeMIEnkf[i] errMMIEnkf[i]])
        #println("m=", [mMIEnkf[i,:]], "c=", [cMIEnkf[i,:]])
        save("mienkfLangevin.jld", "u", u, "y", y, "timeMIEnkf", timeMIEnkf, "costMIEnkf",
             costMIEnkf, "errMMIEnkf", errMMIEnkf, "errCMIEnkf", errCMIEnkf);

    end    

    timeMIEnkf = timeMIEnkf[2:end];
    errMMIEnkf = errMMIEnkf[2:end];
    errCMIEnkf = errCMIEnkf[2:end];
    costMIEnkf = costMIEnkf[2:end];
    
    save("mienkf$(problemText)_T$(T).jld", "u", u, "y", y, "timeMIEnkf", timeMIEnkf, "costMIEnkf",
         costMIEnkf, "errMMIEnkf", errMMIEnkf, "errCMIEnkf", errCMIEnkf);

        end

function generatePathAndObs_Langevin(rng)
    nDt =10000;
    dt = tau/nDt;

    u = zeros(2,nDt*(nobs-1)+1); y   = zeros(1, nobs);
    u[:,1] = u0 .+ sqrt(Gamma)*randn(rng,2,1);
 
    y[:,1] = H*u[:,1] + sqrt(Gamma)*randn(rng, 1);
    r=exp(-kappa*dt);
    sigOU=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dt))/(2*kappa));

    for n=1:(nobs-1)*nDt
       u[2,n+1]= r*u[2,n]+sigOU*randn(rng);
       u[2,n+1]= u[2,n+1]-dVdx(u[1,n])*dt
       u[1,n+1]= u[1,n]+u[2,n+1]*dt;

        if mod(n,nDt)==0
             y[:,round(Int,n/nDt)+1] = H*u[:,n+1] + sqrt(Gamma)*randn(rng,1);
        end
    end

   return u, y
end

function computePseudoRefSol_Langevin(y, numRunsRef, epsRef, rng)

    #generate a sequence of random seeds for parallel computations spaces
    #that is "numRuns" long and each seed is spaced
    #10^20 draws apart in the cycle
    rngSeedVec = let r = rng
        [r; accumulate(randjump, fill(big(10)^20, numRunsRef), init=r)]
    end;

    LstarRef = round(Int, log2(1/epsRef))-1;
    LRef=round(Int,LstarRef+log2(LstarRef))-1;
    NlRef  = 4*2 .^(0:LRef);
    PlRef  = 30*2 .^(0:LRef);
    MlRef = zeros(Int, LRef+1,LRef+1);
    
        for l1=1:LRef+1  
            for l2=1:LRef+2-l1
               MlRef[l1,l2] = ceil(Int, 90*epsRef^(-2)*NlRef[l1]^(-3/2)*PlRef[l2]^(-3/2));
            end
        end
    MlRef[1,1] = 6MlRef[1,1];

    N=1000; P=10000;
    mTmp = zeros(2,nobs); cTmp= zeros(2,2, nobs);
        output=pmap(rng -> MIEnKF_Langevin(MlRef, NlRef, PlRef, LRef, y, rng), rngSeedVec);

    for j=1:numRunsRef
          mTmp += output[j][1];
          cTmp += output[j][2];
    end
    m =mTmp./numRunsRef;
    c =cTmp./numRunsRef;
    
    return m, c
end 


function plotResults(enkfFile, mlenkfFile, mienkfFile)

 
    enkf= load(enkfFile);
    mienkf= load(mienkfFile);
    mlenkf= load(mlenkfFile)

    tEnkf    = enkf["timeEnkf"];
    errMEnkf = enkf["errMEnkf"];
    errCEnkf = enkf["errCEnkf"];
    costEnkf = enkf["costEnkf"];
    
    tMI    = mienkf["timeMIEnkf"];
    errMMI = mienkf["errMMIEnkf"];
    errCMI = mienkf["errCMIEnkf"];
    costMI = mienkf["costMIEnkf"];

    tML    = mlenkf["timeMLEnkf"];
    errMML = mlenkf["errMMLEnkf"];
    errCML = mlenkf["errCMLEnkf"]; 
    costML = mlenkf["costMLEnkf"]; 

   figure(2)

    loglog(tEnkf[1:end], errMEnkf[1:end], "k-o", tEnkf[1:end], tEnkf[1:end].^(-1. /3)/50., "k--", tMI, errMMI, "k-*", tMI, tMI.^(-1. /2)/45, "k:", tML[1:end], errMML[1:end], "k-x", tML[1:end], log.(10 .+tML[1:end]).^(1. /3) .* tML[1:end].^(-1. /2)/20., "k-.")
    xlabel("Runtime [sec]");    ylabel("RMSE");
    show()

   figure(3)

    loglog(tEnkf[1:end], errCEnkf[1:end], "k-o", tEnkf[1:end], tEnkf[1:end].^(-1. /3)/290., "k--", tMI, errCMI, "k-*", tMI, tMI.^(-1. /2)/280, "k:",tML[1:end], errCML[1:end], "k-x", tML[1:end], log.(10 .+tML[1:end]).^(1. /3) .* tML[1:end].^(-1. /2)/110., "k-.")
    xlabel("Runtime [sec]");    ylabel("RMSE");
  
 show()
end

@everywhere function Psi_Langevin(u, N, dt,rng)
    sqrtDt = sqrt(dt);
    P = length(u[1,:]);
    r=exp(-kappa*dt);
    sigOU=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dt))/(2*kappa));

    if(P<1000)
	for n=1:N
	    u[2,:]= r*u[2,:] + sigOU*randn(rng, P);
            u[2,:]= u[2,:]-dVdx(u[1,:])*dt;
            u[1,:]= u[1,:]+u[2,:]*dt;
    	end
    else
        iter = floor(Int, P/1000);
        pStart=1; pEnd = 1000;
        for k=1:iter
            for n=1:N
               u[2,pStart:pEnd]= r*u[2,pStart:pEnd] + sigOU*randn(rng, 1000);
               u[2,pStart:pEnd]= u[2,pStart:pEnd]-dVdx(u[1,pStart:pEnd])*dt;
               u[1,pStart:pEnd]= u[1,pStart:pEnd]+u[2,pStart:pEnd]*dt;
            end
            pStart+=1000; pEnd +=1000;
        end

        for n=1:N
               u[2,pStart:end]= r*u[2,pStart:end] + sigOU*randn(rng, P-pStart+1);
               u[2,pStart:end]= u[2,pStart:end]-dVdx(u[1,pStart:end])*dt;
               u[1,pStart:end]= u[1,pStart:end]+u[2,pStart:end]*dt;
        end
    end
    return u;
end


@everywhere function EnKF_Langevin(N, P, y, rng)
    m = zeros(2,nobs); c= zeros(2,2, nobs);
    m[:,1] = [u0; u0];
    c[:,:,1] = [Gamma 0.0; 0.0 Gamma];


    dt   = tau/N;
    v = m[:,1].+sqrt(c[:,:,1])*randn(rng,2,P);

    for n=2:nobs
	
	v = Psi_Langevin(v, N, dt, rng);
	
	c[:,:,n] = cov(v, dims=2);#Prediction covariance
	K        = c[:,:,n]*H'/(G+H*c[:,:,n]*H');#Kalman gain
	yTilde   = y[:, n].+ sqrt(Gamma)*randn(rng,1,P);#perturbed observation
	
        v        = (Id-K*H)*v+K*yTilde;#ensemble update
        m[:,n]   = mean(v, dims=2);# update mean and covariance 
        c[:,:,n] = cov(v, dims=2);

    end

    return m,c;
end


@everywhere function Psi4(uCC, uCF, uFC, uFF, nC,dtF,rng)
    # Assuming here that dtC = 2dtF
    sqrtDtF = sqrt(dtF); P = length(uFF[1,:]); dtC = 2*dtF;
    
    xCC=uCC[1,:]; xCF=uCF[1,:]; xFC=uFC[1,:]; xFF=uFF[1,:];
    vCC=uCC[2,:]; vCF=uCF[2,:]; vFC=uFC[2,:]; vFF=uFF[2,:];

    rf=exp(-kappa*dtF);
    rc=exp(-kappa*dtC);
    sigOUf=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dtF))/(2*kappa));
    sigOUc=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dtC))/(2*kappa));

    for n=1:nC
        w1 = randn(rng,P);
        w2 = randn(rng,P);

        vFF = vFF*rf+sigOUf*w1;
        vFF = vFF - dVdx(xFF)*dtF;
        xFF = xFF + vFF*dtF;
    
        vFF = vFF*rf+sigOUf*w2;
        vFF = vFF - dVdx(xFF)*dtF;
        xFF = xFF + vFF*dtF;


        vFC = vFC*rf+sigOUf*w1;
        vFC = vFC - dVdx(xFC)*dtF;
        xFC = xFC + vFC*dtF;

        vFC = vFC*rf+sigOUf*w2;
        vFC = vFC - dVdx(xFC)*dtF;
        xFC = xFC + vFC*dtF;

        w2star=(rf*w1+w2)/(sqrt(rf^2+1));
        
        vCF = vCF*rc+sigOUc*w2star;
        vCF = vCF-dVdx(xCF)*dtC;
        xCF = xCF + vCF*dtC;

        vCC = vCC*rc+sigOUc*w2star
        vCC = vCC -dVdx(xCC)*dtC;
        xCC = xCC + vCC*dtC;

    end
    uCC=[xCC vCC]'; uCF=[xCF vCF]'; uFC=[xFC vFC]'; uFF=[xFF vFF]';
    
    return uCC, uCF, uFC, uFF;
end

@everywhere function Psi1(uFC,uFF,nF,dtF,rng)
    sqrtDtF = sqrt(dtF); P = length(uFF[1,:]);
    xFC=uFC[1,:]; xFF=uFF[1,:];
    vFC=uFC[2,:]; vFF=uFF[2,:];

    rf=exp(-kappa*dtF);
    sigOUf=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dtF))/(2*kappa));


    for n=1:nF
        w  = randn(rng,P);

        vFF = vFF*rf+sigOUf*w;
        vFF = vFF - dVdx(xFF)*dtF;
        xFF = xFF + vFF*dtF;

        vFC = vFC*rf+sigOUf*w;
        vFC = vFC - dVdx(xFC)*dtF;
        xFC = xFC + vFC*dtF;

    end

    uFF=[xFF vFF]'; uFC=[xFC vFC]';

    return uFC,uFF;
end


@everywhere function Psi2(uCF, uFF, nC, dtF, rng)
    # Assuming here that dtC = 2dtF
    sqrtDtF = sqrt(dtF); P = length(uFF[1,:]); dtC = 2*dtF;
    
    xCF=uCF[1,:]; xFF=uFF[1,:];
    vCF=uCF[2,:]; vFF=uFF[2,:];
    rf=exp(-kappa*dtF);
    rc=exp(-kappa*dtC);
    sigOUf=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dtF))/(2*kappa));
    sigOUc=sqrt(2*kappa*Temp*(1-exp(-2*kappa*dtC))/(2*kappa));

    for n=1:nC
        w1 = randn(rng,P);
        w2 = randn(rng,P);

        vFF = vFF*rf+sigOUf*w1;
        vFF = vFF - dVdx(xFF)*dtF;
        xFF = xFF + vFF*dtF;

        vFF = vFF*rf+sigOUf*w2;
        vFF = vFF - dVdx(xFF)*dtF;
        xFF = xFF + vFF*dtF;
        
        w2star=(rf*w1+w2)/(sqrt(rf^2+1));
        
        vCF = vCF*rc+sigOUc*w2star;
        vCF = vCF-dVdx(xCF)*dtC;
        xCF = xCF + vCF*dtC;

    end

    uCF=[xCF vCF]'; uFF=[xFF vFF]';

    return uCF,uFF;
end

@everywhere function MIEnKF_Langevin(Ml, Nl, Pl, L, y, rng)
    m = zeros(2,nobs); c= zeros(2,2, nobs);


    dtl = tau ./Nl;

    # solve for l1=1 and l2=1
    for i=1:Ml[1,1]
        mTmp, cTmp = EnKF_Langevin(Nl[1], Pl[1], y, rng);
        m+=mTmp/Ml[1,1];
        c+=cTmp/Ml[1,1];
    end

   # solve for l1=1 and l2=2,3,...
    for l2=2:L+1
        plHalf = Pl[l2-1]; 
        mTmp = zeros(2,nobs); cTmp= zeros(2,2,nobs);

        for i=1:Ml[1,l2]
            #Initialisation for MIEnKF
            uFF = u0 .+ sqrt(Gamma)*randn(rng,2,Pl[l2]); 
            uFC = uFF; 

          for n=2:nobs
            uFC,uFF = Psi1(uFC,uFF,Nl[1],dtl[1],rng);
            covFF  = cov(uFF, dims=2); 
            covFC1 = cov(uFC[:,1:plHalf], dims=2); covFC2 = cov(uFC[:,plHalf+1:end], dims=2);

            kFF  = covFF*H'/(H*covFF*H' + G);
            kFC1 = covFC1*H'/(H*covFC1*H' + G);
            kFC2 = covFC2*H'/(H*covFC2*H' + G);

            yTilde   = y[:, n].+ sqrt(Gamma)*randn(rng,1,Pl[l2]);#perturbed observation
                       
            uFF                  = (Id-kFF*H)*uFF+kFF*yTilde; #ensemble update
            uFC[:, 1:plHalf]     = (Id-kFC1*H)*uFC[:, 1:plHalf]+kFC1*yTilde[:, 1:plHalf];
            uFC[:, plHalf+1:end] = (Id-kFC2*H)*uFC[:, plHalf+1:end]+kFC2*yTilde[:, plHalf+1:end];
           
            mTmp[:, n]    += mean(uFF-uFC, dims=2)./Ml[1,l2];# update mean and covariance 
            cTmp[:,:, n]  += (cov(uFF,dims=2)-(cov(uFC[:, 1:plHalf], dims=2)+cov(uFC[:, plHalf+1:end], dims=2))/2)./Ml[1,l2];
           end
        end
     m +=mTmp; c +=cTmp;
    end

    # solve for l2=1 and l1=2,3,...
    
    for l1=2:L+1
        mTmp = zeros(2, nobs); cTmp= zeros(2,2, nobs);
        for i=1:Ml[l1,1]
            #Initialisation for MIEnKF
            uFF = u0 .+ sqrt(Gamma)*randn(rng,2,Pl[1]);  
            uCF = uFF;
       
           for n=2:nobs
            uCF,uFF = Psi2(uCF,uFF,Nl[l1-1],dtl[l1],rng);
            covFF  = cov(uFF, dims=2);covCF = cov(uCF, dims=2);

            kFF  = covFF*H'/(H*covFF*H' + G);
            kCF  = covCF*H'/(H*covCF*H' + G);
            
            yTilde   = y[:, n].+ sqrt(Gamma)*randn(rng,1,Pl[1]);#perturbed observation  
        
            uFF    = (Id-kFF*H)*uFF+kFF*yTilde; #ensemble update
            uCF    = (Id-kCF*H)*uCF+kCF*yTilde; #ensemble update
           
            mTmp[:, n]    += mean(uFF-uCF, dims=2)./Ml[l1,1];# update mean and covariance 
            cTmp[:,:, n]  += (cov(uFF, dims=2)-cov(uCF, dims=2))./Ml[l1,1];
           end
        end
     m +=mTmp; c +=cTmp;
    end


    # solve for l1>1, l2>1
    for l1=2:L
        mTmp2 = zeros(2,nobs); cTmp2= zeros(2,2, nobs);
           for l2=2:L+2-l1
                                        
               plHalf = round(Int,Pl[l2]/2);
               mTmp1 = zeros(2,nobs); cTmp1= zeros(2,2, nobs);
               for i=1:Ml[l1,l2]
                        
                        #Initialisation for MIEnKF
                        uFF = u0 .+ sqrt(Gamma)*randn(rng,2,Pl[l2]); 
                        uFC = uFF; uCF = uFF; uCC = uFF;
                       

                        for n=2:nobs
                           uCC, uCF, uFC, uFF = Psi4(uCC, uCF, uFC, uFF, Nl[l1-1], dtl[l1], rng);
                           covFF  = cov(uFF, dims=2); covCF = cov(uCF, dims=2); 
                           covFC1 = cov(uFC[:,1:plHalf], dims=2); covFC2 = cov(uFC[:, plHalf+1:end], dims=2);
                           covCC1 = cov(uCC[:,1:plHalf], dims=2); covCC2 = cov(uCC[:, plHalf+1:end], dims=2);

                           kFF  = covFF*H'/(H*covFF*H' + G);
                           kCF  = covCF*H'/(H*covCF*H' + G);
                           kFC1 = covFC1*H'/(H*covFC1*H' + G);
                           kFC2 = covFC2*H'/(H*covFC2*H' + G);
                           kCC1 = covCC1*H'/(H*covCC1*H' + G);
                           kCC2 = covCC2*H'/(H*covCC2*H' + G);

                           yTilde   = y[:, n].+ sqrt(Gamma)*randn(rng,1,Pl[l2]);#perturbed observation

                           uFF                  = (Id-kFF*H)*uFF+kFF*yTilde;  #ensemble update x component
                           uCF                  = (Id-kCF*H)*uCF+kCF*yTilde;
                           uFC[:, 1:plHalf]     = (Id-kFC1*H)*uFC[:, 1:plHalf]+kFC1*yTilde[:, 1:plHalf];
                           uFC[:, plHalf+1:end] = (Id-kFC2*H)*uFC[:, plHalf+1:end]+kFC2*yTilde[:, plHalf+1:end];
                           uCC[:, 1:plHalf]     = (Id-kCC1*H)*uCC[:, 1:plHalf]+kCC1*yTilde[:, 1:plHalf];
                           uCC[:, plHalf+1:end] = (Id-kCC2*H)*uCC[:, plHalf+1:end]+kCC2*yTilde[:, plHalf+1:end];
                         
                           mTmp1[:,n]          += mean(uFF-uCF-uFC+uCC, dims=2)./Ml[l1,l2];# update mean and covariance 
                           cTmp1[:,:,n]        += (cov(uFF, dims=2)-(cov(uFC[:,1:plHalf], dims=2)+cov(uFC[:,plHalf+1:end],dims=2))/2-cov(uCF, dims=2)+(cov(uCC[:,1:plHalf], dims=2)+cov(uCC[:,plHalf+1:end],dims=2))/2)./Ml[l1,l2];
                        end
              end

           mTmp2 +=mTmp1; cTmp2 +=cTmp1;
        end
     m +=mTmp2; c +=cTmp2;
   end

    return m,c;
end



@everywhere function MLEnKF(Ml, Nl, Pl, L, y, rng)

    m = zeros(2,nobs); c= zeros(2,2, nobs);

    dtl = tau ./Nl;

    # solve for l=1
    for i=1:Ml[1]
        mTmp, cTmp = EnKF_Langevin(Nl[1], Pl[1], y, rng);
        m+=mTmp/Ml[1];
        c+=cTmp/Ml[1];
    end
    
    # solve for l>1
    for l=2:L+1
        plHalf = round(Int,Pl[l]/2); 
        mTmp = zeros(2,nobs); cTmp= zeros(2,2, nobs);
        for i=1:Ml[l]
            
            vF = u0 .+ sqrt(Gamma)*randn(rng,2, Pl[l]); vC = vF;
	    for n=2:nobs
	        
	        vC,vF = Psi2(vC, vF, Nl[l-1], dtl[l], rng);
	        covF = cov(vF, dims=2); covC1 = cov(vC[:, 1:plHalf], dims=2); covC2 = cov(vC[:, plHalf+1:end], dims=2);
                kF  = covF*H'/(H*covF*H' + G);
                kC1 = covC1*H'/(H*covC1*H' + G);
                kC2 = covC2*H'/(H*covC2*H' + G);
                
                yTilde           = y[:, n].+ sqrt(Gamma)*randn(rng,1, Pl[l]);#perturbed observation
	         vF               = (Id-kF*H)*vF+kF*yTilde;#ensemble update
                vC[:, 1:plHalf]     = (Id-kC1*H)*vC[:, 1:plHalf]+kC1*yTilde[:, 1:plHalf];
                vC[:, plHalf+1:end] = (Id-kC2*H)*vC[:, plHalf+1:end]+kC2*yTilde[:, plHalf+1:end];
                
                mTmp[:, n]      += mean(vF-vC, dims=2)/Ml[l];# update mean and covariance 
                cTmp[:, :, n]   += (cov(vF, dims=2)-(cov(vC[:, 1:plHalf], dims=2)+cov(vC[:, plHalf+1:end], dims=2))/2.)/Ml[l];
            end
        end
        m +=mTmp; c +=cTmp;
    end
    return m,c;
end




