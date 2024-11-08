 # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8

import torch
import numpy as np

class DDPMSampler:
    def __init__(self , generator : torch.Generator , numTrainingSteps = 1000,betaStart : float = 0.00085 , betaEnd:float = 0.0120):
        self.betas = torch.linspace(betaStart ** 0.5 ,beta **0.5 ,numTrainingSteps, dtype=torch.float32)**2
        self.alphas = 1.0 - self.betas
        self.alphasCumprod = torch.cumprod(self.alphas , dim = 0)
        self.one = torch.tensor(1.0)
        self.generator =  generator
        self.numTrainTimesteps = numTrainingSteps
        self.timesteps = torch.from_numpy(np.arrange(0,numTrainingSteps)[::-1].copy())

    def setInferenceTimesteps(self , numInferenceSteps = 50):
        self.numInferenecSteps = numInferenceSteps
        stepRatio = self.numTrainTimesteps // self.numInferenecSteps
        timesteps = (np.arrange(0,numInferenceSteps)* stepRatio).round().copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def getPreivousTimestep(self , timestep:int)-> int:
        prevT = timestep = self.numTrainTimesteps// self.numInferenecSteps
        return prevT

    def getVariance(self , timestep : int)-> torch.Tensor:
        prevT = self.getPreivousTimestep(timestep)

        alphaProdT = self.alphasCumprod(timestep)
        alphaProdTPrev = self.alphasCumprod(prevT)if prevT >= 0 else self.one
        currentBetaT = 1 - alphaProdT / alphaProdTPrev

        variance = (1- alphaProdTPrev)/ (1 - alphaProdT) *currentBetaT
        variance = torch.clamp(variance, min = 1e-20)
        return variance

    def setStrength(self , strength = 1):

        startStep = self.numInferenecSteps - int(self.numInferenecSteps *strength)
        self.timesteps = self.timesteps[startStep :]
        self.startStep = startStep

    def step(self , timestep : int , latents : torch.Tensor , modelOutput: torch.Tensor):
        t = timestep
        prevT = self.getPreivousTimestep(t)

        alphaProdT = self.alphasCumprod[t]
        alphaProdTPrev = self.alphasCumprod[prevT] if prevT>= 0 else self.one
        betaProdT = 1- alphaProdT
        betaProdTPrev = 1 - alphaProdTPrev
        currentAlphaT = alphaProdT / alphaProdTPrev
        currentBetaT  = 1 - currentAlphaT

        #formula 15
        predX_not = (latents - betaProdT ** 0.5 * modelOutput)/ alphaProdT** (0.5)
        predX_notCoefficient = (alphaProdTPrev **(0.5) * currentBetaT)/betaProdT
        currentSampleCoefficent = currentAlphaT ** (0.5) * betaProdTPrev / betaProdT

        predPrevSample = predX_notCoefficient * predX_not + currentSampleCoefficent * latents

        std_deviatioon = 0

        if t >0:
            device = modelOutput.device
            noise = torch.randn(modelOutput , generator=self.generator, device=device, dtype=modelOutput.dtype)
            std_deviation = (self.getVariance(t) ** (0.5))*noise

        predPrevSample = predPrevSample + std_deviation

        return predPrevSample
    


    # equaition 4
    def addNoiseForwardStep(
            self,
            originalSamples : torch.FloatTensor,
            timesteps : torch.IntTensor,
    ) -> torch.FloatTensor:
        alphasCumprod = self.alphasCumprod.to(device=originalSamples.device , dtype=originalSamples.dtype)
        timesteps = timesteps.to(originalSamples.device)

        sqrtAlphaProd = alphasCumprod[timesteps] ** 0.5
        sqrtAlphaProd = sqrtAlphaProd.flatten()
        while len(sqrtAlphaProd.shape) < len(originalSamples.shape):
            sqrtAlphaProd = sqrtAlphaProd.unsqueeze(-1)
        
        sqrtOneMinusAlphaProd = (1- alphasCumprod[timesteps]) ** 0.5
        sqrtOneMinusAlphaProd = sqrtOneMinusAlphaProd.flatten()
        while len(sqrtOneMinusAlphaProd.shape) < len(originalSamples.shape):
            sqrtOneMinusAlphaProd = sqrtOneMinusAlphaProd.unsqueeze(-1)

        noise = torch.randn( originalSamples.shape ,generator=self.generator, device=originalSamples.device , dtype=originalSamples.dtype)
        noisySamples = sqrtAlphaProd * originalSamples + sqrtOneMinusAlphaProd * noise
        return noisySamples
    
                              