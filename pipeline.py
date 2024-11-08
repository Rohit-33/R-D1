import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH //8
LATENT_HEIGHT = WIDTH // 8

def generate(
        prompt,
        uncond_prompt = None,
        inputImage  = None,
        strength = 0.7,
        doCFG = True,
        CFGScale = 7.5,
        samplerName = "ddpm",
        nInferenceSteps = 50,
        models = {},
        seed = None,
        device = None,
        idleDevice = None,
        tokenizer = None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        if idleDevice:
            toIdle = lambda x : x.to(idleDevice)
        else:
            toIdle = lambda x : x

        generator = torch.Generator(device = device)
        
        if seed is None:
            generator = seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if doCFG:
            condTokens = tokenizer.batch_enocde_plus(
                [prompt], padding="max_length", max_length = 77
            ).input_ids
            condTokens = torch.tensor(condTokens, dtype=torch.long, device= device)
            condContext = clip(condTokens)

            uncondTokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding = "max_length", max_lenght = 77
            ).input_ids

            uncondTokens = torch.tensor(uncondTokens , dtype=torch.long, device= device)
            uncondContext = clip(uncondTokens)
            context = torch.cat([condContext, uncondContext])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt] , padding = "max_length", max_length = 77
            ).input_ids
            tokens = torch.tensor(tokens , dtype=torch.long, device = device)
            context = clip(tokens)

        toIdle(clip)

        if samplerName == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.setInferenceTimesteps(nInferenceSteps)
        else :
            raise ValueError("Unknown Sampler")
        
        latentShape = (1, 4 , LATENT_HEIGHT, LATENT_WIDTH)

        if inputImage:
            encoder = models["encoder"]
            encoder.to(device)
            inputImageTensor = inputImage.resize((HEIGHT, WIDTH))
            inputImageTensor = np.array(inputImageTensor)
            inputImageTensor = torch.tensor(inputImageTensor,dtype=torch.float32, device = device)
            inputImageTensor = rescale(inputImageTensor, (0, 255), (-1, 1))
            inputImageTensor = inputImageTensor.unsqueeze(0)
            inputImageTensor = inputImageTensor.permute(0,3,1,2)

            encoderNoise = torch.rand(latentShape, generator=generator, device=device)
            latents = encoder(inputImageTensor, encoderNoise)

            sampler.setStrength(strength=strength)
            latents= sampler.addNoiseForwardStep(latents,sampler.timesteps[0])
            toIdle(encoder)
        else:
            latents = torch.randn(latentShape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            timeEmbedding  = getTimeEmbedding(timestep).to(device)
            modelInput = latents
            if doCFG:
                modelInput = modelInput.repeat(2,1,1,1)
            modelOutput = diffusion(modelInput, context, timeEmbedding)

            if doCFG:
                outCond , outUncond = modelOutput.chunk(2)
                modelOutput= CFGScale*(outCond - outUncond) + outUncond
            latents = sampler.step(timestep, latents,modelOutput)
        
        toIdle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        toIdle(decoder)

        images = rescale(images, (-1, 1), (0,255), clamp = True)
        images = images.permute(0,2,3,1)
        images = images.to("cpu" , torch.uint8).numpy()
        return images

def rescale(x , oldRange, newRange , clamp = False):
    oldMin , oldMax = oldRange
    newMin , newMax = newRange
    x -= oldMin
    x *= (newMax - newMin)/(oldMax- oldMin)
    x += newMin
    if clamp:
        x = x.clamp(newMin , newMax)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    sin_x = torch.sin(x)
    cos_x = torch.cos(x)
    interleaved = torch.empty(2 * x.shape[1], dtype=torch.float32)
    interleaved[0::2] = sin_x.flatten()  
    interleaved[1::2] = cos_x.flatten()  
    return interleaved.unsqueeze(0)
embedding = get_time_embedding(1)
print(embedding)



    