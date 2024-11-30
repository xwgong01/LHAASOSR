import torch
import torch.nn.functional as F

def jsdiv(img1, img2):
    shape = img2.shape
    img1 = img1.reshape(shape[0],shape[1],-1)
    img2 = img2.reshape(shape[0],shape[1],-1)
    img1 = F.softmax(img1,-1)
    img2 = F.softmax(img2,-1)
    img1 = img1.reshape(shape)
    img2 = img2.reshape(shape)
    img_mean = (img1/2+img2/2)
    ks12 = (img1*torch.log(img1/img_mean)).sum(-1).sum(-1)
    ks21 = (img2*torch.log(img2/img_mean)).sum(-1).sum(-1)
    #print((img1*torch.log(img1/img2)))
    
    jsdivergence = ((ks12+ks21)/2).mean()
    #print(jsdivergence)
    #raise ValueError('Stop')
    
    return jsdivergence

def jsdiv_single(img1, img2):
    shape = img2.shape
    img1 = img1.reshape(shape[0],shape[1],-1)
    img2 = img2.reshape(shape[0],shape[1],-1)
    img1 = F.softmax(img1,-1)
    img2 = F.softmax(img2,-1)
    img1 = img1.reshape(shape)
    img2 = img2.reshape(shape)
    img_mean = (img1/2+img2/2)
    ks12 = (img1*torch.log(img1/img_mean)).sum(-1).sum(-1)
    ks21 = (img2*torch.log(img2/img_mean)).sum(-1).sum(-1)
    #print((img1*torch.log(img1/img2)))
    
    jsdivergence = ((ks12+ks21)/2)#.mean()
    #print(jsdivergence)
    #raise ValueError('Stop')
    
    return jsdivergence

# 使用示例
# img1和img2需要是torch张量，并且在CUDA设备上
# loss = ssim_function(img1, img2)