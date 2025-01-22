import numpy as np
def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0]))#2
    x_aug = np.multiply(x, factor[:, np.newaxis])#2*1
    return x_aug

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0]))
    rotate_axis = np.arange(x.shape[0])
    np.random.shuffle(rotate_axis)    
    x_aug = flip[:, np.newaxis] * x[rotate_axis, :]
    return x_aug

def permutation(x, max_segments=5, seg_mode="equal"):
    x  = x.transpose(1,0)
    x = x[np.newaxis,:,:]
    x_aug = np.zeros_like(x)
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return np.squeeze(ret.transpose((0, 2, 1)))

def magnitude_warp(x, sigma=0.2, knot=4):
    x  = x.transpose(1,0)
    x = x[np.newaxis,:,:]
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return np.squeeze(ret.transpose((0, 2, 1)))

def time_warp(x, sigma=0.2, knot=4):
    x  = x.transpose(1,0)
    x = x[np.newaxis,:,:]
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return np.squeeze(ret.transpose ((0, 2, 1)))

def window_slice(x, reduce_ratio=0.9):
    x  = x.transpose(1,0)
    x = x[np.newaxis,:,:]
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return np.squeeze(ret.transpose((0, 2, 1)))

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    x  = x.transpose(1,0)
    x = x[np.newaxis,:,:]
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return np.squeeze(ret.transpose((0, 2, 1)))

augmentations = [scaling, rotation, permutation, magnitude_warp, time_warp, window_slice, window_warp]