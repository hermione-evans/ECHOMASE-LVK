import numpy as np

def trianglecomb(height,phase,width,spacing,x):
    # 高度、相位、峰宽度、单个周期长度、频率（输入的x）
    '''
    the first parameter is the height of wave.
    the second parameter is phase, should between 0 to spacing. 
    When frequency = phase + n * spacing, the comb reaches its peaks. 
    As the phase grows larger, the comb goes right.
    the third parameter is width，which describes the width of signal.
    the fourth parameter is spacing, which describes the whole length of a wave.
    the fifth parameter is input frequency.
    The unit of frequency is Hz.
    '''    
    phase = phase * spacing
    # define a normalized phase
    x = x + width / 2
    xprime = ((x - phase) / spacing - np.floor((x - phase) / spacing)) * spacing - width / 2
    # here the range of xprime is [-width/2 , spacing-width/2], the length of xprime is spacing
    # when x = phase, xprime should = 0
    y = np.piecewise(xprime, [xprime < 0, np.logical_and(0 <= xprime, xprime < width / 2),xprime >= width /2 ], 
                     [lambda xprime:1 + 2 * xprime/width, lambda xprime:1 - 2 * xprime/width,0])
    return y * height


def trianglecombmodel(frequency, ** params):
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            params[key] = params[key][:,None]
    '''the params should be a dictionary containing four keys:
    amplitude,phase,width and spacing
    '''

    amplitude = params['amplitude'] * 1.0
    phase = params['phase'] * 1.0
    width = params['width'] * 1.0
    spacing = params['spacing'] * 1.0
    if isinstance(amplitude,np.ndarray) or isinstance(phase,np.ndarray) or isinstance(width,np.ndarray) or isinstance(spacing,np.ndarray):
        # 我感觉这个if应该有更优雅的写法但是没想出来...
        mu = trianglecomb(amplitude,phase,width,spacing,frequency[None,:])

        return mu
    else:
        # this allows a vectorised evaluation of the waveform for easy plotting
        mu = trianglecomb(amplitude,phase,width,spacing,frequency)

        return mu
    
def qnmcomb_cut(height,phase,width,spacing,duration,x):
    ##高度、相位、峰宽度、单个周期长度、频率（输入的x）
    # cut 表示在归一化的峰高小于cut时截断，目前暂时取的是0.2
    '''
    the first parameter is the height of wave.
    the second parameter is phase, should between 0 to spacing.
    When frequency = phase + n * spacing, the comb reaches its peaks.
    As the phase grows larger, the comb goes right.
    the third parameter is width，which describes the width of signal.
    the fourth parameter is spacing, which describes the whole length of a wave.
    the fifth parameter is the time duration of the starin time-domain data.
    the sixth parameter is input frequency
    The unit of frequency is Hz.
    '''
    phase = phase * spacing
    omega_p = ((x - phase) / spacing -np.floor((x - phase) / spacing + 1/2) ) *spacing *2 *np.pi
    # here the range of omega_p is [-spacing/2 , spacing/2]*2 *np.pi, the length of omega_p is spacing*2*pi
    # when x = phase ,omega_p = 0
    cut = 0.2
    cutx1 = np.sqrt(1 - cut**2) * width / cut
    cutx2 = 3/duration*2 *np.pi
# here this number*2 means the smallest number for bins used in one cycle.
    cutx3 = spacing*np.pi
    cutx = np.min([np.max([cutx1,cutx2]),cutx3])
# cutx > 2 *pi width/T
    #res = -1j*width /(omega_p-1j*width)
    # res means resonance structure
    #T_dep = 1 - np.exp(-duration*width*(1-1j*omega_p/width))
    # T_dep means the T dependence factor
    y = np.piecewise(omega_p, [omega_p <= - cutx, np.logical_and(-cutx < omega_p, omega_p < cutx),omega_p >= cutx ], 
                   [0, lambda omega_p:width/np.sqrt(width**2+omega_p**2)*np.abs(1 - np.exp(-duration*width*(1-1j*omega_p/width))),0])
    arg = np.piecewise(omega_p, [omega_p <= - cutx, np.logical_and(-cutx < omega_p, omega_p < cutx),omega_p >= cutx ], 
               [np.pi/2, lambda omega_p:np.angle(-1j*width /(omega_p-1j*width)) + np.angle(1 - np.exp(-duration*width*(1-1j*omega_p/width))) ,-np.pi/2])
    #arg = np.angle(res)+np.angle(T_dep)
    qnm_number = np.floor((x - phase) / spacing + 1/2)
    return y * height , arg , qnm_number

def qnmcombmodel_cut(frequency, ** params):
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            params[key] = params[key][:,None]
    '''the params should be a dictionary containing four keys:
    amplitude,phase,width and spacing
    '''

    amplitude = params['amplitude']*1.0
    phase = params['phase']*1.0
    width = params['width']*1.0
    spacing = params['spacing']*1.0
    duration = params['duration']*1.0

    mu = qnmcomb_cut(amplitude,phase,width,spacing,duration,frequency)[0]
    arg = qnmcomb_cut(amplitude,phase,width,spacing,duration,frequency)[1]
    qnm_number = qnmcomb_cut(amplitude,phase,width,spacing,duration,frequency)[2].astype('int64')
    return mu,arg,qnm_number