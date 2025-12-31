################## import packages #############################
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
# If you do not understand the lines above, ignore it.

import numpy as np
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import bilby
import csv
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_event_urls
from gwpy.signal import filter_design
import os
from bilby.core.prior import PriorDict


import sys

sys.path.append("..")
from modular_code_part.burstevidence_newcomb import burstevidence2_old_response as burstevidence_old
from modular_code_part.burstevidence_newcomb import burstevidence2_qnm_response as burstevidence
from modular_code_part.comb_models_version3 import qnmcombmodel_cut as combmodel
from modular_code_part.notch import NotchFilterProcessor as NotchFilterProcessor
from modular_code_part.detresponse import RIeff as RIeff

################################################# define functions #########################################################
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def dimention2to1(i,j):
    temp = max(i-1,j-1)
    return temp**2+i+(temp-j)+1
    ## 从i，j转换到ij，注意第一项编号是1
    ## transform ij to i,j, note that the first item is numbered 1
    
def dimention1to2(ij_input):
    ij = ij_input-1
    temp = int(np.sqrt(ij)) ##表示该组中最大的数字
    #part = temp*2+1 ##表示该组共有多少个元素
    #n = ij - temp**2 +1 ##表示处于该组第几个位置
    ## temp represents the largest number in the group
    ## part = temp*2+1 represents the total number of elements in the group
    ## n = ij - temp**2 +1 represents the position in the group
    minus = ij - temp**2 - temp  ##j-i = n - temp-1
    if minus==0:
        j = temp
        i = temp
    elif minus>0:
        i = temp
        j = temp-minus
    else:
        j =temp
        i = temp+minus
    return (i+1,j+1)
################################################# define functions end #####################################################

################################################# parameter setting #########################################################


lhi = tem
lhf = tem
# nlh = tem
# deltalh = 2
# lhi = deltalh*nlh
# lhf = deltalh*nlh+deltalh
##See lhlistnew

## 0: old likelihood; 1: new likelihood
likelihood_index = tem

whether_print_header = tem
## 0: create a new csv with header; 1: write in the existing csv
# whether_print_header = tem
nduration = tem
notch_index = tem

notchlist = ['nonotch', 'notch']
notch_string = notchlist[notch_index]

likelihoodlist = ['oldlikelihood','newlikelihood']
likelihood_string = likelihoodlist[likelihood_index]

chi = 0.67
f_RD = (1.5251-1.1568*(1-chi)**0.129)/(2*np.pi)
R_bar = 0.00547/(1+1/np.sqrt(1-chi**2))
G = 6.67430e-11
c = 299792458
M_sun = 1.988409870698051e30

eventname = tem
if eventname == 'GW150914':
    tlag = 6.9*1e-3
    finalmass_150914 = 61.5 * 1.1
    f_RD = f_RD * c**3 / G / M_sun /finalmass_150914
    R_bar = R_bar * c**3 / G / M_sun / finalmass_150914
    ## f_RD,R_bar = 249.54729925083058, 6.994344828897565
    ## 61.5 M_sun is the final mass of the binary system, which is the source-frame mass
    ## M f_RD is dimensionless, to convert to Hz, we need to use:
    ## G M f_RD /c^3 = 1.5251-1.1568*(1-chi)**0.129
    durationlist = np.array([114.4, 57.2]) # 150914 benchmarks of time duration in unit of second 
    # The two durations refer to 1 / (R_bar/2) * 400 and 1 / (R_bar/2) * 200 respectively.
    gps = 1126259462.4 # roughly the main event time
    fnameH = 'H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'
    fnameL = 'L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'
    ## background estimation parameters (use noise prior to the main event in [gps-t0-n*duration,gps-t0])
    t0 = 250 # in unit of second
    n = 25 # the number of duration
    if nduration == 0:
        n = 14# the number of duration
        lhlist = np.arange(1,164)#the number 164 can result in the length of lhlistnew equal to 150
        lhlist2d = [dimention1to2(lh) for lh in lhlist]
        lhlistnew = np.array([l!=h for l,h in lhlist2d])*lhlist
        lhlistnew = lhlistnew[lhlistnew!=0]
    if nduration == 1:
        n = 25 # the number of duration
        lhlist = np.arange(1,524) # The number 524 results in the length of lhlistnew equal to 500
        lhlist2d = [dimention1to2(lh) for lh in lhlist]
        lhlistnew = np.array([l!=h for l,h in lhlist2d])*lhlist
        # Here we exclude the cases with l=h because they are correnlated. 
        lhlistnew = lhlistnew[lhlistnew!=0]
    
    notchout_list_l_large = np.array([60,2*60,3*60,4*60]) #The frequency center for notching out
    notchout_list_errorbar_l_large = np.array([0.06,2*0.06,3*0.06,4*0.06])#The frequency error bar for notching out
    notchout_list_h_large = np.array([60,2*60,3*60,4*60] )#The frequency center for notching out
    notchout_list_errorbar_h_large =np.array([0.06,2*0.06,3*0.06,4*0.06]) #The frequency error bar for notching out
    notchout_list_l_small = np.array([73.9,122.2,147.8,170.5,196.1,244.4,267.1]) #The frequency center for notching out
    notchout_list_errorbar_l_small = np.array([0.03,0.03,0.03,0.03,0.03,0.03,0.03])#The frequency error bar for notching out
    notchout_list_h_small = np.array([64,128,160,192,256,272])#The frequency center for notching out
    notchout_list_errorbar_h_small =np.array([0.03,0.03,0.03,0.03,0.03,0.03]) #The frequency error bar for notching out

    notchout_list_l = np.concatenate((notchout_list_l_large, notchout_list_l_small), axis = 0)
    notchout_list_h = np.concatenate((notchout_list_h_large, notchout_list_h_small), axis = 0)
    notchout_list_errorbar_l = np.concatenate((notchout_list_errorbar_l_large, notchout_list_errorbar_l_small), axis = 0)
    notchout_list_errorbar_h = np.concatenate((notchout_list_errorbar_h_large, notchout_list_errorbar_h_small), axis = 0)

    notchout_errorbar_for_likelihood = [-0.3,0.3]
    notchout_errorbar_for_psd = [-2,2]

    notchout_amplitude_list = [6,6] #The normalized strain data in frequency domain higher than this parameter will be outched out.
    croptime = 1 # in unit of second

if eventname == 'GW231226':
    tlag = -4*1e-3
    finalmass_231226 = 71.4 * 1.2
    f_RD = f_RD * c**3 / G / M_sun /finalmass_231226
    R_bar = R_bar * c**3 / G / M_sun / finalmass_231226
    ## f_RD,R_bar = 197.03401954153466, 5.522495654469191
    durationlist = np.array([144.9, 72.4]) # 231226 benchmarks of time duration in unit of second
    gps = 1387620938.3
    fnameH = 'H-H1_GWOSC_O4a_4KHZ_R1-1387618304-4096.hdf5'
    fnameL = 'L-L1_GWOSC_O4a_4KHZ_R1-1387618304-4096.hdf5'
    t0 = 100 # in unit of second
    if nduration == 0:
        n = 15 # the number of duration
        lhlist = np.arange(1,176) # The number 176 results in the length of lhlistnew equal to 150
        lhlist2d = [dimention1to2(lh) for lh in lhlist]
        lhlistnew = np.array([l!=h for l,h in lhlist2d])*np.array([h!=2 for l,h in lhlist2d])*lhlist
        # Here we exclude the cases with l=h because they are correnlated. 
        # We also exclude the cases with h=4 because a glitch occurs in the Hanford detector around this time. 
        lhlistnew = lhlistnew[lhlistnew!=0]
    if nduration == 1:
        n = 25 # the number of duration
        
        lhlist = np.arange(1,546) # The number 546 results in the length of lhlistnew equal to 500
        lhlist2d = [dimention1to2(lh) for lh in lhlist]
        lhlistnew = np.array([l!=h for l,h in lhlist2d])*np.array([h!=4 for l,h in lhlist2d])*lhlist
        # Here we exclude the cases with l=h because they are correnlated. 
        # We also exclude the cases with h=4 because a glitch occurs in the Hanford detector around this time. 
        lhlistnew = lhlistnew[lhlistnew!=0]
    
    
    notchout_list_l_large = np.array([60,2*60,3*60,4*60]) #The frequency center for notching out
    notchout_list_errorbar_l_large = np.array([0.06,2*0.06,3*0.06,4*0.06])#The frequency error bar for notching out
    notchout_list_h_large = np.array([60,2*60,3*60,4*60] )#The frequency center for notching out
    notchout_list_errorbar_h_large =np.array([0.06,2*0.06,3*0.06,4*0.06]) #The frequency error bar for notching out
    notchout_list_l_small = np.array([]) #The frequency center for notching out
    notchout_list_errorbar_l_small = np.array([])#The frequency error bar for notching out
    notchout_list_h_small = np.array([])#The frequency center for notching out
    notchout_list_errorbar_h_small =np.array([]) #The frequency error bar for notching out

    notchout_list_l = np.concatenate((notchout_list_l_large, notchout_list_l_small), axis = 0)
    notchout_list_h = np.concatenate((notchout_list_h_large, notchout_list_h_small), axis = 0)
    notchout_list_errorbar_l = np.concatenate((notchout_list_errorbar_l_large, notchout_list_errorbar_l_small), axis = 0)
    notchout_list_errorbar_h = np.concatenate((notchout_list_errorbar_h_large, notchout_list_errorbar_h_small), axis = 0)

    notchout_errorbar_for_likelihood = [-0.3,0.3]
    notchout_errorbar_for_psd = [-2,2]

    notchout_amplitude_list = [6,6] #The normalized strain data in frequency domain higher than this parameter will be outched out.
    croptime = 1 # in unit of second
    
    

resampti = 1
resdti = tlag
resph01 = -0.5*np.pi
resph02 = 1.5*np.pi
f_minimal = 50

## psd setting
samplingrate = 4096 
dt = 1/samplingrate # time resolution in unit of second
NFFT = int(0.5 / dt) # the segment length is 0.5s
fs = int(1/ dt)
psd_window = np.blackman(NFFT)
NOVL = int(NFFT/2)
psd_params = {
    'Fs': fs,
    'NFFT': NFFT,
    'window': psd_window,
    'noverlap': NOVL
    }

##notch out setting


## sampler setting
npoints = 1000
# nact = 10# up to now, we do not use this parameter yet
maxmcmc = 10000
walks = 100

# LVK arm unit vectors
uL=np.array([-0.95457412153, -0.14158077340, -0.26218911324])
vL=np.array([0.29774156894, -0.48791033647, -0.82054461286])
uH=np.array([-0.22389266154, 0.79983062746, 0.55690487831])
vH=np.array([-0.91397818574, 0.02609403989, -0.40492342125])
uV=np.array([-0.70045821479,0.20848948619,0.68256166277])
vV=np.array([-0.05379255368,-0.96908180549,0.24080451708])

xH=np.array([-2.16141492636*1e6, -3.83469517889*1e6, 4.6003502264*1e6])
xL=np.array([-7.42760447238*1e4, -5.49628371971*1e6, 3.22425701744*1e6])
xV=np.array([4.54637409900*1e6,8.42989697626*1e5,4.37857696241*1e6])

## dry run setting
# npoints = 10
# nact = 1
# maxmcmc = 100
# walks = 1


# nduration = 1
duration = int(durationlist[nduration]/dt)*dt # time duration in unit of second
notchout_amplitude = notchout_amplitude_list[nduration]
Nt = int(duration/dt) # strain data timeseries length
df=1/duration
frequency = np.arange(0,int(1/dt/2),df)

## directory
label0 = eventname+'_ndata={0:1d}'.format(n)+'_npoint={0:1d}'.format(npoints)+'_duration={:.0f}'.format(duration)+'_'+likelihood_string
outdirtemp = 'outdir_'+notch_string+'_'+eventname+'noise_version3.5_psd0.5_rwalk'
outdir = '../'+outdirtemp+'/'+label0

if(whether_print_header == 0):
    headers = ['duration',  'indexlh', 'indexl', 'indexh', 'logB', 'maxloglikelihood', 'SNR_comb_median', 'SNR_comb_global', 'run_time', 
        'total_croptimel', 'total_croptimeh','notchlist_l', 'notchlist_h',
        'width_median', 'width_plus', 'width_minus', 'width_global',
        'amplitude_median', 'amplitude_plus', 'amplitude_minus', 'amplitude_global',
        'phase_median', 'phase_plus', 'phase_minus', 'phase_global',
        'spacing_median', 'spacing_plus', 'spacing_minus', 'spacing_global',
        'fmin_median', 'fmin_plus', 'fmin_minus', 'fmin_global',
        'fmax_median', 'fmax_plus', 'fmax_minus', 'fmax_global',
        'res_phase0_median', 'res_phase0_plus', 'res_phase0_minus', 'res_phase0_global',
        'res_amplitude_median', 'res_dt_median','duration_median']
    if notch_index == 0:#nonotch
        headers = ['duration',  'indexlh', 'indexl', 'indexh', 'logB', 'maxloglikelihood', 'SNR_comb_median', 'SNR_comb_global', 'run_time', 
            # 'total_croptimel', 'total_croptimeh','notchlist_l', 'notchlist_h',
            'width_median', 'width_plus', 'width_minus', 'width_global',
            'amplitude_median', 'amplitude_plus', 'amplitude_minus', 'amplitude_global',
            'phase_median', 'phase_plus', 'phase_minus', 'phase_global',
            'spacing_median', 'spacing_plus', 'spacing_minus', 'spacing_global',
            'fmin_median', 'fmin_plus', 'fmin_minus', 'fmin_global',
            'fmax_median', 'fmax_plus', 'fmax_minus', 'fmax_global',
            'res_phase0_median', 'res_phase0_plus', 'res_phase0_minus', 'res_phase0_global',
            'res_amplitude_median', 'res_dt_median','duration_median']
    if not os.path.exists('../'+outdirtemp+'/'):
        os.mkdir('../'+outdirtemp+'/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(outdir+'/../'+label0+'_all.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
    whether_print_header = 1
################################################ parameter setting end #######################################################

strainLall = TimeSeries.read('../noise/'+fnameL,format='hdf5.gwosc',start=gps-t0-n*duration-duration, end=gps-t0+duration)
strainHall = TimeSeries.read('../noise/'+fnameH,format='hdf5.gwosc',start=gps-t0-n*duration-duration, end=gps-t0+duration)
##前后各增加一段以保证whiten时数据足够
Lpart=np.split(strainLall,n+2)
Hpart=np.split(strainHall,n+2)
Lpart.reverse()
Hpart.reverse()

for lh in lhlistnew[lhi:lhf]:
    label = label0+'lh'+str(lh)
    indexl,indexh = dimention1to2(lh)

    noiseL = Lpart[indexl]
    noiseH = Hpart[indexh]

    noiseLp = (np.concatenate((Lpart[indexl+1],Lpart[indexl],Lpart[indexl-1])))
    noiseHp = (np.concatenate((Hpart[indexh+1],Hpart[indexh],Hpart[indexh-1])))

    strainL = noiseL 
    strainH = noiseH
    strainLp = noiseLp 
    strainHp = noiseHp 
    ################################################ generate origin data ##################################################
    Pxx_strainL_origin, freqs = mlab.psd(strainL, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    psd_strainL_origin = interp1d(freqs, Pxx_strainL_origin)
    Pxx_strainH_origin, freqs = mlab.psd(strainH, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    psd_strainH_origin = interp1d(freqs, Pxx_strainH_origin)

    noiseLpW_origin = whiten(noiseLp, psd_strainL_origin, dt)
    noiseHpW_origin = whiten(noiseHp, psd_strainH_origin, dt)
    noisefreL_origin = dt * np.fft.fft(noiseLpW_origin[Nt:(2*Nt)])[0:len(frequency)]*np.sqrt(psd_strainL_origin(frequency))
    noisefreH_origin = dt * np.fft.fft(noiseHpW_origin[Nt:(2*Nt)])[0:len(frequency)]*np.sqrt(psd_strainH_origin(frequency))

    strainLpW_origin = whiten(strainLp, psd_strainL_origin, dt)
    strainHpW_origin = whiten(strainHp, psd_strainH_origin, dt)
    strainLfre_origin = dt * np.fft.fft(strainLpW_origin[Nt:(2*Nt)])[0:len(frequency)]*np.sqrt(psd_strainL_origin(frequency))
    strainHfre_origin = dt * np.fft.fft(strainHpW_origin[Nt:(2*Nt)])[0:len(frequency)]*np.sqrt(psd_strainH_origin(frequency))
    strainLfre_normalized_abs_origin = np.abs(strainLfre_origin/(np.sqrt(psd_strainL_origin(frequency)/(4*df))))
    strainHfre_normalized_abs_origin = np.abs(strainHfre_origin/(np.sqrt(psd_strainH_origin(frequency)/(4*df))))
    if notch_index == 1:#notch
        processor = NotchFilterProcessor(frequency=frequency, samplingrate=samplingrate, df=df, psd_params=psd_params)

        filteredL, strainLpW, psd_strainL, whether_include_in_likelihood_l,  total_notchlist_l , total_croptime_l= processor.process(
            strainLp, Nt, samplingrate, notchout_amplitude, notchout_list_l, notchout_list_errorbar_l, notchout_errorbar_for_psd, notchout_errorbar_for_likelihood, croptime
        )
        filteredH, strainHpW, psd_strainH, whether_include_in_likelihood_h,  total_notchlist_h , total_croptime_h= processor.process(
            strainHp, Nt, samplingrate, notchout_amplitude, notchout_list_h, notchout_list_errorbar_h, notchout_errorbar_for_psd, notchout_errorbar_for_likelihood, croptime
        )
        total_croptime = np.max([total_croptime_l, total_croptime_h])
        
        noiseLpW = whiten(processor.apply_notch_filters(noiseLp, total_notchlist_l, total_croptime),psd_strainL, dt)
        noiseHpW = whiten(processor.apply_notch_filters(noiseHp, total_notchlist_h, total_croptime),psd_strainH ,dt)
        noiseLfre = dt * np.fft.fft(noiseLpW[(Nt-samplingrate*total_croptime):(2*Nt-samplingrate*total_croptime)])[0:len(frequency)]*np.sqrt(psd_strainL(frequency))
        noiseHfre = dt * np.fft.fft(noiseHpW[(Nt-samplingrate*total_croptime):(2*Nt-samplingrate*total_croptime)])[0:len(frequency)]*np.sqrt(psd_strainH(frequency))
        noiseLfre = noiseLfre.value
        noiseHfre = noiseHfre.value

        strainLfre = dt * np.fft.fft(strainLpW[(Nt-samplingrate*total_croptime):(2*Nt-samplingrate*total_croptime)])[0:len(frequency)]*np.sqrt(psd_strainL(frequency))
        strainHfre = dt * np.fft.fft(strainHpW[(Nt-samplingrate*total_croptime):(2*Nt-samplingrate*total_croptime)])[0:len(frequency)]*np.sqrt(psd_strainH(frequency)) 
        
        strainLfre_normalized_abs = np.abs(strainLfre/(np.sqrt(psd_strainL(frequency)/(4*df))))
        strainHfre_normalized_abs = np.abs(strainHfre/(np.sqrt(psd_strainH(frequency)/(4*df))))
        strainLfre = strainLfre.value
        strainHfre = strainHfre.value
        
    elif notch_index == 0:#nonotch
        strainLpW = strainLpW_origin
        strainHpW = strainHpW_origin
        psd_strainL = psd_strainL_origin
        psd_strainH = psd_strainH_origin
        
        noiseLpW = noiseLpW_origin
        noiseHpW = noiseHpW_origin
        noiseLfre = noisefreL_origin.value
        noiseHfre = noisefreH_origin.value
        
        strainLfre = strainLfre_origin.value
        strainHfre = strainHfre_origin.value
        
        strainLfre_normalized_abs = strainLfre_normalized_abs_origin
        strainHfre_normalized_abs = strainHfre_normalized_abs_origin
        whether_include_in_likelihood_h = 1
        whether_include_in_likelihood_l = 1


    amplitude_min = (np.mean(np.abs(strainLfre_origin[int(f_minimal/df):int(1.1 * f_RD/df+2)]))+np.mean(np.abs(strainHfre_origin[int(f_minimal/df):int(1.1 * f_RD/df+2)])))/2/100
    # select the frequency range[f_minimal,1.1 * f_RD] for calculating amplitude_min
    amplitude_min = amplitude_min.value
    # amplitude_max = (np.mean(np.abs(strainLfre_origin))+np.mean(np.abs(strainHfre_origin)))/2*10
    amplitude_max = amplitude_min *1000
    def frange(parameters):
        converted_parameters = parameters.copy()
        converted_parameters['z'] = parameters['fmax'] - parameters['fmin'] - parameters['spacing']*10
        return converted_parameters
    priors=PriorDict(conversion_function=frange)
    priors['width']=bilby.core.prior.LogUniform(1/duration, R_bar, 'fw')
    priors['amplitude']=bilby.core.prior.Uniform(amplitude_min, amplitude_max, 'amplitude')
    # In our paper, we use the notation \(\langle \tilde{P} \rangle\) to represent `np.mean(psd_strain(frequency))`, although in the code we actually employ `np.mean(np.abs(noisefre))`. 
    # These two expressions are numerically similar, with `np.mean(np.abs(noisefre))` being approximately 1.2 times the value of `np.mean(psd_strain(frequency))`."

    priors['phase']=bilby.core.prior.Uniform(0, 1, 'phase')
    priors['spacing']=bilby.core.prior.Uniform(R_bar/4, R_bar, 'spacing')
    priors['z'] = bilby.core.prior.Constraint(minimum=0, maximum=f_RD)
    priors['fmin'] = bilby.core.prior.Uniform(f_minimal, 1.1 * f_RD, 'fmin')
    priors['fmax'] = bilby.core.prior.Uniform(f_minimal, 1.1 * f_RD, 'fmax')
    priors['duration'] = duration
    priors['res_amplitude'] = resampti
    priors['res_phase0'] = bilby.core.prior.Uniform(resph01, resph02, 'res_phase0')
    priors['res_dt'] = resdti
    if likelihood_index == 0 :
        likelihood = burstevidence_old(x=frequency, y1=strainHfre, y2=strainLfre,
                                    sn1=psd_strainH(frequency), sn2=psd_strainL(frequency),
                                    function = combmodel, df = df )
    if likelihood_index == 1 :
        likelihood = burstevidence(x=frequency, y1=np.abs(strainHfre), y2=np.abs(strainLfre),
                                angle1=np.angle(strainHfre), angle2=np.angle(strainLfre),
                                sn1=psd_strainH(frequency), sn2=psd_strainL(frequency),
                                function = combmodel, df = df )
        
    # if os.path.exists(outdir+'/'+label+'_result.json' ):
    #     result = bilby.result.read_in_result(outdir=outdir, label=label)
    #     sampling_time = result.sampling_time.total_seconds()
    # # If the result file exists, we read the result file. Otherwise, we run the sampler.
    # else:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', sample="rwalk", #bound="multi",nact=nact,
    npoints=npoints,  maxmcmc=maxmcmc, walks=walks, outdir=outdir, label=label)
    sampling_time = result.sampling_time
    result.plot_corner()
    logb = result.log_evidence
    maxloglikelihood = result.posterior.max()['log_likelihood']
    #The suffix 'm' means 'median'; the suffix 'g' means 'global'; the suffix 'p' means 'plus'; the suffix 'n' means 'minus'
    #'global' related to the parameter which gives the maximum likelihood
    #'median','plus','minus' related to the median parameter from Bayes posterior
    posterior_params_m=dict()
    for key in result.search_parameter_keys+result.fixed_parameter_keys:
        posterior_params_m[key]=result.get_one_dimensional_median_and_error_bar(key).median
    posterior_params_p=dict()
    for key in result.search_parameter_keys:
        posterior_params_p[key]=result.get_one_dimensional_median_and_error_bar(key).plus
    posterior_params_n=dict()
    for key in result.search_parameter_keys:
        posterior_params_n[key]=result.get_one_dimensional_median_and_error_bar(key).minus
    posterior_params_g=dict()
    posterior_params_g = result.posterior.loc[[result.posterior.idxmax()['log_likelihood']]].to_dict(orient ='records')[0]


    fmin_m=posterior_params_m['fmin']
    fmax_m=posterior_params_m['fmax']
    res_amplitude_m=posterior_params_m['res_amplitude']
    res_phase0_m=posterior_params_m['res_phase0']
    res_dt_m=posterior_params_m['res_dt']

    fmin_g=posterior_params_g['fmin']
    fmax_g=posterior_params_g['fmax']
    res_amplitude_g=posterior_params_g['res_amplitude']
    res_phase0_g=posterior_params_g['res_phase0']
    res_dt_g=posterior_params_m['res_dt']

    frequencye_m= frequency[int(fmin_m/df):int(fmax_m/df+2)]
    frequencye_g= frequency[int(fmin_g/df):int(fmax_g/df+2)]
    # echo_fre_H = dt*np.fft.fft(echoraw_H)[0:len(frequency)]
    # echo_fre_L = dt*np.fft.fft(echoraw_L)[0:len(frequency)]

    psd_strainLH_list_origin=psd_strainL_origin(frequency)*psd_strainH_origin(frequency)/(psd_strainL_origin(frequency)+res_amplitude_m**2*psd_strainH_origin(frequency)) 
    psd_strainLH_list=psd_strainL(frequency)*psd_strainH(frequency)/(psd_strainL(frequency)+res_amplitude_m**2*psd_strainH(frequency))                       
    strainfreN_m_origin = strainHfre_origin/psd_strainH_origin(frequency)+res_amplitude_m*np.exp(res_phase0_m*1j)*np.exp(2*1j*np.pi*res_dt_m*frequency)*strainLfre_origin/psd_strainL_origin(frequency)
    strainfreN_g_origin = strainHfre_origin/psd_strainH_origin(frequency)+res_amplitude_g*np.exp(res_phase0_g*1j)*np.exp(2*1j*np.pi*res_dt_g*frequency)*strainLfre_origin/psd_strainL_origin(frequency)

    strainfreN_m = whether_include_in_likelihood_h*strainHfre/psd_strainH(frequency)+res_amplitude_m*whether_include_in_likelihood_l*np.exp(res_phase0_m*1j)*np.exp(2*1j*np.pi*res_dt_m*frequency)*strainLfre/psd_strainL(frequency)
    strainfreN_g = whether_include_in_likelihood_h*strainHfre/psd_strainH(frequency)+res_amplitude_g*whether_include_in_likelihood_l*np.exp(res_phase0_g*1j)*np.exp(2*1j*np.pi*res_dt_g*frequency)*strainLfre/psd_strainL(frequency)

    hi_m = combmodel(frequencye_m, ** posterior_params_m)[0]* np.exp(1j*combmodel(frequencye_m, ** posterior_params_m)[1])
    rho_optcomb_m = np.sqrt(4*df * np.sum(np.abs(hi_m)**2/psd_strainH_origin(frequencye_m)) +  4*df * np.sum((res_amplitude_m*np.abs(hi_m))**2/psd_strainL_origin(frequencye_m)))   
    hi_g = combmodel(frequencye_g, ** posterior_params_g)[0]* np.exp(1j*combmodel(frequencye_g, ** posterior_params_g)[1])
    rho_optcomb_g = np.sqrt(4*df * np.sum(np.abs(hi_g)**2/psd_strainH_origin(frequencye_g)) +  4*df * np.sum((res_amplitude_g*np.abs(hi_g))**2/psd_strainL_origin(frequencye_g)))


    # rho_inj_m = np.sqrt(4*df * np.sum(np.abs(echo_fre_H[int(fmin_m/df):int(fmax_m/df+2)])**2/psd_strainH_origin(frequencye_m)) +  4*df * np.sum((res_amplitude_m*np.abs(echo_fre_L[int(fmin_m/df):int(fmax_m/df+2)]))**2/psd_strainL_origin(frequencye_m)))
    # rho_inj_g = np.sqrt(4*df * np.sum(np.abs(echo_fre_H[int(fmin_g/df):int(fmax_g/df+2)])**2/psd_strainH_origin(frequencye_g)) +  4*df * np.sum((res_amplitude_g*np.abs(echo_fre_L[int(fmin_g/df):int(fmax_g/df+2)]))**2/psd_strainL_origin(frequencye_g)))
    # These lines are used to calculate the SNR of the injected signal and the SNR of the best-fit comb.
    ######################################################  csv data end  ##############################################
    plotlabel = label
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.plot(frequency,np.abs(strainfreN_m_origin)*np.sqrt(psd_strainLH_list_origin*(4*df)),label='normalized abs(dH/sH+Ares*dL/sL)',color='lightgray')
    plt.plot(frequency,np.abs(strainfreN_m*np.sqrt(psd_strainLH_list*(4*df))),label='notched normalized abs(dH/sH+Ares*dL/sL)',color='b')
    # plt.plot(frequency,np.abs(echo_fre_H/psd_strainH(frequency)+echo_fre_L/psd_strainL(frequency))*np.sqrt(psd_strainLH_list*(4*df)),label='normalized injected echoes',color='g')
    # plt.plot(frequencye_m, hi_m*np.sqrt((4*df)/psd_strainH_origin(frequencye_m)+res_amplitude_m**2*(4*df)/psd_strainL_origin(frequencye_m)),label='normalized best-fit comb')
    plt.plot(frequencye_m, np.abs(hi_m*np.sqrt((4*df)/psd_strainH(frequencye_m)+res_amplitude_m**2*(4*df)/psd_strainL(frequencye_m))),label='normalized best-fit comb',color='r',linestyle='-.')
    plt.axis([fmin_m, fmax_m, 0, 8])
    plt.ylabel(''),plt.xlabel('f (Hz)')
    plt.legend(loc='upper right')
    plt.title('Posterior median: comb SNR = {0:.2f}'.format(rho_optcomb_m)+', logB = {0:.2f}'.format(result.log_evidence))
    plt.subplot(2,1,2)
    plt.plot(frequency,np.abs(strainfreN_g_origin)*np.sqrt(psd_strainLH_list_origin*(4*df)),label='normalized abs(dH/sH+Ares*dL/sL)',color='lightgray')
    plt.plot(frequency,np.abs(strainfreN_g*np.sqrt(psd_strainLH_list*(4*df))),label='notched normalized abs(dH/sH+Ares*dL/sL)',color='b')
    # plt.plot(frequency,np.abs(echo_fre_H/psd_strainH(frequency)+echo_fre_L/psd_strainL(frequency))*np.sqrt(psd_strainLH_list*(4*df)),label='normalized injected echoes',color='g')
    # plt.plot(frequencye_g, hi_g*np.sqrt((4*df)/psd_strainH_origin(frequencye_g)+res_amplitude_g**2*(4*df)/psd_strainL_origin(frequencye_g)),label='normalized best-fit comb')
    plt.plot(frequencye_g, np.abs(hi_g*np.sqrt((4*df)/psd_strainH(frequencye_g)+res_amplitude_g**2*(4*df)/psd_strainL(frequencye_g))),label='normalized best-fit comb',color='r',linestyle='-.')

    plt.axis([fmin_g, fmax_g, 0, 8])
    plt.ylabel(''),plt.xlabel('f (Hz)')
    plt.legend(loc='upper right')
    plt.title('Global maximum, comb SNR = {0:.2f}'.format(rho_optcomb_g))
    plt.savefig(outdir+'/'+label+''+'_bestfit_comb.png')
    plt.close()

    #############################################################  ploting end #####################################################################

    if notch_index == 1:#notch
        rows = [duration, lh, indexl, indexh, logb, maxloglikelihood,rho_optcomb_m,rho_optcomb_g, sampling_time, total_croptime_l, total_croptime_h ,total_notchlist_l, total_notchlist_h]
    elif notch_index == 0:#nonotch
        rows = [duration, lh, indexl, indexh, logb, maxloglikelihood,rho_optcomb_m,rho_optcomb_g, sampling_time]
    # for key in result.search_parameter_keys:
    #     rows = np.append(rows,[posterior_params_m[key],posterior_params_p[key],posterior_params_n[key],posterior_params_g[key]])
    # for key in result.fixed_parameter_keys:
    #     rows = np.append(rows,[posterior_params_m[key]])
    for key in result.search_parameter_keys:
        rows.append(posterior_params_m[key])
        rows.append(posterior_params_p[key])
        rows.append(posterior_params_n[key])
        rows.append(posterior_params_g[key])
    for key in result.fixed_parameter_keys:
        rows.append(posterior_params_m[key])
    with open(outdir+'/../'+label0+'_all.csv', 'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(rows)
