################## import packages #############################
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
# If you do not understand the lines above, ignore it.

import numpy as np
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
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
from modular_code_part.burstevidence_newcomb import burstevidence2_old_response_notchout as burstevidence_old
from modular_code_part.burstevidence_newcomb import burstevidence2_qnm_response_notchout as burstevidence
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
# nlh = tem
# deltalh = 5
lhi = 0
lhf = 150

# lhi = deltah*nlh
# lhf = deltalh*nlh+deltalh
##See lhlistnew

## 0: old likelihood; 1: new likelihood
# likelihood_index = tem

## 0: create a new csv with header; 1: write in the existing csv
# whether_print_header = tem
# nduration = tem
inject_index = 0

injectlist = ['notch', 'inject']
inject_string = injectlist[inject_index]

likelihoodlist = ['oldlikelihood','newlikelihood']
# likelihood_string = likelihoodlist[likelihood_index]

eventname = 'GW150914'

echoamplitude = 6.4e-20
if inject_index == 0:
    echoamplitude = 0
tlag = 7.4*1e-3
durationlist = np.array([48.6, 36.5, 24.3, 12.1, 4.9]) # benchmarks of time duration in unit of second
signallist = ['time48.npy','time36.npy','time24.npy','time12.npy','time5.npy']
chi = 0.67
f_RD = (1.5251-1.1568*(1-chi)**0.129)/(2*np.pi)
R_bar = 0.00547/(1+1/np.sqrt(1-chi**2))
G = 6.67430e-11
c = 299792458
M_sun = 1.988409870698051e30
finalmass_150914 = 61.5
## Source-frame or detector-frame?
## 61.5 M_sun is the final mass of the binary system, which is the source-frame mass
## M f_RD is dimensionless, to convert to Hz, we need to use:
## G M f_RD /c^3 = 1.5251-1.1568*(1-chi)**0.129


f_RD = f_RD * c**3 / G / M_sun /finalmass_150914
R_bar = R_bar * c**3 / G / M_sun / finalmass_150914
## f_RD,R_bar =  274.50202917591366, 7.693779311787322



gps = 1126259462.4 # roughly the main event time
fnameH = 'H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'
fnameL = 'L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'

## background estimation parameters (use noise prior to the main event in [gps-t0-n*duration,gps-t0])
t0 = 250 # in unit of second
n = 25 # the number of duration


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

notchout_amplitude_list = [6,6,5,5,5] #The normalized strain data in frequency domain higher than this parameter will be outched out.
croptime = 1 # in unit of second

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

rai, dei ,GMSTi = 1.6768742520431272, -1.2148064804269771, 9.383265080483397
psii = 0
phi0i = 0
Aplusi = 1
Acrossi = 0
## dry run setting
# npoints = 10
# nact = 1
# maxmcmc = 100
# walks = 1


# nduration = 1
# for nduration in [0]:
for nduration in [0,1,2,3,4]:
    duration = int(durationlist[nduration]/dt)*dt # time duration in unit of second
    notchout_amplitude = notchout_amplitude_list[nduration]
    Nt = int(duration/dt) # strain data timeseries length
    df=1/duration

    frequency_shift = np.arange(0,int(1/dt),df) - samplingrate/2
    echoraw0 = np.load('../signal/'+signallist[nduration])
    echoraw = echoamplitude*echoraw0
    fre_echo_raw = np.fft.fft(echoraw)
    fre_echo_raw_shift = np.fft.fftshift(fre_echo_raw)

    RHeff = RIeff(rai, dei, psii, Aplusi, Acrossi, phi0i, GMSTi, uH, vH, xH, frequency_shift)
    RLeff = RIeff(rai, dei, psii, Aplusi, Acrossi, phi0i, GMSTi, uL, vL, xL, frequency_shift)

    echorawfre_H = np.fft.fftshift(RHeff * fre_echo_raw_shift + np.conjugate(RHeff * fre_echo_raw_shift)[::-1])/2
    echorawfre_L = np.fft.fftshift(RLeff * fre_echo_raw_shift + np.conjugate(RLeff * fre_echo_raw_shift)[::-1])/2

    # echoraw_H=np.fft.ifft(echorawfre_H)
    echoraw_H = np.fft.irfft(echorawfre_H[0:Nt//2+1])
    echoraw_L = np.fft.irfft(echorawfre_L[0:Nt//2+1])
    if len(echoraw_H) != Nt:
        echoraw_H = np.append(echoraw_H,0)
        echoraw_L = np.append(echoraw_L,0)

    frequency = np.arange(0,int(1/dt/2),df)

    ## directory


    # widthi = 11/duration # teeth width in unit of Hz
    # ampti2 = 5*1e-23/np.sqrt(4*df) # from normalization 5*ASD*sqrt(4*df)
    ################################################ strain data setting ####################################################
    strainLall = TimeSeries.read('../noise/'+fnameL,format='hdf5.gwosc',start=gps-t0-n*duration-duration, end=gps-t0+duration)
    strainHall = TimeSeries.read('../noise/'+fnameH,format='hdf5.gwosc',start=gps-t0-n*duration-duration, end=gps-t0+duration)
    ##前后各增加一段以保证whiten时数据足够
    Lpart=np.split(strainLall,n+2)
    Hpart=np.split(strainHall,n+2)
    Lpart.reverse()
    Hpart.reverse()

    lhlist = np.arange(1,544)#the number 544 can result in the length of lhlistnew equal to 52*10
    lhlist2d = [dimention1to2(lh) for lh in lhlist]
    lhlistnew = np.array([l!=h for l,h in lhlist2d])*lhlist
    lhlistnew = lhlistnew[lhlistnew!=0]



    for lh in lhlistnew[lhi:lhf]:


        indexl,indexh = dimention1to2(lh)

        noiseL = Lpart[indexl]
        noiseH = Hpart[indexh]

        noiseLp = (np.concatenate((Lpart[indexl+1],Lpart[indexl],Lpart[indexl-1])))
        noiseHp = (np.concatenate((Hpart[indexh+1],Hpart[indexh],Hpart[indexh-1])))

        strainL = noiseL + echoraw_L
        strainH = noiseH + echoraw_H
        strainLp = noiseLp + np.concatenate((np.zeros(Nt),echoraw_L,np.zeros(Nt)),axis=0)
        strainHp = noiseHp + np.concatenate((np.zeros(Nt),echoraw_H,np.zeros(Nt)),axis=0)
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

        processor = NotchFilterProcessor(frequency=frequency, samplingrate=samplingrate, df=df, psd_params=psd_params)

        filteredL, strainLpW, psd_strainL, whether_include_in_likelihood_l,  total_notchlist_l , total_croptime_l= processor.process(
            strainLp, Nt, samplingrate, notchout_amplitude, notchout_list_l, notchout_list_errorbar_l, notchout_errorbar_for_psd, notchout_errorbar_for_likelihood, croptime
        )
        filteredH, strainHpW, psd_strainH, whether_include_in_likelihood_h,  total_notchlist_h , total_croptime_h= processor.process(
            strainHp, Nt, samplingrate, notchout_amplitude, notchout_list_h, notchout_list_errorbar_h, notchout_errorbar_for_psd, notchout_errorbar_for_likelihood, croptime
        )
        # total_croptime = np.max([total_croptime_l, total_croptime_h])

        noiseLpW = whiten(processor.apply_notch_filters(noiseLp, total_notchlist_l, total_croptime_l),psd_strainL, dt)
        noiseHpW = whiten(processor.apply_notch_filters(noiseHp, total_notchlist_h, total_croptime_h),psd_strainH ,dt)
        noiseLfre = dt * np.fft.fft(noiseLpW[(Nt-samplingrate*total_croptime_l):(2*Nt-samplingrate*total_croptime_l)])[0:len(frequency)]*np.sqrt(psd_strainL(frequency))
        noiseHfre = dt * np.fft.fft(noiseHpW[(Nt-samplingrate*total_croptime_h):(2*Nt-samplingrate*total_croptime_h)])[0:len(frequency)]*np.sqrt(psd_strainH(frequency))
        noiseLfre = noiseLfre.value
        noiseHfre = noiseHfre.value

        strainLfre = dt * np.fft.fft(strainLpW[(Nt-samplingrate*total_croptime_l):(2*Nt-samplingrate*total_croptime_l)])[0:len(frequency)]*np.sqrt(psd_strainL(frequency))
        strainHfre = dt * np.fft.fft(strainHpW[(Nt-samplingrate*total_croptime_h):(2*Nt-samplingrate*total_croptime_h)])[0:len(frequency)]*np.sqrt(psd_strainH(frequency))

        strainLfre_normalized_abs = np.abs(strainLfre/(np.sqrt(psd_strainL(frequency)/(4*df))))
        strainHfre_normalized_abs = np.abs(strainHfre/(np.sqrt(psd_strainH(frequency)/(4*df))))
        strainLfre = strainLfre.value
        strainHfre = strainHfre.value


        amplitude_min = (np.mean(np.abs(strainLfre_origin))+np.mean(np.abs(strainHfre_origin)))/2/100
        amplitude_min = amplitude_min.value
        # amplitude_max = (np.mean(np.abs(strainLfre_origin))+np.mean(np.abs(strainHfre_origin)))/2*10
        amplitude_max = amplitude_min *1000

        for likelihood_index in [0,1]:
            likelihood_string = likelihoodlist[likelihood_index]
            label0 = eventname+'_ndata={0:1d}'.format(n)+'_npoint={0:1d}'.format(npoints)+'_duration={:.0f}'.format(duration)+'_'+likelihood_string
            outdirtemp = 'outdir_'+inject_string+'_GW150914noise_version3.5_psd0.5_rwalk'
            outdir = '../'+outdirtemp+'/'+label0
            label = label0+'lh'+str(lh)

            result = bilby.result.read_in_result(outdir=outdir, label=label)


            posterior = result.posterior
            posterior_SNR = []
            for j in np.arange(len(posterior)):
                inject_params = posterior.iloc[j]
                fmin = inject_params['fmin']
                fmax = inject_params['fmax']
                frequencye = frequency[int(fmin/df):int(fmax/df+2)]
                res_amplitude = inject_params['res_amplitude']
                hi = combmodel(frequencye,**inject_params)[0]

                posterior_SNR.append(np.sqrt(  4*df * np.sum(hi**2/psd_strainH_origin(frequencye))
                                            +  4*df * np.sum(res_amplitude * hi**2/psd_strainL_origin(frequencye)) ))
            posterior_SNR = np.array(posterior_SNR)
            result.posterior['SNR'] = posterior_SNR
            result.outdir = outdir
            result.label = label+'_SNR'
            result.save_to_file()
            print(likelihood_string,duration,lh)
