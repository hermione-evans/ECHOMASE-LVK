import bilby
import numpy as np
from scipy.special import ive as Ive
from scipy.special import iv as Iv
import inspect
def logi0(x):
    return np.log(Ive(0,x))+np.abs(x)

class burstevidence_old(bilby.Likelihood):
    """
    Likelihood for one detector with no detector response
    """

    def __init__(self, x, y, sn, function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y: array-like
            The strain data of frequency domain
        sn: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        """
        super().__init__(parameters={'fmin': None,'fmax': None})

        self.df = df
        self.x = x
        self.y = y
        self.sn = sn
        ## maybe later sn will be replaced by an interpolating function
        self.function = function
        self.N = len(x)



        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sne = self.sn[int(fmin/self.df):int(fmax/self.df+2)]
        ye = self.y[int(fmin/self.df):int(fmax/self.df+2)]
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        hi = self.function(xe,  **model_parameters)[0]
        burst = np.sum(logi0(4*self.df*np.abs(ye)*np.abs(hi)/sne)-2*self.df*np.abs(hi)**2/sne)
        return burst

class burstevidence2_old_response(bilby.Likelihood):
    """
    Likelihood for two detectors with correlation and detector response
    """

    def __init__(self, x, y1, y2, sn1, sn2, function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y1,2: array-like
            The strain data of frequency domain
        sn1,2: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        fmin & fmax: float
            describes the minimum and maximum of frequency array
        res_amplitude： float
            describes the amplitude difference of the two detector responses
        res_phase0： float
            describes the overall phase difference of the two detector responses
        res_dt： float
            describes the arrival time delay between two detecotrs, determining the frequency dependent phase difference of the two detector responses
        """
        super().__init__(parameters={'fmin': None,'fmax': None,'res_amplitude': None,'res_phase0': None,'res_dt': None})
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.sn1 = sn1
        self.sn2 = sn2
        self.function = function
        self.N = len(x)
        self.df = df


        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        res_amplitude = self.parameters['res_amplitude']
        res_phase0 = self.parameters['res_phase0']
        res_dt = self.parameters['res_dt']


        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sn1e = self.sn1[int(fmin/self.df):int(fmax/self.df+2)]
        y1e = self.y1[int(fmin/self.df):int(fmax/self.df+2)]
        sn2e = self.sn2[int(fmin/self.df):int(fmax/self.df+2)]
        y2e = self.y2[int(fmin/self.df):int(fmax/self.df+2)]
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        hi = self.function(xe,  **model_parameters)[0]

        y = y1e/sn1e + res_amplitude * y2e/sn2e *np.exp(res_phase0*1j)*np.exp(-2*1j*np.pi*res_dt*xe)

        burst = np.sum(logi0(4*self.df*np.abs(y)*np.abs(hi))-2*self.df*np.abs(hi)**2/sn1e-2*self.df*np.abs(res_amplitude*hi)**2/sn2e)
        return burst


class burstevidence2_old_response_notchout(bilby.Likelihood):
    """
    Likelihood for two detectors with correlation and detector response
    Including notch out settings
    """

    def __init__(self, x, y1, y2, sn1, sn2, w1, w2, function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y1,2: array-like
            The strain data of frequency domain
        sn1,2: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        fmin & fmax: float
            describes the minimum and maximum of frequency array
        res_amplitude： float
            describes the amplitude difference of the two detector responses
        res_phase0： float
            describes the overall phase difference of the two detector responses
        res_dt： float
            describes the arrival time delay between two detecotrs, determining the frequency dependent phase difference of the two detector responses
        """
        super().__init__(parameters={'fmin': None,'fmax': None,'res_amplitude': None,'res_phase0': None,'res_dt': None})
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.sn1 = sn1
        self.sn2 = sn2
        self.w1 = w1
        self.w2 = w2
        self.function = function
        self.N = len(x)
        self.df = df


        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        res_amplitude = self.parameters['res_amplitude']
        res_phase0 = self.parameters['res_phase0']
        res_dt = self.parameters['res_dt']


        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sn1e = self.sn1[int(fmin/self.df):int(fmax/self.df+2)]
        y1e = self.y1[int(fmin/self.df):int(fmax/self.df+2)]
        sn2e = self.sn2[int(fmin/self.df):int(fmax/self.df+2)]
        y2e = self.y2[int(fmin/self.df):int(fmax/self.df+2)]
        w1e = self.w1[int(fmin/self.df):int(fmax/self.df+2)]
        w2e = self.w2[int(fmin/self.df):int(fmax/self.df+2)]
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        hi = self.function(xe,  **model_parameters)[0]

        y = w1e * y1e/sn1e + res_amplitude * w2e * y2e/sn2e *np.exp(res_phase0*1j)*np.exp(-2*1j*np.pi*res_dt*xe)

        burst = np.sum(logi0(4*self.df*np.abs(y)*np.abs(hi))-2*self.df*np.abs(hi)**2*w1e/sn1e-2*self.df*np.abs(res_amplitude*hi)**2*w2e/sn2e)
        return burst


class burstevidence_qnm(bilby.Likelihood):
    """
    Likelihood for one detector with no detector response, while keeping the phase associated with the pole structure
    """

    def __init__(self, x, y, angle, sn, function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y: array-like
            The abs of strain data of frequency domain
        angle: array-like
            The angle of strain data of frequency domain
        sn: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        """
        super().__init__(parameters={'fmin': None,'fmax': None})

        self.df = df
        self.x = x
        self.y = y
        self.angle = angle
        self.sn = sn
        ## maybe later sn will be replaced by an interpolating function
        self.function = function
        self.N = len(x)



        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sne = self.sn[int(fmin/self.df):int(fmax/self.df+2)]
        ye = self.y[int(fmin/self.df):int(fmax/self.df+2)]
        phi = self.angle[int(fmin/self.df):int(fmax/self.df+2)]

        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        hj = self.function(xe,  **model_parameters)[0]
        arg_h = self.function(xe,  **model_parameters)[1]
        qnm_number = self.function(xe,  **model_parameters)[2]
        arg = phi - arg_h

        arg_part = np.split(arg,np.unique(qnm_number, return_index=True)[1][1:])
        hj_part = np.split(hj,np.unique(qnm_number, return_index=True)[1][1:])
        y_part = np.split(ye,np.unique(qnm_number, return_index=True)[1][1:])
        sn_part = np.split(sne,np.unique(qnm_number, return_index=True)[1][1:])
        # here we divide this all array into different parts by qnm_number, which means the array for n-th resonance
        burst_n = []

        for part_index in np.arange(0,len(hj_part)):
            hj_i = hj_part[part_index]
            arg_i = arg_part[part_index]
            y_i = y_part[part_index]
            sn_i = sn_part[part_index]
            burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i*np.exp(1j*arg_i)/sn_i)))-np.sum(2*self.df*np.abs(hj_i)**2/sn_i ))
        burst = np.sum(burst_n)
        return burst

# old version
# class burstevidence2_qnm_response(bilby.Likelihood):
#     """
#     Likelihood for two detector with correlation and detector response, while keeping the phase associated with the pole structure
#     """

#     def __init__(self, x, y1, angle1, sn1, y2, angle2, sn2, function,df):
#         """

#         Parameters
#         ----------
#         x: array-like
#             The frequency array
#         y: array-like
#             The abs of strain data of frequency domain
#         angle: array-like
#             The angle of strain data of frequency domain
#         sn: array-like
#             noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
#         phase: float
#             The parametre of comb function, it should be between 0 to 1
#         width: float
#             describes the width of signal
#         spacing: float
#             describes the whole length of a wave
#         fmin & fmax: float
#             describes the minimum and maximum of frequency array
#         res_amplitude： float
#             describes the amplitude difference of the two detector responses
#         res_phase0： float
#             describes the overall phase difference of the two detector responses
#         res_dt： float
#             describes the arrival time delay between two detecotrs, determining the frequency dependent phase difference of the two detector responses
#         """
#         super().__init__(parameters={'fmin': None,'fmax': None,'res_amplitude': None,'res_phase0': None,'res_dt': None})

#         self.df = df
#         self.x = x
#         self.y1 = y1
#         self.angle1 = angle1
#         self.sn1 = sn1
#         self.y2 = y2
#         self.angle2 = angle2
#         self.sn2 = sn2
#         ## maybe later sn will be replaced by an interpolating function
#         self.function = function
#         self.N = len(x)



#         parameters = inspect.getfullargspec(function).args
#         parameters.pop(0)
#         self.parameters = dict.fromkeys(parameters)
#         self.function_keys = self.parameters.keys()

#     def log_likelihood(self):
#         fmin = self.parameters['fmin']
#         fmax = self.parameters['fmax']
#         res_amplitude = self.parameters['res_amplitude']
#         res_phase0 = self.parameters['res_phase0']
#         res_dt = self.parameters['res_dt']


#         xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
#         sn1e = self.sn1[int(fmin/self.df):int(fmax/self.df+2)]
#         y1e = self.y1[int(fmin/self.df):int(fmax/self.df+2)]
#         sn2e = self.sn2[int(fmin/self.df):int(fmax/self.df+2)]
#         y2e = self.y2[int(fmin/self.df):int(fmax/self.df+2)]
#         phi1e = self.angle1[int(fmin/self.df):int(fmax/self.df+2)]
#         phi2e = self.angle2[int(fmin/self.df):int(fmax/self.df+2)]
#         model_parameters = {k: self.parameters[k] for k in self.function_keys}

#         hj = self.function(xe,  **model_parameters)[0]
#         arg_h = self.function(xe,  **model_parameters)[1]
#         qnm_number = self.function(xe,  **model_parameters)[2]
#         arg1 = phi1e - arg_h
#         arg2 = phi2e - arg_h

#         ye = y1e/sn1e*np.exp(1j*arg1) +  y2e/sn2e *np.exp(1j*arg2) * res_amplitude * np.exp(res_phase0*1j) * np.exp(-2*1j*np.pi*res_dt*xe)
#         sne_reverse = 1/sn1e + 1/sn2e * np.abs(res_amplitude)**2

#         hj_part = np.split(hj,np.unique(qnm_number, return_index=True)[1][1:])
#         y_normalized_part = np.split(ye,np.unique(qnm_number, return_index=True)[1][1:])
#         sn_part = np.split(sne_reverse,np.unique(qnm_number, return_index=True)[1][1:])
#         # here we divide this all array into different parts by qnm_number, which means the array for n-th resonance
#         burst_n = []

#         for part_index in np.arange(0,len(hj_part)):
#             hj_i = hj_part[part_index]
#             # arg_i = arg_part[part_index]
#             y_i = y_normalized_part[part_index]
#             sn_i = sn_part[part_index]
#             # burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i*np.exp(1j*arg_i)/sn_i)))-np.sum(2*self.df*np.abs(hj_i)**2/sn_i ))
#             burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i))-np.sum(2*self.df*np.abs(hj_i)**2*sn_i )))
#         burst = np.sum(burst_n)
#         return burst

class burstevidence2_qnm_response(bilby.Likelihood):
    """
    Likelihood for two detector with correlation and detector response, while keeping the phase associated with the pole structure
    """

    def __init__(self, x, y1, angle1, sn1, y2, angle2, sn2, function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y: array-like
            The abs of strain data of frequency domain
        angle: array-like
            The angle of strain data of frequency domain
        sn: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        fmin & fmax: float
            describes the minimum and maximum of frequency array
        res_amplitude： float
            describes the amplitude difference of the two detector responses
        res_phase0： float
            describes the overall phase difference of the two detector responses
        res_dt： float
            describes the arrival time delay between two detecotrs, determining the frequency dependent phase difference of the two detector responses
        """
        super().__init__(parameters={'fmin': None,'fmax': None,'res_amplitude': None,'res_phase0': None,'res_dt': None})

        self.df = df
        self.x = x
        self.y1 = y1
        self.angle1 = angle1
        self.sn1 = sn1
        self.y2 = y2
        self.angle2 = angle2
        self.sn2 = sn2
        ## maybe later sn will be replaced by an interpolating function
        self.function = function
        self.N = len(x)



        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        res_amplitude = self.parameters['res_amplitude']
        res_phase0 = self.parameters['res_phase0']
        res_dt = self.parameters['res_dt']


        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sn1e = self.sn1[int(fmin/self.df):int(fmax/self.df+2)]
        y1e = self.y1[int(fmin/self.df):int(fmax/self.df+2)]
        sn2e = self.sn2[int(fmin/self.df):int(fmax/self.df+2)]
        y2e = self.y2[int(fmin/self.df):int(fmax/self.df+2)]
        phi1e = self.angle1[int(fmin/self.df):int(fmax/self.df+2)]
        phi2e = self.angle2[int(fmin/self.df):int(fmax/self.df+2)]
        model_parameters = {k: self.parameters[k] for k in self.function_keys}

        # hj = self.function(xe,  **model_parameters)[0]
        # arg_h = self.function(xe,  **model_parameters)[1]
        # qnm_number = self.function(xe,  **model_parameters)[2]
        hj, arg_h, qnm_number = self.function(xe, **model_parameters)
        arg1 = phi1e - arg_h
        arg2 = phi2e - arg_h

        combined_phase2 = arg2 + res_phase0 - 2 * np.pi * res_dt * xe
        # ye = y1e/sn1e*np.exp(1j*arg1) +  y2e/sn2e *np.exp(1j*arg2) * res_amplitude * np.exp(res_phase0*1j) * np.exp(-2*1j*np.pi*res_dt*xe)
        ye = (y1e / sn1e) * np.exp(1j * arg1) + (y2e / sn2e) * np.exp(1j * combined_phase2) * res_amplitude
        # sne_reverse = 1/sn1e + 1/sn2e * np.abs(res_amplitude)**2
        sne_reverse = (1 / sn1e) + (1 / sn2e) * np.abs(res_amplitude) ** 2


        hj_part = np.split(hj,np.unique(qnm_number, return_index=True)[1][1:])
        y_normalized_part = np.split(ye,np.unique(qnm_number, return_index=True)[1][1:])
        sn_part = np.split(sne_reverse,np.unique(qnm_number, return_index=True)[1][1:])
        # here we divide this all array into different parts by qnm_number, which means the array for n-th resonance
        burst_n = []

        # for part_index in np.arange(0,len(hj_part)):
        #     hj_i = hj_part[part_index]
        #     # arg_i = arg_part[part_index]
        #     y_i = y_normalized_part[part_index]
        #     sn_i = sn_part[part_index]
        #     # burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i*np.exp(1j*arg_i)/sn_i)))-np.sum(2*self.df*np.abs(hj_i)**2/sn_i ))
        #     burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i))-np.sum(2*self.df*np.abs(hj_i)**2*sn_i )))
        for hj_i, y_i, sn_i in zip(hj_part, y_normalized_part, sn_part):
            burst_n.append(
                logi0(np.abs(np.sum(4 * self.df * hj_i * y_i))) - np.sum(2 * self.df * np.abs(hj_i) ** 2 * sn_i)
            )
        burst = np.sum(burst_n)
        return burst


class burstevidence2_qnm_response_notchout(bilby.Likelihood):
    """
    Likelihood for two detector with correlation and detector response, while keeping the phase associated with the pole structure.
    Including notch out settings
    """

    def __init__(self, x, y1, angle1, sn1, y2, angle2, sn2, w1, w2,function,df):
        """

        Parameters
        ----------
        x: array-like
            The frequency array
        y1,y2: array-like
            The abs of strain data of frequency domain
        angle1,angle2: array-like
            The angle of strain data of frequency domain
        sn1,sn2: array-like
            noise spectral density. this symbol is coherent with that in (5.2),arXiv:1811.02042v1
        w1,2: array-like
            A bool array for user to justify whether include the frequency bin in the sum of likelihood
            We use this parameter to notch out system errors.
        phase: float
            The parametre of comb function, it should be between 0 to 1
        width: float
            describes the width of signal
        spacing: float
            describes the whole length of a wave
        fmin & fmax: float
            describes the minimum and maximum of frequency array
        res_amplitude： float
            describes the amplitude difference of the two detector responses
        res_phase0： float
            describes the overall phase difference of the two detector responses
        res_dt： float
            describes the arrival time delay between two detecotrs, determining the frequency dependent phase difference of the two detector responses
        """
        super().__init__(parameters={'fmin': None,'fmax': None,'res_amplitude': None,'res_phase0': None,'res_dt': None})

        self.df = df
        self.x = x
        self.y1 = y1
        self.angle1 = angle1
        self.sn1 = sn1
        self.y2 = y2
        self.angle2 = angle2
        self.sn2 = sn2
        self.w1 = w1
        self.w2 = w2
        ## maybe later sn will be replaced by an interpolating function
        self.function = function
        self.N = len(x)



        parameters = inspect.getfullargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        fmin = self.parameters['fmin']
        fmax = self.parameters['fmax']
        res_amplitude = self.parameters['res_amplitude']
        res_phase0 = self.parameters['res_phase0']
        res_dt = self.parameters['res_dt']


        xe = self.x[int(fmin/self.df):int(fmax/self.df+2)]
        sn1e = self.sn1[int(fmin/self.df):int(fmax/self.df+2)]
        y1e = self.y1[int(fmin/self.df):int(fmax/self.df+2)]
        sn2e = self.sn2[int(fmin/self.df):int(fmax/self.df+2)]
        y2e = self.y2[int(fmin/self.df):int(fmax/self.df+2)]
        phi1e = self.angle1[int(fmin/self.df):int(fmax/self.df+2)]
        phi2e = self.angle2[int(fmin/self.df):int(fmax/self.df+2)]
        w1e = self.w1[int(fmin/self.df):int(fmax/self.df+2)]
        w2e = self.w2[int(fmin/self.df):int(fmax/self.df+2)]
        model_parameters = {k: self.parameters[k] for k in self.function_keys}

        # hj = self.function(xe,  **model_parameters)[0]
        # arg_h = self.function(xe,  **model_parameters)[1]
        # qnm_number = self.function(xe,  **model_parameters)[2]
        hj, arg_h, qnm_number = self.function(xe, **model_parameters)
        arg1 = phi1e - arg_h
        arg2 = phi2e - arg_h

        combined_phase2 = arg2 + res_phase0 - 2 * np.pi * res_dt * xe
        # ye = y1e/sn1e*np.exp(1j*arg1) +  y2e/sn2e *np.exp(1j*arg2) * res_amplitude * np.exp(res_phase0*1j) * np.exp(-2*1j*np.pi*res_dt*xe)
        ye = (w1e * y1e / sn1e) * np.exp(1j * arg1) + (w2e * y2e / sn2e) * np.exp(1j * combined_phase2) * res_amplitude
        # sne_reverse = 1/sn1e + 1/sn2e * np.abs(res_amplitude)**2
        sne_reverse = w1e * (1 / sn1e) + w2e * (1 / sn2e) * np.abs(res_amplitude) ** 2


        hj_part = np.split(hj,np.unique(qnm_number, return_index=True)[1][1:])
        y_normalized_part = np.split(ye,np.unique(qnm_number, return_index=True)[1][1:])
        sn_part = np.split(sne_reverse,np.unique(qnm_number, return_index=True)[1][1:])
        # here we divide this all array into different parts by qnm_number, which means the array for n-th resonance
        burst_n = []

        # for part_index in np.arange(0,len(hj_part)):
        #     hj_i = hj_part[part_index]
        #     # arg_i = arg_part[part_index]
        #     y_i = y_normalized_part[part_index]
        #     sn_i = sn_part[part_index]
        #     # burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i*np.exp(1j*arg_i)/sn_i)))-np.sum(2*self.df*np.abs(hj_i)**2/sn_i ))
        #     burst_n = np.append(burst_n, logi0(np.abs(np.sum(4*self.df*hj_i*y_i))-np.sum(2*self.df*np.abs(hj_i)**2*sn_i )))
        for hj_i, y_i, sn_i in zip(hj_part, y_normalized_part, sn_part):
            burst_n.append(
                logi0(np.abs(np.sum(4 * self.df * hj_i * y_i))) - np.sum(2 * self.df * np.abs(hj_i) ** 2 * sn_i)
            )
        burst = np.sum(burst_n)
        return burst
