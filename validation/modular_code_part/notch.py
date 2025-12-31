import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
import matplotlib.mlab as mlab
from gwpy.signal import filter_design

class NotchFilterProcessor:
    def __init__(self, frequency, samplingrate, df, psd_params):
        self.frequency = frequency
        self.samplingrate = samplingrate
        self.df = df
        self.psd_params = psd_params

    def identify_notch_frequencies(self, strain_normalized_abs, notchout_amplitude, notchout_list, notchout_list_errorbar):
        # Find the frequency bins where strain_normalized larger than notchout_amplitude
        notchout = self.frequency * np.array(strain_normalized_abs > notchout_amplitude)
        notchout = notchout[notchout != 0]
        
        # Create the frequency range list
        nol = np.transpose([notchout_list, notchout_list - notchout_list_errorbar, notchout_list + notchout_list_errorbar])

        # Determine the frequency bins that need to be notched out
        notchlist = []
        for no in nol:
            if np.any((notchout >= no[1]) & (notchout < no[2])):
                notchlist.append(no[0])
        return np.array(notchlist)

    def apply_notch_filters(self, data, notchlist, croptime):
        for f in notchlist:
            notches = [filter_design.notch(f, self.samplingrate)]
            zpk = filter_design.concatenate_zpks(*notches)
            data = data.filter(zpk, filtfilt=True)
        data = data.crop(*data.span.contract(croptime))
        return data

    # def compute_psd(self, data, Fs, NFFT, window, noverlap):
    #     # freqs, Pxx = welch(data, fs=fs, nperseg=nperseg, window=window, noverlap=noverlap)
    #     freqs , Pxx , = welch(data, fs = Fs, nfft = NFFT, window=window, noverlap=noverlap)
    #     psd = interp1d(freqs, Pxx, bounds_error=False, fill_value="extrapolate")
    #     return psd
    
    def compute_psd(self, data, Fs, NFFT, window, noverlap):
        Pxx, freqs = mlab.psd(data, Fs=Fs, NFFT=NFFT, window=window, noverlap=noverlap)
        psd = interp1d(freqs, Pxx)
        return psd

    def whiten_data(self, data, psd, dt):
        data_freq = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), dt)
        psd_values = psd(np.abs(freqs))
        psd_values[psd_values == 0] = np.inf  # 防止除以零
        #norm = 1./np.sqrt(1./(dt*2))
        norm = 1
        white_data_freq = data_freq / np.sqrt(psd_values) *norm
        white_data = np.fft.irfft(white_data_freq,n = len(data))
        return white_data
    


    def correct_psd(self, psd, notchlist, notchout_errorbar_for_psd):
        psd_values = psd(self.frequency)
        exclude_indices = []
        for center in notchlist:
            indices = np.arange(
                int((center + notchout_errorbar_for_psd[0]) / self.df),
                int((center + notchout_errorbar_for_psd[1]) / self.df) + 1
            )
            exclude_indices.extend(indices)
        exclude_indices = np.array(exclude_indices, dtype=int)
        exclude_indices = exclude_indices[
            (exclude_indices >= 0) & (exclude_indices < len(psd_values))
        ]
        exclude_indices = np.sort(np.unique(exclude_indices))

        # 识别连续的区间
        gaps = np.diff(exclude_indices)
        discontinuity_indices = np.where(gaps > 1)[0]
        interval_starts = exclude_indices[np.insert(discontinuity_indices + 1, 0, 0)]
        interval_ends = exclude_indices[np.append(discontinuity_indices, len(exclude_indices) - 1)]

        psd_values_corrected = psd_values.copy()
        for start, end in zip(interval_starts, interval_ends):
            # 获取区间左右的有效索引
            # left = start - 1
            # right = end + 1
            # while left in exclude_indices and left > 0:
            #     left -= 1
            # while right in exclude_indices and right < len(psd_values) - 1:
            #     right += 1
            # 线性插值修正区间内的PSD值
            left = start
            right = end + 1
            if left >= 0 and right < len(psd_values):
                psd_values_corrected[start:end+1] = np.interp(
                    self.frequency[start:end+1],
                    [self.frequency[left], self.frequency[right]],
                    [psd_values[left], psd_values[right]]
                )
            elif left >= 0:
                psd_values_corrected[start:end+1] = psd_values[left]
            elif right < len(psd_values):
                psd_values_corrected[start:end+1] = psd_values[right]

        corrected_psd = interp1d(
            self.frequency, psd_values_corrected,
            bounds_error=False, fill_value="extrapolate"
        )
        return corrected_psd

    def compute_whether_include_in_likelihood(self, notchlist, notchout_errorbar_for_likelihood):
        whether_include_in_likelihood = np.ones(len(self.frequency), dtype=bool)
        exclude_indices = []
        for center in notchlist:
            indices = np.arange(int((center + notchout_errorbar_for_likelihood[0]) / self.df),
                                int((center + notchout_errorbar_for_likelihood[1]) / self.df) + 1)
            exclude_indices.extend(indices)
        exclude_indices = np.array(exclude_indices, dtype=int)
        exclude_indices = exclude_indices[(exclude_indices >= 0) & (exclude_indices < len(whether_include_in_likelihood))]
        whether_include_in_likelihood[exclude_indices] = False
        return whether_include_in_likelihood

    def process(self, data, Nt, sampling_rate , notchout_amplitude, notchout_list, notchout_list_errorbar,
                     notchout_errorbar_for_psd, notchout_errorbar_for_likelihood,  croptime, max_iterations=10):
        dt = 1/sampling_rate
        total_notchlist = []
        iteration = 0
        data_filtered = data.copy()
        total_croptime = 0
        while iteration < max_iterations:        
            print(total_croptime)   
            psd = self.compute_psd(data_filtered[Nt-self.samplingrate*total_croptime:(2*Nt-self.samplingrate*total_croptime)], **self.psd_params)
            # psd uses the intermidiate data to compute
            data_whitened = self.whiten_data(data_filtered, psd, dt)
            
            strain_normalized_abs = self.compute_normalized_abs(data_whitened, dt, total_croptime, Nt)
            # strain_normalized_abs also uses the intermidiate data to compute
            
            notchlist = self.identify_notch_frequencies(
                strain_normalized_abs, notchout_amplitude, notchout_list, notchout_list_errorbar)
            # exit the loop if there is no new frequency to notch out
            new_notches = [f for f in notchlist if f not in total_notchlist]
            if len(new_notches) == 0:
                break
            # update the total_notchlist
            total_notchlist.extend(new_notches)
            data_filtered = self.apply_notch_filters(data_filtered, new_notches, croptime)

            total_croptime += croptime
            iteration += 1
            print(f"Iteration {iteration}: {new_notches}")

        whether_include_in_likelihood = self.compute_whether_include_in_likelihood(
            total_notchlist, notchout_errorbar_for_likelihood)      

        psd_corrected = self.correct_psd(psd, total_notchlist, notchout_errorbar_for_psd)
        # Note here we use the intermidiate data to compute psd, then correct it.
        # It causes the psd_corrected is not the same as the previous code. The previous code is wrong.

        return data_filtered, data_whitened, psd_corrected, whether_include_in_likelihood,  total_notchlist, total_croptime # notchout_frequencies,
    
    
    def process_longerpsd(self, data, Nt, sampling_rate , notchout_amplitude, notchout_list, notchout_list_errorbar,
                     notchout_errorbar_for_psd, notchout_errorbar_for_likelihood,  croptime, max_iterations=10):
        dt = 1/sampling_rate
        total_notchlist = []
        iteration = 0
        data_filtered = data.copy()
        total_croptime = 0
        while iteration < max_iterations:        
            print(total_croptime)   
            psd = self.compute_psd(data_filtered[Nt-self.samplingrate*total_croptime:(2*Nt-self.samplingrate*total_croptime)], **self.psd_params)
            # psd uses the intermidiate data to compute
            data_whitened = self.whiten_data(data_filtered, psd, dt)
            
            strain_normalized_abs = self.compute_normalized_abs(data_whitened, dt, total_croptime, Nt)
            # strain_normalized_abs also uses the intermidiate data to compute
            
            notchlist = self.identify_notch_frequencies(
                strain_normalized_abs, notchout_amplitude, notchout_list, notchout_list_errorbar)
            # exit the loop if there is no new frequency to notch out
            new_notches = [f for f in notchlist if f not in total_notchlist]
            if len(new_notches) == 0:
                break
            # update the total_notchlist
            total_notchlist.extend(new_notches)
            data_filtered = self.apply_notch_filters(data_filtered, new_notches, croptime)

            total_croptime += croptime
            iteration += 1
            print(f"Iteration {iteration}: {new_notches}")

        whether_include_in_likelihood = self.compute_whether_include_in_likelihood(
            total_notchlist, notchout_errorbar_for_likelihood)      

        psd = self.compute_psd(data_filtered, **self.psd_params)
        psd_corrected = self.correct_psd(psd, total_notchlist, notchout_errorbar_for_psd)
        data_whitened = self.whiten_data(data_filtered, psd_corrected, dt)
        # Note here we use the full data to compute psd, then correct it. It is the same as the previous code, as for a test of coherence.

        return data_filtered, data_whitened, psd_corrected, whether_include_in_likelihood,  total_notchlist, total_croptime # notchout_frequencies,

    def compute_normalized_abs(self, data_whitened, dt, total_croptime, Nt):
        data_freq = dt * np.fft.fft(data_whitened[Nt-self.samplingrate*total_croptime:(2*Nt-self.samplingrate*total_croptime)])
        # data_freq = data_freq[:len(self.frequency)] *np.sqrt(psd(frequency)) 
        # strain_normalized_abs = np.abs(data_freq / np.sqrt(psd(self.frequency) / (4 * self.df)))
        data_freq = data_freq[:len(self.frequency)]
        strain_normalized_abs = np.abs(data_freq * np.sqrt(4 * self.df))
        return strain_normalized_abs
