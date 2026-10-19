"""DUET blind source separation via attenuation-delay histograms.

Implements the Degenerate Unmixing Estimation Technique (DUET) for two
microphone channels.  A weighted 2-D histogram of symmetric attenuation
and relative delay is built from the STFT, peaks in that histogram seed
per-source masks, and each time-frequency bin is assigned to the closest
source prototype before synthesis back to the time domain.  The flow
follows the original CASA495 reference implementation.
"""

import logging

import numpy as np

from chandelier.localizer.peaks import find_peak_indices


logger = logging.getLogger(__name__)
# Guards the attenuation ratio and logarithms for silent spectral bins.
EPSILON = 1e-16


def tf_synthesis(n_sources, timefreqmat, swin, hop_length, n_fft):
    """
    time-frequency synthesis\n
    TIMEFREQMAT is the complex matrix time-freq representation\n
    SWIN is the synthesis window\n
    TIMESTEP is the # of samples between adjacent time windows.\n
    NUMFREQ is the # of frequency components per time point.\n
    X contains the reconstructed signal.\n
    """
    # MATLAB and Fortran use column-major layout by default,
    # whereas C and C++ use row-major layout
    swin = np.reshape(swin, -1, "F")

    win_length = swin.size
    _, n_fft, numtime = timefreqmat.shape

    # Map each synthesis-window sample onto its position in the inverse
    # FFT output; necessary because the window may be longer than n_fft.
    ind = np.fmod(np.arange(win_length), n_fft)
    x = np.zeros((n_sources, (numtime - 1) * hop_length + win_length))

    # Using broadcasted version can speed up about 4 times.
    # Origin code:
    # for i in range(numtime):
    #     temp = n_fft * ifft(timefreqmat[:, i]).real
    #     sind = i * hop_length
    #     for j in range(win_length):
    #         x[sind+j] = x[sind+j] + temp[ind[j]] * swin[j]
    # np.fft.ifft normalises by 1/n_fft; undo that to match the textbook
    # synthesis formula.
    temp = n_fft * np.fft.ifft(timefreqmat, axis=1).real
    for i in range(numtime):
        # Each frame contributes win_length samples starting at i * hop;
        # the two-hop slice matches the fixed 2:1 window/hop ratio here.
        x[:, i * hop_length : (i + 2) * hop_length] += temp[:, ind, i] * swin

    return x


def multichannel_tf_synthesis(n_sources, timefreqmat, swin, hop_length, n_fft):
    """
    time-frequency synthesis\n
    TIMEFREQMAT is the complex matrix time-freq representation\n
    SWIN is the synthesis window\n
    TIMESTEP is the # of samples between adjacent time windows.\n
    NUMFREQ is the # of frequency components per time point.\n
    X contains the reconstructed signal.\n
    """
    # Same Fortran-order reshape as the single-channel path.
    swin = np.reshape(swin, -1, "F")

    win_length = swin.size
    _, n_fft, numtime, chls = timefreqmat.shape

    ind = np.fmod(np.arange(win_length), n_fft)
    x = np.zeros((n_sources, (numtime - 1) * hop_length + win_length, chls))

    # Undo numpy's 1/n_fft normalisation, as in tf_synthesis.
    temp = n_fft * np.fft.ifft(timefreqmat, axis=1).real
    for i in range(numtime):
        # swin[..., None] broadcasts the one window across all channels.
        x[:, i * hop_length : (i + 2) * hop_length, :] += (
            temp[:, ind, i, :] * swin[..., None]
        )
    return x


def two_d_smooth(mat, ker):
    """
    Smoothening for better identification of the peaks in a graph.
    Could have used Gaussian Kernels to do the same but it seemed
    better visual effects were given when this algorithm was followed
    ( Again, based on original CASA495) MAT is the 2D matrix to be
    smoothed. KER is either\n
    (1) a scalar\n
    (2) a matrix which is used as the averaging kernel.\n
    """
    from scipy.signal import convolve2d

    # A scalar kernel means "average over a ker x ker square".
    try:
        len(ker)
        kmat = ker

    except TypeError:
        kmat = np.ones((ker, ker)) / ker**2

    kr, kc = kmat.shape
    # "same" convolution is only centred for odd kernels; grow even
    # dimensions by one so smoothing does not shift the histogram peaks.
    if kr % 2 == 0:
        kmat = convolve2d(kmat, np.ones((2, 1)), "symm", "same")
        kr += 1

    if kc % 2 == 0:
        kmat = convolve2d(kmat, np.ones((1, 2)), "symm", "same")
        kc += 1

    # convolve2d flips the kernel; rotate it so the operation behaves as
    # a centred weighted average.  "symm" mirrors the borders, avoiding
    # artificial dips where the histogram meets its edge.
    rota = np.rot90(kmat, 2)
    mat = convolve2d(mat, rota, "same", "symm")
    return mat


class Duet(object):
    """computes the Degenerate Unmixxing Estimation Technique (DUET).

    This class computes the the Degenerate Unmixxing Estimation Technique
    of an audio signal. It supports a microphone pair inputs. (more contents)

    Arguments
    ---------
    x : ndarray
        The input audio signal with at least two channels,
        The ndarray must have the following format: (n_channels, time_step).
    n_sources : int
        How many sources want to be seperated (maximun observed sources).
        (relative `numsources` in the paper)
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    mic_pair : tuple
        Configure which channel x1 and x2 are, assuming the input is a
        multi-channel audio. It raised an error if input is mono, i.g.
        shape=(1, :) is mono, shape=(2, :) is stereo. The default is
        None (equivalent to tuple(0, 1)).
    attenuation_max : float
        Only consider attenuation yielding estimates in bounds.
        (relative `maxa` in the paper)
    n_attenuation_bins : int
        The range of attenuation values distributed into bins, default is 35.
        (relative `abins` in the paper)
    delay_max : float
        Only consider delay yielding estimates in bounds.
        (relative `maxd` in the paper)
    n_delay_bins : int
        The range of delay values distributed into bins, default is 50.
        (relative `dbins` in the paper)
    p : int
        Weight the histogram with the symmetric attenuation estimator,
        default is 1.
    q : int
        Weight the histogram with the delay estimator, default is 0.
    output_all_channels : bool
        If True, using the mask on the whole channels, the ndarray
        must have the following format (batch, time_step, n_channels).
        The ndarray must have the following format (batch, time_step)
        if setting is False, default is False.

    Example
    -------
    >>> from bss import Duet
    >>> from scipy.io.wavfile import read, write
    >>> # x is stereo(2 channels)
    >>> fs, x = read("<FILEDIR>/x.wav")
    >>> duet = Duet(x, n_sources=5, sample_rate=fs)
    >>> estimates = duet()
    >>> for i in range(duet.n_sources):
    >>>     write(f"output{i}.wav", duet.fs, estimates[i, :]+0.05*duet.x1)
    """

    def __init__(
        self,
        x,
        n_sources,
        sample_rate,
        mic_pair=None,
        attenuation_max=0.7,
        n_attenuation_bins=35,
        delay_max=3.6,
        n_delay_bins=50,
        p=1,
        q=0,
        output_all_channels=False,
    ):
        self.x = x
        self.n_sources = n_sources
        self.fs = sample_rate
        self.mic_pair = mic_pair
        self.attenuation_max = attenuation_max
        self.n_attenuation_bins = n_attenuation_bins
        self.delay_max = delay_max
        self.n_delay_bins = n_delay_bins
        self.p = p
        self.q = q

        self.x1 = None
        self.x2 = None
        self.tf1 = None
        self.tf2 = None
        self.fmat = None
        self.symmetric_atn = None
        self.delay = None
        self.sym_atn_peak = None
        self.delay_peak = None
        self.atn_peak = None
        self.norm_atn_delay_hist = None
        self.tf_weight = None
        self.bestind = None
        self.prominences = None
        self._nfft = 1024
        self._win_length = 1024
        self._hop_length = 512
        self._awin = np.hamming(1024)
        self._output_all_channels = output_all_channels

        if self.mic_pair is None:
            self.mic_pair = (0, 1)

    def __call__(self):
        return self.run()

    def run(self):
        # Create the spectrogram of the Left and Right channels, and remove DC
        # component to avoid dividing by zero frequency in the delay estimation.
        self.tf1, self.tf2, self.fmat = self._construct_histogram(self.mic_pair)

        # For each time/frequency compare the phase and amplitude of the left and
        # right channels. This gives two new coordinates, instead of time-frequency
        # it is phase-amplitude differences.
        self.symmetric_atn, self.delay = self._compute_atn_delay(
            self.tf1, self.tf2, self.fmat
        )

        # Build a 2-d histogram (one dimension is phase, one is amplitude) where
        # the height at any phase/amplitude is the count of time-frequency bins that
        # have approximately that phase/amplitude.
        self.norm_atn_delay_hist, self.tf_weight = self._compute_weighted_hist(
            self.symmetric_atn, self.delay
        )

        # Find the location of peaks in the attenuation-delay plane
        self.sym_atn_peak, self.delay_peak = self._find_n_peaks(
            self.norm_atn_delay_hist, n_peaks=self.n_sources, width=0.5, prominence=5.0
        )

        # Assign each time-frequency frame to the nearest peak in phase/amplitude
        # space. This partitions the spectrogram into sources (one peak per source)
        self.atn_peak, self.bestind = self._convert_peaks(self.sym_atn_peak)

        # Compute masks for separation
        # (1) Create a binary mask (1 for each tf-point belonging to my source, 0 for others)
        # (2) Mask the spectrogram with the mask created in (1).
        # (3) Rebuild the original wave file from (2).
        if self._output_all_channels:
            return self._build_multichannel_masks(self.atn_peak, self.bestind)

        return self._build_masks(self.atn_peak, self.bestind)

    def _construct_histogram(self, mic_pair):
        """
        Construct the two-dimensional weighted histogram.

        Following the step.1 in the paper.

        Arguments
        ---------
        mic_pair : tuple
            Configure which channel x1 and x2 are.

        Returns
        -------
        tf1 : ndarray
            STFT of x1, the ndarray must have the following format (t, f).
        tf2 : ndarray
            STFT of x2, the ndarray must have the following format (t, f).
        fmat : ndarray
            Frequency matrix, the ndarray must have the following format (t, f).
        """
        import scipy.signal as sp_signal

        # Scale int16 PCM to [-1, 1] so both channels share one reference.
        self.x1 = self.x[mic_pair[0]] / np.iinfo(np.int16).max
        self.x2 = self.x[mic_pair[1]] / np.iinfo(np.int16).max

        # time-freq domain
        # return_onesided=False keeps signed frequencies, which the delay
        # estimator needs to distinguish positive from negative delays.
        _, _, tf1 = sp_signal.stft(
            self.x1,
            fs=self.fs,
            window=self._awin,
            nperseg=self._win_length,
            return_onesided=False,
        )
        _, _, tf2 = sp_signal.stft(
            self.x2,
            fs=self.fs,
            window=self._awin,
            nperseg=self._win_length,
            return_onesided=False,
        )

        # removing DC component
        # The DC bin carries no phase information and would divide by zero
        # in the delay estimate below.
        # Since the scipy stft will scale the return value, in order to match the
        # paper result, it should be rescaled back to the origin. Scipy stft pass
        # the scaling == 'spectrum' and mode == 'stft', the values will multiply
        # np.sqrt(1.0 / win.sum()**2). Here are the source codes below.
        # (1) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1174
        # (2) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1806
        tf1 = tf1[1:, :] * self._awin.sum()
        tf2 = tf2[1:, :] * self._awin.sum()

        # Frequencies must follow scipy's full-FFT ordering: positive bins
        # first, then negative bins (DC already dropped).
        # calculate pos/neg frequencies for later use in delay calc
        h1 = np.arange(1, (self._nfft / 2) + 1)
        h2 = np.arange(-(self._nfft / 2) + 1, 0)
        freq = np.concatenate((h1, h2)) * ((2 * np.pi) / self._nfft)

        # Replicate the angular-frequency vector for every time frame so
        # it broadcasts against tf1/tf2 (shape (t, f)).
        h = np.ones((tf1.shape[1], freq.shape[0]))
        for i in range(h.shape[0]):
            h[i] *= freq
        fmat = h.transpose()

        return tf1, tf2, fmat

    def _compute_atn_delay(self, tf1, tf2, fmat):
        """
        Calculate the symmetric attenuation (alpha) and delay (delta) for each t-f point.

        Following the step.2 in the paper, 'alpha' relative symmetric attenuation and
        'delta' relative delay

        Arguments
        ---------
        tf1 : ndarray
            STFT of x1, output from the stft function.
        tf2 : ndarray
            STFT of x2, output from the stft function.
        fmat : ndarray
            Frequency matrix.

        Returns
        -------
        alpha : ndarray
            The symmetric attenuation.
            The ndarray must have the following format (t, f).
        delta : ndarray
            The relative delay.
            The ndarray must have the following format (t, f).
        """
        # Ratio of the two channels at each time-frequency bin.  EPSILON
        # keeps silent bins finite instead of producing inf/nan.
        R21 = (tf2 + EPSILON) / (tf1 + EPSILON)
        # |R21| is the level ratio; alpha = a - 1/a is symmetric around
        # zero for equal levels, so the histogram does not privilege a
        # particular channel ordering.
        a = np.abs(R21)
        alpha = a - 1.0 / a
        # The phase of the ratio is -omega * delta for a pure delay, so
        # dividing by the angular frequency recovers delta in samples.
        # DC was removed from fmat, so this division cannot hit zero.
        delta = -np.imag(np.log(R21)) / fmat

        return alpha, delta

    def _compute_weighted_hist(self, alpha, delta):
        """
        Calculate weighted histogram

        Following the step.3 in the paper.

        Arguments
        ---------
        alpha : ndarray
            The symmetric attenuation.
            The ndarray must have the following format (t, f).
        delta : ndarray
            The relative delay.
            The ndarray must have the following format (t, f).

        Returns
        -------
        A : ndarray
            A normalized 2D histogram of symmetric attenuation and delay.
            The ndarray must have the following format (alpha, delta).
        tf_weight : ndarray
            Weights. (should add more explanation to this param)
        """
        import scipy.sparse as sp_sparse

        # Weight each bin by the energy in both channels; p controls how
        # strongly the quieter channel counts, while q emphasises higher
        # frequencies, where a phase error means a smaller delay error.
        h1 = np.abs(self.tf1) * np.abs(self.tf2) ** self.p
        h2 = np.abs(self.fmat) ** self.q
        tf_weight = h1 * h2

        # only consider time-freq points yielding estimates in bounds
        amask = (np.abs(alpha) < self.attenuation_max) & (
            np.abs(delta) < self.delay_max
        )
        alpha_vec = alpha[amask]
        delta_vec = delta[amask]
        tf_weight = tf_weight[amask]

        # determine histogram indices
        # Map alpha/delta from [-max, +max] onto a zero-based bin; rounding
        # is required because sparse matrix indices must be integers.
        alphaind = np.around(
            (self.n_attenuation_bins - 1)
            * (alpha_vec + self.attenuation_max)
            / (2 * self.attenuation_max)
        )
        deltaind = np.around(
            (self.n_delay_bins - 1)
            * (delta_vec + self.delay_max)
            / (2 * self.delay_max)
        )

        # FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
        # A(alphaind(k),deltaind(k)) = tf_weight(k), S is abins-by-dbins
        # Duplicate coordinates are summed by the COO-to-CSR conversion,
        # which yields the weighted histogram in a single pass.
        A = sp_sparse.csr_matrix(
            (tf_weight, (alphaind, deltaind)),
            shape=(self.n_attenuation_bins, self.n_delay_bins),
        ).toarray()

        # smooth the histogram - local average 3-by-3 neighboring bins
        A = two_d_smooth(A, 3)

        return A, tf_weight

    def _find_n_peaks(
        self,
        norm_atn_delay_hist,
        n_peaks=None,
        width=None,
        threshold=0.2,
        prominence=None,
    ):
        """
        Find the n largest peaks in the 2D histogram.

        Following the step.4 in the paper.

        Arguments
        ---------
        norm_atn_delay_hist : ndarray
            A normalized 2D histogram of symmetric attenuation and delay.
            The ndarray must have the following format (alpha, delta).
        n_peaks : int
            How many peaks should be detected. If is None, it will set ot 5.
        width : ndarray
            Required width of peaks in samples.
        prominences : ndarray
            The calculated prominences for each peak in peaks. Wikipedia
            article for Topographic Prominence:
            https://en.wikipedia.org/wiki/Topographic_prominence

        Returns
        -------
        atn_peak : ndarray
            An array contains the peaks of symmetric attenuation.
            The ndarray must have the following format (n_peaks, ).
        delay_peak : ndarray
            An array contains the peaks of delay.
            The ndarray must have the following format (n_peaks, ).
        """
        # Bin centres converted back to physical units.  Rows of the
        # histogram follow attenuation and columns follow delay.
        x = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        y = np.linspace(
            -self.attenuation_max, self.attenuation_max, self.n_attenuation_bins
        )

        if n_peaks is None:
            n_peaks = 5

        if prominence is None:
            logger.debug("using max-peak searching")
            # Peaks: [a_idx, d_inx] in descending height.
            peaks = np.asarray(
                find_peak_indices(
                    norm_atn_delay_hist,
                    n_peaks=n_peaks,
                    min_dist=1,
                    threshold=threshold,
                )
            )

            cand_peaks = norm_atn_delay_hist[peaks[:, 0], peaks[:, 1]]
            if n_peaks is None:
                std = np.sqrt((np.abs(cand_peaks - cand_peaks[0]) ** 2).mean())
                cand_peaks = cand_peaks[np.abs(cand_peaks - cand_peaks[0]) < std]

            # Fewer peaks than requested can come back once the
            # neighbourhood zeroing has removed everything else.
            amax_idx = peaks[: cand_peaks.size, 0]
            dmax_idx = peaks[: cand_peaks.size, 1]

        else:
            # https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
            from scipy.signal import find_peaks

            # Collapse the histogram onto the delay axis, keeping the
            # strongest attenuation for each delay bin.
            delay_side = np.max(norm_atn_delay_hist, axis=0)

            dmax_idx, prop = find_peaks(
                delay_side,
                width=width,
                prominence=prominence,
            )

            # Rank the 1-D peaks by prominence and keep the best n_peaks.
            prom_rank = np.argsort(prop["prominences"])[::-1][:n_peaks]
            dmax_idx = dmax_idx[prom_rank]
            self.prominences = prop["prominences"][prom_rank]
            if dmax_idx.size == 0:
                # No column passed the prominence/width test: fall back to
                # the global maximum so the caller still gets a direction.
                dmax_idx = [np.argmax(delay_side)]
            # For each selected delay column, take its strongest
            # attenuation bin.
            amax_idx = np.argmax(norm_atn_delay_hist[:, dmax_idx], axis=0)
        atn_peak = y[amax_idx]
        delay_peak = x[dmax_idx]

        return atn_peak, delay_peak

    def _convert_peaks(self, sym_atn_peak):
        """
        Determine masks for separation.

        Following the step.5 in the paper.

        Arguments
        ---------
        sym_atn_peak : ndarray
            An array contains the peaks of symmetric attenuation.
            The ndarray must have the following format (n_peaks, ).

        Returns
        -------
        peaka : ndarray
            an array contains the peaks of attenuation.
            The ndarray must have the following format (n_peaks, ).
        bestind : ndarray
            An array contains the each source which is a mask.
            The ndarray must have the following format (n_peaks, t, f)
        """
        # convert the symmetric attenuation back to attenuation
        # Positive root of a^2 - alpha * a - 1 = 0; a is a level ratio,
        # so only this root is physical.
        peaka = (sym_atn_peak + np.sqrt(np.square(sym_atn_peak) + 4)) / 2
        bestsofar = float("inf") * np.ones(self.tf1.shape)
        # 1-based labels leave zero free to mean "unassigned".
        bestind = np.zeros(self.tf1.shape)

        for i in range(sym_atn_peak.size):
            # Score how well this prototype's predicted second-channel
            # spectrum matches the observed one, normalised by its gain.
            score = (
                np.abs(
                    peaka[i] * np.exp(-1j * self.fmat * self.delay_peak[i]) * self.tf1
                    - self.tf2
                )
                ** 2
            ) / (1 + peaka[i] ** 2)
            mask = score < bestsofar
            s_mask = score[mask]
            np.place(bestind, mask, i + 1)
            np.place(bestsofar, mask, s_mask)

        return peaka, bestind

    def _build_masks(self, atn_peak, bestind):
        """
        Demix with ML alignment and convert to time domain.

        Following the step.6 and step.7 in the paper.

        Arguments
        ---------
        atn_peak : ndarray
            An array contains the peaks of attenuation.
            The ndarray must have the following format (n_peaks, ).
        bestind : ndarray
            An array contains the each source which is a mask.
            The ndarray must have the following format (n_peaks, t, f).

        Returns
        -------
        est : ndarray
            an array contains a seperated wave stream of all speakers.
            The ndarray must have the following format (batch, time_step).
        """
        # 'h' stands for helper, we're using helper variables to break down
        # the logic of what's going on. Apologies for the order of the 'h's
        # Broadcast(a bit faster) the n_sources estimations and return directly.
        # h1     -> (1, 129)
        # h3     -> (1,) * (1023, 129) * (1023, 129)
        # h4     -> (1023, 129) / (1,)
        # h2     -> (1023, 129) * (1023, 129)
        # h      -> (1+1023, 129)
        # new_h1 -> (n_src, 1, 129)
        # new_h3 -> (n_src, None, None) * (n_src, 1023, 129) * (None, 1023, 129)
        # new_h4 -> (None, 1023, 129) / (n_src,)
        # new_h2 -> (n_src, 1023, 129) * (n_src, 1023, 129)
        # new_h  -> (n_src, 1+1023, 129)
        #
        # Origin code:
        # est = np.zeros((self.n_sources, self.x1.shape[-1]))
        # for i in range(self.n_sources):
        #     mask = (bestind == i+1)
        #     h1 = np.zeros((1, self.tf1.shape[-1]))
        #     h3 = atn_peak[i] * np.exp(1j*self.fmat*self.delay_peak[i]) * self.tf2
        #     h4 = ((self.tf1+h3) / (1+atn_peak[i]**2))
        #     h2 = h4 * mask
        #     h = np.concatenate((h1, h2))
        #
        #     esti = tf_synthesis(h, np.sqrt(2)*self._awin/1024, self._hop_length, self._nfft)
        #
        #     # add back into the demix a little bit of the mixture
        #     # as that eliminates most of the masking artifacts
        #     est[i] = esti[0:self.x1.shape[-1]]
        #     write(f"out{i}.wav", self.fs, est[i]+0.05*self.x1)
        # Least-squares demixing estimate for each source:
        #   (X1 + a * exp(j*omega*delta) * X2) / (1 + a^2),
        # with a and delta taken from that source's histogram peak.
        h3 = (
            atn_peak[:, None, None]
            * np.exp(1j * self.fmat[None, ...] * self.delay_peak[:, None, None])
            * self.tf2[None, ...]
        )
        h4 = (self.tf1[None, ...] + h3) / (1 + atn_peak[:, None, None] ** 2)

        # In order to avoid errors caused by the observed source
        # being less than the source we set.
        observed_src = h4.shape[0]
        mask = np.zeros((observed_src, *bestind.shape))
        for i in range(observed_src):
            mask[i, ...] = bestind == i + 1

        # tf_synthesis expects the DC row that _construct_histogram
        # removed, so prepend a zero row to each masked spectrogram.
        h1 = np.zeros((observed_src, 1, self.tf1.shape[-1]))
        h2 = h4 * mask

        h = np.concatenate((h1, h2), axis=1)

        # Reuse the analysis window for synthesis, scaled for the 50 %
        # overlap-add performed by tf_synthesis.
        est = tf_synthesis(
            observed_src,
            h,
            np.sqrt(2) * self._awin / 1024,
            self._hop_length,
            self._nfft,
        )

        return est[:, 0 : self.x1.shape[-1]]

    def _build_multichannel_masks(self, atn_peak, bestind):
        """
        Demix with ML alignment and convert to time domain.

        Following the step.6 and step.7 in the paper.

        Arguments
        ---------
        bestind : ndarray
            An array contains the each source which is a mask.
            The ndarray must have the following format (n_peaks, t, f)

        Returns
        -------
        est : ndarray
            An array contains a seperated wave stream of all speakers.
            The ndarray must have the following format (batch, time_step).
        """
        # 'h' stands for helper, we're using helper variables to break down
        # the logic of what's going on. Apologies for the order of the 'h's
        # Broadcast(a bit faster) the n_sources estimations and return directly.
        # mask -> (n_src, t, f)
        # h3   -> (t, f, n_channels)
        # h1   -> (n_src, 1, f, n_channels)  # DC component
        # h2   -> (None, t, f, n_channels) * (n_src, t, f, None)
        # h    -> (n_src, 1+t, f, n_channels)

        import scipy.signal as sp_signal

        # Scale int16 PCM the same way as _construct_histogram.
        xs = self.x / np.iinfo(np.int16).max

        # time-freq domain
        _, _, tfs = sp_signal.stft(
            xs,
            fs=self.fs,
            window=self._awin,
            nperseg=self._win_length,
            return_onesided=False,
        )

        # removing DC component
        tfs = tfs[:, 1:, :] * self._awin.sum()

        # (n_channels, t, f) -> (t, f, n_channels)
        h3 = tfs.transpose(1, 2, 0)

        observed_src = atn_peak.shape[0]
        mask = np.zeros((observed_src, *bestind.shape))
        for i in range(observed_src):
            mask[i, ...] = bestind == i + 1

        # The same per-source mask applies to every channel; the trailing
        # None broadcasts it across the channel axis.
        h1 = np.zeros((observed_src, 1, self.tf1.shape[-1], h3.shape[-1]))
        h2 = h3[None, ...] * mask[..., None]

        h = np.concatenate((h1, h2), axis=1)

        est = multichannel_tf_synthesis(
            observed_src,
            h,
            np.sqrt(2) * self._awin / 1024,
            self._hop_length,
            self._nfft,
        )

        return est[:, 0 : self.x1.shape[-1], :]

    def plot_atn_delay_hist(self):
        """Render the attenuation-delay histogram as a 3-D surface.

        Requires :meth:`run` to have populated the histogram.  Side
        effect: writes ``DUET.html`` in the current directory and opens it
        in the default browser.
        """

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.norm_atn_delay_hist is None:
            raise RuntimeError("It should compute a weighted histogram first.")

        x = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        y = np.linspace(
            -self.attenuation_max, self.attenuation_max, self.n_attenuation_bins
        )
        # Default "xy" indexing puts attenuation on rows and delay on
        # columns, matching norm_atn_delay_hist.
        x, y = np.meshgrid(x, y)
        z = self.norm_atn_delay_hist

        fig = make_subplots(specs=[[{"type": "scene"}]])
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="plasma", opacity=0.8))

        # Draw the two 1-D marginals along the back edges: these are the
        # projections used when finding peaks.
        fig.add_trace(
            go.Scatter3d(
                x=x[0, :],
                y=[self.attenuation_max] * len(x[0, :]),
                z=np.max(z, axis=0),
                mode="lines",
                line=dict(color="hotpink", width=3),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[-self.delay_max] * len(y[:, 0]),
                y=y[:, 0],
                z=np.max(z, axis=1),
                mode="lines",
                line=dict(color="hotpink", width=3),
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Delay",
                yaxis_title="Attenuation",
                zaxis_title="Magnitude",
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12)),
                zaxis=dict(tickfont=dict(size=12)),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=-1.5, z=0.2),
                ),
            ),
            width=800,
            height=800,
        )
        fig.write_html("DUET.html", auto_open=True)
