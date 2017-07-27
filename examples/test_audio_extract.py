from audio_tools import fetch_sample_speech_tapestry, world_synthesis
from audio_tools import harvest, cheaptrick, d4c, sp2mgc, mgc2sp
from audio_tools import soundsc
from scipy.io import wavfile
from scipy import fftpack
import numpy as np
import time


def run_world_mgc_example():
    # run on chromebook
    # enc 839.71
    # synth 48.79
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15

    # harcoded for 16k from
    # https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world/extract_features_for_merlin.sh
    mgc_alpha = 0.58
    #mgc_order = 59
    mgc_order = 59
    # this is actually just mcep
    mgc_gamma = 0.0

    def enc():
        temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
        temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
                temporal_positions_h, f0_h, vuv_h)
        temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
                temporal_positions_h, f0_h, vuv_h)

        mgc_arr = sp2mgc(spectrogram_ct, mgc_order, mgc_alpha, mgc_gamma,
                verbose=True)
        return mgc_arr, spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c


    start = time.time()
    mgc_arr, spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c = enc()
    enc_done = time.time()

    sp_r = mgc2sp(mgc_arr, mgc_alpha, mgc_gamma, fs=fs, verbose=True)
    synth_done = time.time()

    print("enc time: {}".format(enc_done - start))
    print("synth time: {}".format(synth_done - enc_done))
    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, sp_r, fs)
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, sp_r, fs)
    wavfile.write("out_mgc.wav", fs, soundsc(y))


def run_world_base_example():
    # on chromebook
    # enc 114.229
    # synth 5.165
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15

    def enc():
        temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
        temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
                temporal_positions_h, f0_h, vuv_h)
        temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
                temporal_positions_h, f0_h, vuv_h)

        return spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c


    start = time.time()
    spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c = enc()
    enc_done = time.time()

    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, spectrogram_ct, fs)
    synth_done = time.time()

    print("enc time: {}".format(enc_done - start))
    print("synth time: {}".format(synth_done - enc_done))
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, sp_r, fs)
    wavfile.write("out_base.wav", fs, soundsc(y))


def run_world_dct_example():
    # on chromebook
    # enc 114.229
    # synth 5.165
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15

    def enc():
        temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
        temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
                temporal_positions_h, f0_h, vuv_h)
        temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
                temporal_positions_h, f0_h, vuv_h)

        return spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c

    start = time.time()
    spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c = enc()
    dct_buf = fftpack.dct(spectrogram_ct)
    n_fft = 512
    n_dct = 20
    dct_buf = dct_buf[:, :n_dct]
    idct_buf = np.zeros((dct_buf.shape[0], n_fft + 1))
    idct_buf[:, :n_dct] = dct_buf
    ispectrogram_ct = fftpack.idct(idct_buf)
    enc_done = time.time()

    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, spectrogram_ct, fs)
    synth_done = time.time()

    print("enc time: {}".format(enc_done - start))
    print("synth time: {}".format(synth_done - enc_done))
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, sp_r, fs)
    wavfile.write("out_dct.wav", fs, soundsc(y))


#run_world_mgc_example()
#run_world_base_example()
run_world_dct_example()
