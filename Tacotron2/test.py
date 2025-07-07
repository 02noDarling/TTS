import librosa
import librosa.display
import numpy as np
import soundfile as sf
import torch
import os

def wav_to_mel_spectrogram(wav_path, n_mels=128, n_fft=2048, hop_length=512, sample_rate=22050):
    """
    Convert a WAV file to a Mel spectrogram.
    
    Parameters:
    - wav_path: Path to the input WAV file.
    - n_mels: Number of Mel bands.
    - n_fft: Length of FFT window.
    - hop_length: Number of samples between successive frames.
    - sample_rate: Sampling rate of the audio.
    
    Returns:
    - mel_spectrogram: Mel spectrogram as a numpy array.
    """
    # Load the WAV file
    audio, sr = librosa.load(wav_path, sr=sample_rate)
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # # Convert to log scale (dB)
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram
    print(mel_spectrogram[0])

    print("torch!!!!!!!!!!!!!!!")
    print(torch.tensor(mel_spectrogram_db))
    # print(mel_spectrogram)
    
    return mel_spectrogram_db

def mel_spectrogram_to_wav(mel_spectrogram, output_path, n_mels=128, n_fft=2048, hop_length=512, sample_rate=22050):
    """
    Convert a Mel spectrogram back to a WAV file.
    
    Parameters:
    - mel_spectrogram: Input Mel spectrogram (in dB).
    - output_path: Path to save the output WAV file.
    - n_mels: Number of Mel bands (must match the one used in wav_to_mel_spectrogram).
    - n_fft: Length of FFT window.
    - hop_length: Number of samples between successive frames.
    - sample_rate: Sampling rate for the output audio.
    
    Returns:
    - None: Saves the reconstructed audio to output_path.
    """
    # # Convert dB back to power
    # mel_spectrogram_power = librosa.db_to_power(mel_spectrogram)
    
    # Reconstruct audio from Mel spectrogram using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Save the reconstructed audio to a WAV file
    sf.write(output_path, audio, sample_rate)

def main():
    """
    Main function to test WAV to Mel spectrogram conversion and back.
    """
    # Define input and output paths
    input_wav = "/Users/hudaili/Desktop/VsCodeProjects/TTS/data/LJSpeech-1.1/wavs/LJ001-0001.wav"  # Replace with your WAV file path
    output_wav = "output_reconstructed.wav"
    
    # Parameters for Mel spectrogram
    n_mels = 80
    n_fft = 2048
    hop_length = 512
    sample_rate = 22050
    
    try:
        # Check if input file exists
        if not os.path.exists(input_wav):
            raise FileNotFoundError(f"Input WAV file '{input_wav}' not found.")
        
        print(f"Converting {input_wav} to Mel spectrogram...")
        # Convert WAV to Mel spectrogram
        mel_spec = wav_to_mel_spectrogram(
            input_wav,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        
        print(f"Converting Mel spectrogram back to WAV and saving as {output_wav}...")
        # Convert Mel spectrogram back to WAV
        mel_spectrogram_to_wav(
            mel_spec,
            output_wav,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        print(f"Conversion complete! Output saved as {output_wav}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()