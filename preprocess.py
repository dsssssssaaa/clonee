import os
import argparse
import librosa
import numpy as np
import zipfile

def preprocess_audio(audio, sr):
    '''
    Extracts features from an audio clip.
    
    Parameters:
        audio (ndarray): The audio clip.
        sr (int): The sample rate of the audio clip.
        
    Returns:
        ndarray: A feature vector that summarizes the audio clip.
    '''
    tempo, beat_frames = librosa.beat.beat_track(audio, sr=sr)
    chromagram = librosa.feature.chroma_cqt(audio, sr=sr)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    features = np.vstack((tempo, chromagram, mfcc)).T
    return features

if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Preprocess audio files')
    parser.add_argument('--audiozip', type=str, required=True, help='Path to zip file containing audio files')
    args = parser.parse_args()

    
    with zipfile.ZipFile(args.audiozip, 'r') as zip_ref:
        zip_ref.extractall('temp')

    
    for filename in os.listdir('temp'):
        if filename.endswith('.wav'):
            filepath = os.path.join('temp', filename)
            audio, sr = librosa.load(filepath)
            features = preprocess_audio(audio, sr)
            np.save(filepath[:-4] + '_features.npy', features)
            os.remove(filepath)

   
    os.rmdir('temp')
