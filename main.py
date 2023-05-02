import argparse
import librosa
import numpy as np
import preprocess
import model

def main(args):
    
    X = []
    for acapella_file in args.acapella_files:
        audio, sr = librosa.load(acapella_file)
        X.append(preprocess.preprocess_audio(audio, sr))
    X = np.vstack(X)

  
    trained_model = model.train_model(X, epochs=args.epochs, batch_size=args.batch_size)
    trained_model.save(args.savedir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acapella_files', nargs='+', help='List of acapella audio file paths')
    parser.add_argument('--savedir', type=str, help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    main(args)
