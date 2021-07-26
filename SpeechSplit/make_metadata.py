import os
import pickle
import numpy as np
from hparams import hparams
from audioRead import get_id
import random
import math
# rootDir = 'assets/spmel'

def main(rootDir, train_proportion):
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    speaker_ids = pickle.load(open("assets/spk2gen.pkl", "rb"))

    train_speakers, val_speakers = [], []
    for speaker in sorted(subdirList):
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        num_train = math.floor(len(fileList) * train_proportion)
        print('Processing speaker: %s' % speaker)
        print(f"Train samples: {num_train}")
        print(f"Validation samples: {len(fileList)-num_train}")
        train_uttrs = []
        train_uttrs.append(speaker)

        # may use generalized speaker embedding for zero-shot conversion
        embedding = get_id(speaker, speaker_ids)
        train_uttrs.append(embedding)
        test_uttrs = train_uttrs[:]
        random.shuffle(fileList)
        for i, fileName in enumerate(fileList):
            if i <= num_train:
                train_uttrs.append(os.path.join(speaker,fileName))
            else:
                test_uttrs.append(os.path.join(speaker,fileName))
        train_speakers.append(train_uttrs)
        val_speakers.append(test_uttrs)
        
    with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
        pickle.dump(train_speakers, handle)    
    with open(os.path.join(rootDir, 'validation.pkl'), 'wb') as handle:
        pickle.dump(val_speakers, handle)    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='assets/spmel')
    parser.add_argument('--train_proportion', default=0.99, type=float)
    args = parser.parse_args()
    main(args.root_dir, args.train_proportion)