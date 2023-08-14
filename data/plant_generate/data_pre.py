import os
import torch
import argparse
import numpy as np

from cython_data import GenomeSequence, GenomicFeatures
from cython_file import BedFileSampler
from plants import plant_feature, plant_bed


def classify_main(num_train, length):
    data_dir = '../plant_pt/'
    os.makedirs(data_dir, exist_ok=True)

    add_len = int((length - 1000) / 2)
    pre_name = '0pre' + str(length) + '_tr' + str(num_train*7)

    pre = []
    for plant in ['ar', 'bd', 'mh', 'sb', 'si', 'zm', 'zs']:
    # for plant in ['ar', 'ar', 'ar', 'ar', 'ar', 'ar', 'ar']:
        fasta_path, bed_path, features_path, n_feature = plant_feature(plant)
        genome_sequence = GenomeSequence(fasta_path)
        genome_features = GenomicFeatures(bed_path, features_path)
        train_path, val_path, test_path, _, num_eval = plant_bed(plant)

        train_sampler = BedFileSampler(train_path, genome_sequence, genome_features)
        train_sequences, _ = train_sampler.sample(num_train, add_len)
        print(train_sequences.shape)
        pre.append(train_sequences)

    all_tr = np.concatenate(pre)
    print(all_tr.shape)
    torch.save(all_tr, data_dir + pre_name + '_seq.pt', pickle_protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('-num_train', type=int, default=30000)
    parser.add_argument('-length', type=int, default=1000)
    args = parser.parse_args()

    classify_main(args.num_train, args.length)
