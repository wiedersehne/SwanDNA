import os
import torch
import argparse

from cython_data import GenomeSequence, GenomicFeatures
from cython_file import BedFileSampler
from plants import plant_feature, plant_bed


def classify_main(plant, num_train, length):
    data_dir = '../plant_pt/'
    os.makedirs(data_dir, exist_ok=True)

    add_len = int((length - 1000) / 2)

    # data
    fasta_path, bed_path, features_path, n_feature = plant_feature(plant)
    genome_sequence = GenomeSequence(fasta_path)
    genome_features = GenomicFeatures(bed_path, features_path)
    train_path, val_path, test_path, _, num_eval = plant_bed(plant)

    train_sampler = BedFileSampler(train_path, genome_sequence, genome_features)
    train_sequences, train_targets = train_sampler.sample(num_train, add_len)
    print(train_sequences.shape, train_targets.shape)
    tr_name = plant + str(length) + '_tr' + str(num_train)
    torch.save(train_sequences, data_dir + tr_name + '_seq.pt', pickle_protocol=4)
    torch.save(train_targets, data_dir + tr_name + '_target.pt', pickle_protocol=4)

    # val_sampler = BedFileSampler(val_path, genome_sequence, genome_features)
    # val_sequences, val_targets = val_sampler.sample(num_eval, add_len)
    # print(val_sequences.shape, val_targets.shape)

    test_sampler = BedFileSampler(test_path, genome_sequence, genome_features)
    test_sequences, test_targets = test_sampler.sample(num_eval, add_len)
    print(test_sequences.shape, test_targets.shape)
    te_name = plant + str(length) + '_te' + str(num_eval)
    torch.save(test_sequences, data_dir + te_name + '_seq.pt', pickle_protocol=4)
    torch.save(test_targets, data_dir + te_name + '_target.pt', pickle_protocol=4)

    # train_sequences = torch.load(save_dir + tr_name + '_seq.pt')
    # train_targets = torch.load(save_dir + tr_name + '_target.pt')
    # test_sequences = torch.load(save_dir + te_name + '_seq.pt')
    # test_targets = torch.load(save_dir + te_name + '_target.pt')
    # print(train_sequences.shape, train_targets.shape)
    # print(test_sequences.shape, test_targets.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('-plant', type=str, default='ar')  # ar bd mh sb si zm zs
    parser.add_argument('-num_train', type=int, default=100000)
    parser.add_argument('-length', type=int, default=1000)
    args = parser.parse_args()

    classify_main(args.plant, args.num_train, args.length)
