import sys
import numpy as np
from abc import ABCMeta
from _cython_genome import _fast_get_feature_data_thresholds


def _define_feature_thresholds(feature_thresholds, features):
    feature_thresholds_vec = np.zeros(len(features))
    if 0 < feature_thresholds <= 1:
        feature_thresholds_vec += feature_thresholds
    else:
        print('wrong thresholds: {}'.format(feature_thresholds))
        sys.exit()
    return feature_thresholds_vec.astype(np.float32)


class BedFileSampler(metaclass=ABCMeta):
    def __init__(self, filepath, genome_sequence, genome_features):
        super(BedFileSampler, self).__init__()
        self.filepath = filepath
        self._file_handle = open(self.filepath, 'r')
        self.genome_sequence = genome_sequence
        self.genome_features = genome_features

    def sample(self, num_train, add_len, center=200, thresholds=0.5):
        sequences = []
        targets = []

        while len(sequences) < num_train:
            line = self._file_handle.readline()
            if not line:
                self._file_handle.close()
                self._file_handle = open(self.filepath, 'r')
                line = self._file_handle.readline()
            cols = line.split('\n')[0].split('\t')

            chrom = cols[0]
            start = int(cols[1]) - add_len
            end = int(cols[2]) + add_len
            strand = cols[3]
            # print(chrom, start, end, strand)

            # sequence = self.genome_sequence.sequence_from_coords(chrom, start, end, strand)
            sequence = self.genome_sequence.encoding_from_coords(chrom, start, end, strand)
            # print(sequence)

            new_start = int((end - start - center) / 2)
            target_position = self.genome_features.feature_data(chrom, start, end)
            targets_center = target_position[new_start:new_start + center, :]
            feature_thresholds_vec = _define_feature_thresholds(thresholds, self.genome_features.feature_index_dict)
            target = _fast_get_feature_data_thresholds(targets_center, feature_thresholds_vec, center)

            sequences.append(sequence)
            targets.append(target.astype(float))

        sequences = np.array(sequences)
        targets = np.array(targets)
        return sequences, targets

