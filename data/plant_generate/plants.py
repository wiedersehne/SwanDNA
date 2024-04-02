import sys
from torch.utils.data import Dataset

plant_path = '../'


def plant_feature(plant):
    if plant == 'ar':
        fasta_path = plant_path + 'plant_genome/Tair.fa'
        bed_path = plant_path + 'plant_data/download_data/ar/sorted_Dnase_rm_low_and_ATAC_all_peak_sorted.bed.gz'
        n_feature = 19
    elif plant == 'bd':
        fasta_path = plant_path + 'plant_genome/Bdistachyon.fasta'
        bed_path = plant_path + 'plant_data/download_data/bd/sorted_bd.bed.gz'
        n_feature = 9
    elif plant == 'mh':
        fasta_path = plant_path + 'plant_genome/MH63.fasta'
        bed_path = plant_path + 'plant_data/download_data/mh/sorted_mh.bed.gz'
        n_feature = 15
    elif plant == 'sb':
        fasta_path = plant_path + 'plant_genome/Sbicolor.fasta'
        bed_path = plant_path + 'plant_data/download_data/sb/sorted_sb.bed.gz'
        n_feature = 14
    elif plant == 'si':
        fasta_path = plant_path + 'plant_genome/Sitalica.fasta'
        bed_path = plant_path + 'plant_data/download_data/si/sorted_si.bed.gz'
        n_feature = 9
    elif plant == 'zm':
        fasta_path = plant_path + 'plant_genome/Zmays.fasta'
        bed_path = plant_path + 'plant_data/download_data/zm/sorted_zm.bed.gz'
        n_feature = 19
    elif plant == 'zs':
        fasta_path = plant_path + 'plant_genome/ZS97.fasta'
        bed_path = plant_path + 'plant_data/download_data/zs/sorted_zs97_15tissues.bed.gz'
        n_feature = 15
    else:
        print('no this plant')
        sys.exit()

    features_path = plant_path + 'plant_data/download_data/' + plant + '/distinct_features.txt'

    return fasta_path, bed_path, features_path, n_feature


def plant_bed(plant):
    if plant == 'ar':
        num_train = 5120000
        num_eval = 9984
    elif plant == 'bd':
        num_train = 5120000
        num_eval = 14848
    elif plant == 'mh':
        num_train = 5120000
        num_eval = 14848
    elif plant == 'sb':
        num_train = 5120000
        num_eval = 29952
    elif plant == 'si':
        num_train = 5120000
        num_eval = 19968
    elif plant == 'zm':
        num_train = 6400000
        num_eval = 79872
    elif plant == 'zs':
        num_train = 5120000
        num_eval = 14848
    else:
        print('no this plant')
        sys.exit()

    train_path = plant_path + 'plant_data/datasets_download/' + plant + '_L1000_train_' + str(num_train) + '.bed'
    val_path = plant_path + 'plant_data/datasets_download/' + plant + '_L1000_validate_' + str(num_eval) + '.bed'
    test_path = plant_path + 'plant_data/datasets_download/' + plant + '_L1000_test_' + str(num_eval) + '.bed'

    return train_path, val_path, test_path, num_train, num_eval
