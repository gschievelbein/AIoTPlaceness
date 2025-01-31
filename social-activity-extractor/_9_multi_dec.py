import argparse

import numpy as np
import pandas as pd

import config
import requests
import json

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import util
from sklearn.model_selection import train_test_split, StratifiedKFold

from model.Weight_Calculator import WeightCalculator
from model.util import load_multi_csv_data, load_semi_supervised_csv_data, load_transductive_semi_supervised_csv_data, \
    load_full_csv_data
from model.multidec import MDEC_encoder, MultiDEC

CONFIG = config.Config


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=1e-02, help='initial learning rate')
    parser.add_argument('-kappa', type=float, default=0.2, help='lr adjust rate')
    parser.add_argument('-tol', type=float, default=1e-03, help='tolerance for early stopping')
    parser.add_argument('-es', action='store_true', default=False, help='early stops when unsupervised loss increases')
    parser.add_argument('-trade_off', type=float, default=1e-04, help='trade_off value for semi-supervised learning')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train')
    parser.add_argument('-update_time', type=int, default=1, help='update time within epoch')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for training')
    # data
    parser.add_argument('-prefix_csv', type=str, default=None, help='prefix of csv name')
    parser.add_argument('-target_dataset', type=str, default=None, help='file name of target csv')
    parser.add_argument('-n_clusters', type=int, default=12, help='num of class')
    parser.add_argument('-image_csv', type=str, default=None,
                        help='file name of target csv')
    parser.add_argument('-text_csv', type=str, default=None, help='file name of target csv')
    parser.add_argument('-label_csv', type=str, default='category_label.csv', help='file name of target labels')
    parser.add_argument('-sampled_n', type=int, default=None, help='number of fold')
    # model
    parser.add_argument('-prefix_model', type=str, default=None, help='prefix of csv name')
    parser.add_argument('-img_dim', type=int, default=300, help='size of image input dimension')
    parser.add_argument('-txt_dim', type=int, default=300, help='size of text input dimension')
    parser.add_argument('-latent_dim', type=int, default=10, help='size of latent variable')
    parser.add_argument('-ours', action='store_true', default=False, help='use our target distribution')
    parser.add_argument('-use_prior', action='store_true', default=False, help='use prior knowledge')
    # train
    parser.add_argument('-start_fold', type=int, default=0, help='fold for start')
    parser.add_argument('-fold', type=int, default=5, help='number of fold')
    parser.add_argument('-trans', action='store_true', default=False, help='transductive learning')
    parser.add_argument('-trans_csv', type=str, default='0.02_category_label.csv',
                        help='file name of target transductive label')
    parser.add_argument('-ssldec', action='store_true', default=False, help='use if ssldec')
    parser.add_argument('-tsne', action='store_true', default=False, help='whether to print tsne result')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    # option
    parser.add_argument('-eval', action='store_true', default=False, help='whether evaluate or train it')

    args = parser.parse_args()

    if args.eval:
        eval_multidec(args)
    elif args.trans:
        train_multidec_transductive(args)
    else:
        train_multidec(args)


def train_multidec(args):
    print("Training multidec")
    device = torch.device(args.gpu)

    df_image_data = pd.read_csv(os.path.join(CONFIG.IMAGE_EMBEDDINGS,
                                             args.prefix_csv + '_' + args.image_csv + '_' +
                                             args.target_dataset + ".csv"),
                                index_col=0,
                                encoding='utf-8-sig')
    df_text_data = pd.read_csv(os.path.join(CONFIG.TEXT_EMBEDDINGS,
                                            args.prefix_csv + '_' + args.text_csv + '_' + args.target_dataset + ".csv"),
                               index_col=0,
                               encoding='utf-8-sig')

    df_label = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.label_csv), index_col=0,
                           encoding='utf-8-sig')
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1
    # n_clusters = args.n_clusters

    acc_list = []
    nmi_list = []
    f_1_list = []
    kf_count = 0
    for fold_idx in range(args.start_fold, args.fold):
        print("Current fold: ", fold_idx)
        df_train = pd.read_csv(
            os.path.join(CONFIG.CSV_PATH, "train_" + str(fold_idx) + "_" + args.target_dataset + "_label.csv"),
            index_col=0,
            encoding='utf-8-sig')
        if args.sampled_n is not None:
            df_train = df_train.sample(n=args.sampled_n, random_state=42)
        df_test = pd.read_csv(
            os.path.join(CONFIG.CSV_PATH, "test_" + str(fold_idx) + "_" + args.target_dataset + "_label.csv"),
            index_col=0,
            encoding='utf-8-sig')
        print("Loading full dataset...")
        full_dataset, train_dataset, val_dataset = load_semi_supervised_csv_data(df_image_data, df_text_data, df_train,
                                                                                 df_test, CONFIG)
        print("\nLoading dataset completed")

        image_encoder = MDEC_encoder(input_dim=args.img_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                     encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
        image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, 'images',
                                              args.prefix_model + "_image_" + args.target_dataset + "_sdae_" + str(
                                                  args.latent_dim) + '_' + str(fold_idx)) + ".pt")
        # image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "sampled_plus_labeled_scaled_image_sdae_" + str(fold_idx)) + ".pt")
        text_encoder = MDEC_encoder(input_dim=args.txt_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                    encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
        text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, 'text',
                                             args.prefix_model + "_use_" + args.target_dataset + '_' +
                                             str(args.txt_dim) + "_sdae_" + str(args.latent_dim) + '_' +
                                             str(fold_idx)) + ".pt")
        # text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "sampled_plus_labeled_scaled_text_sdae_" + str(fold_idx)) + ".pt")
        mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder, ours=args.ours,
                        use_prior=args.use_prior,
                        n_clusters=n_clusters)

        if args.ssldec:
            mdec.fit_predict_ssldec(full_dataset, train_dataset, val_dataset, args, CONFIG, lr=args.lr,
                                    batch_size=args.batch_size, num_epochs=args.epochs,
                                    save_path=os.path.join(CONFIG.CHECKPOINT_PATH, args.prefix_csv + "_mdec_" +
                                                           str(args.latent_dim) + '_' + str(fold_idx)) + ".pt",
                                    tol=args.tol, kappa=args.kappa)
        else:
            mdec.fit_predict(full_dataset, train_dataset, val_dataset, args, CONFIG, lr=args.lr,
                             batch_size=args.batch_size, num_epochs=args.epochs,
                             save_path=os.path.join(CONFIG.CHECKPOINT_PATH, 'mdec', args.prefix_csv + "_mdec_" +
                                                    str(args.txt_dim) + '_' + str(args.latent_dim) + '_' +
                                                    str(fold_idx)) + '_' + str(args.kappa) + ".pt", tol=args.tol,
                             kappa=args.kappa)

        acc_list.append(mdec.acc)
        nmi_list.append(mdec.nmi)
        f_1_list.append(mdec.f_1)
        kf_count = kf_count + 1
        print("#Average acc: %.4f, Average nmi: %.4f, Average f_1: %.4f" % (
            np.mean(acc_list), np.mean(nmi_list), np.mean(f_1_list)))


def train_multidec_transductive(args):
    print("Training multidec")
    device = torch.device(args.gpu)
    df_image_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.prefix_csv + "_" + args.image_csv), index_col=0,
                                encoding='utf-8-sig')
    df_text_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.prefix_csv + "_" + args.text_csv), index_col=0,
                               encoding='utf-8-sig')

    df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1

    exp = Experiment(args.prefix_csv + "_MDEC", capture_io=True)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:

        df_train = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.trans_csv),
                               index_col=0,
                               encoding='utf-8-sig')
        print("Loading dataset...")
        full_dataset, train_dataset = load_transductive_semi_supervised_csv_data(df_image_data, df_text_data, df_label,
                                                                                 df_train, CONFIG)
        print("\nLoading dataset completed")

        image_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                     encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
        image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH,
                                              args.prefix_model + "_image" "_" + args.target_dataset + "_sdae_" + str(
                                                  args.latent_dim) + "_all.pt"))
        # image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "sampled_plus_labeled_scaled_image_sdae_" + str(fold_idx)) + ".pt")
        text_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                    encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
        text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH,
                                             args.prefix_model + "_text""_" + args.target_dataset + "_sdae_" + str(
                                                 args.latent_dim) + "_all.pt"))
        # text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "sampled_plus_labeled_scaled_text_sdae_" + str(fold_idx)) + ".pt")
        mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder, ours=args.ours,
                        use_prior=args.use_prior,
                        n_clusters=n_clusters)

        if args.ssldec:
            mdec.fit_predict_transductive_ssldec(full_dataset, train_dataset, args, CONFIG, lr=args.lr,
                                                 batch_size=args.batch_size, num_epochs=args.epochs,
                                                 save_path=os.path.join(CONFIG.CHECKPOINT_PATH,
                                                                        args.prefix_csv + "_mdec_" +
                                                                        str(args.latent_dim) + "_all.pt"), tol=args.tol,
                                                 kappa=args.kappa)
        else:
            mdec.fit_predict_transductive(full_dataset, train_dataset, args, CONFIG, lr=args.lr,
                                          batch_size=args.batch_size, num_epochs=args.epochs,
                                          save_path=os.path.join(CONFIG.CHECKPOINT_PATH,
                                                                 args.prefix_csv + "_mdec_" +
                                                                 str(args.latent_dim) + "_all.pt"), tol=args.tol,
                                          kappa=args.kappa)
        print("#Average acc: %.4f, Average nmi: %.4f, Average f_1: %.4f" % (
            mdec.acc, mdec.nmi, mdec.f_1))

    finally:
        exp.end()


def eval_multidec(args):
    print("Evaluate multidec")
    device = torch.device(args.gpu)
    print("Loading dataset...")
    df_image_data = pd.read_csv(os.path.join(CONFIG.IMAGE_EMBEDDINGS,
                                             args.prefix_csv + '_' + args.image_csv + '_' +
                                             args.target_dataset + ".csv"),
                                index_col=0,
                                encoding='utf-8-sig')
    df_text_data = pd.read_csv(os.path.join(CONFIG.TEXT_EMBEDDINGS,
                                            args.prefix_csv + '_' + args.text_csv + '_' + args.target_dataset + ".csv"),
                               index_col=0,
                               encoding='utf-8-sig')

    df_label = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.label_csv), index_col=0,
                           encoding='utf-8-sig')
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1
    print("Loading dataset completed")
    print("Joining datasets...")
    full_dataset = load_full_csv_data(df_image_data, df_text_data, CONFIG)
    del df_text_data
    del df_image_data
    del label_array

    image_encoder = MDEC_encoder(input_dim=args.img_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                 encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, 'images',
                                          args.prefix_model + "_image_" + args.target_dataset + "_sdae_" +
                                          str(args.latent_dim) + '_' + str(0)) + ".pt")
    text_encoder = MDEC_encoder(input_dim=args.txt_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, 'text',
                                         args.prefix_model + "_use_" + args.target_dataset + '_' +
                                         str(args.txt_dim) + "_sdae_" + str(args.latent_dim) + '_' +
                                         str(0)) + ".pt")
    mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder, n_clusters=n_clusters)
    mdec.load_model(
        os.path.join(CONFIG.CHECKPOINT_PATH, 'mdec', args.prefix_model + "_mdec_" + str(args.txt_dim) + '_' +
                     str(args.latent_dim) + '_0_' + str(args.kappa) + ".pt"))
    print('Predicting...')
    short_codes, y_pred, y_confidence, pvalue = mdec.predict(full_dataset, args.batch_size)
    print('Writing csv...')
    result_df = pd.DataFrame(data={'cluster_id': y_pred, 'confidence': y_confidence}, index=short_codes)
    result_df.index.name = "short_code"
    result_df.sort_index(inplace=True)
    result_df.to_csv(
        os.path.join(CONFIG.RESULT_PATH, 'multidec_result_' + args.prefix_csv + '_' + str(args.txt_dim) + '.csv'),
        encoding='utf-8-sig')

    pvalue_df = pd.DataFrame(data=pvalue, index=short_codes, columns=[str(i) for i in range(n_clusters)])
    pvalue_df.index.name = "short_code"
    pvalue_df.sort_index(inplace=True)
    pvalue_df.to_csv(
        os.path.join(CONFIG.RESULT_PATH, 'multidec_pvalue_' + args.prefix_csv + '_' + str(args.txt_dim) + '.csv'),
        encoding='utf-8-sig')
    print('Finished :)')


if __name__ == '__main__':
    main()
