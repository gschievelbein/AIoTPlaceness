from tqdm import tqdm

import config
import argparse
import os

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import pandas as pd


CONFIG = config.Config


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # data
    parser.add_argument('--target_dataset', type=str, default=None, help='folder name of target dataset')
    parser.add_argument('--model_directory', type=str, default=None,
                        help='directory the pretrained sentence embedding model')
    parser.add_argument('--target_csv', type=str, default='posts.csv', help='filename of target dataset')
    args = parser.parse_args()

    embed_sentences(args)


def embed_sentences(args):
    print("Loading sentence embedding model...")
    model = tf.saved_model.load(os.path.join(CONFIG.EMBEDDING_PATH, args.model_directory))
    #use_model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    #model = hub.load(use_model_url)
    print("Loading sentence embedding model completed!")

    print("Loading dataframe...")
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.target_csv), header=None,
                          encoding='utf-8-sig')
    print("Loading dataframe completed!")

    short_code_list = []
    row_list = []
    csv_name = 'text_se_' + args.target_dataset + '.csv'
    pbar = tqdm(total=df_data.shape[0])

    print("Embedding sentences...")
    for index, row in df_data.iterrows():
        pbar.update(1)
        short_code = row.iloc[0]
        short_code_list.append(short_code)
        text_data = row.iloc[1]
        vector = model(text_data).numpy()[0]
        row_list.append(vector)
        del text_data
    pbar.close()
    print("Embedding sentences completed!!!")

    print("Creating dataframe...")
    result_df = pd.DataFrame(data=row_list, index=short_code_list,
                             columns=[i for i in range(row_list[0].shape[0])])
    result_df.index.name = "short_code"
    result_df.sort_index(inplace=True)
    result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')
    print("Dataframe saved to: " + os.path.join(CONFIG.CSV_PATH, csv_name))
    print("Finished :)")


if __name__ == '__main__':
    main()
