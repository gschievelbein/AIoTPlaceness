import argparse
import config
import requests
import json
import pickle
import datetime
import os
import csv
import math
import numpy as np
import pandas as pd
import _pickle as cPickle
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.similarities.index import AnnoyIndexer
from hyperdash import Experiment
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from model import util
from model import text_model, imgseq_model, text_model

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

CONFIG = config.Config


def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# data
	parser.add_argument('-target_dataset', type=str, default=None, help='folder name of target dataset')
	args = parser.parse_args()

	get_latent(args)


def get_latent(args):
	print("Loading embedding model...")
	model_name = 'DOC2VEC_' + args.target_dataset + '.model'
	embedding_model = Doc2Vec.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("Loading embedding model completed")
	
	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
						  encoding='utf-8-sig')

	short_code_list = []
	row_list = []
	csv_name = 'text_doc2vec_' + args.target_dataset + '.csv'
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		short_code = row.iloc[0]
		short_code_list.append(short_code)
		text_data = row.iloc[1]
		vector = embedding_model.infer_vector(text_data.split())
		row_list.append(vector)
		del text_data
	pbar.close()

	result_df = pd.DataFrame(data=row_list, index=short_code_list, columns=[i for i in range(embedding_model.vector_size)])
	result_df.index.name = "short_code"
	result_df.sort_index(inplace=True)
	result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')
	print("Finish!!!")


if __name__ == '__main__':
	main()
