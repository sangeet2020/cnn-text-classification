#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
<Function of script>
"""

import os
import sys
import time
import argparse
import numpy as np
from dataset_loader import DataLoader
from parameters import Parameters
from model import TextClassifier
from utils import Run, DEVICE


def prepare_data():
    """ purpose of my function """
    dl = DataLoader(max_seq_len=Parameters.max_seq_len)
    dl.load_data()
    dl.get_targets()
    dl.get_sentences()
    dl.get_id2target()
    dl.targets2one_hot()
    dl.tokenize()
    dl.build_vocabulary()
    dl.padding()
    dl.split_data()
    
    return (
        {
            'x_train': dl.X_train,
            'y_train': dl.y_train,
            'x_test': dl.X_test,
            'y_test': dl.y_test
        },
        dl.t_words
    )

def main():
    """ main method """
    # args = parse_arguments()
    # os.makedirs(args.out_dir, exist_ok=True)
    
    # Prepare the data
    print("Loading dataset....")
    data, t_words = prepare_data()
    print("--Done--")
    
    # Initialize the model
    start = time.time()
    model = TextClassifier(t_words, Parameters)
    model.to(DEVICE)
    
    # Train and Evaluate the pipeline
    Run().train(model, data, Parameters)
    end = time.time()
    print("*** Training Complete ***")
    print("Runtime: {:.2f} s".format(end-start))

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arg_1", help="describe arg_1")
    parser.add_argument("arg_2", help="describe arg_2")
    parser.add_argument("-optional_arg", default=23, type=int, help='optional_arg meant for some purpose')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()