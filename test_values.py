import numpy as np
import sys
from optparse import OptionParser

from utils import obtain_parameters
from test_models_20news import test20news
from test_models_cifar import testcifar
from test_models_snippets import testsnippets
from test_models_TMC import testtmc

op = OptionParser()
op.add_option("-M", "--model", type=int, default=3, help="model type (1,2,3)")
op.add_option("-p", "--ps", type=float, default=1.0, help="supervision level (float[0.1,1.0])")
op.add_option("-a", "--alpha", type=float, default=0.0, help="alpha value")
op.add_option("-b", "--beta", type=float, default=0.0, help="beta value")
op.add_option("-l", "--lambda_", type=float, default=0.0, help="lambda value")
op.add_option("-r", "--repetitions", type=int, default=2, help="repetitions")
op.add_option("-s", "--reseed", type=int, default=0, help="if >0 reseed numpy for each repetition")
op.add_option("-v", "--addvalidation", type=int, default=1, help="if >0 add the validation set to the train set")
op.add_option("-c", "--nbits", type=int, default=16, help="number of bits")
op.add_option("-d", "--ds", type="string", default="20news", help="Dataset to train: 20news, cifar, tmc, snippets")

(opts, args) = op.parse_args()
ps = float(opts.ps)
nbits = opts.nbits
df = str(opts.ds).lower()
model_dict = {1:"VDHS-S", 2: "PHS-GS", 3:"SSBVAE" }
model = model_dict.get(opts.model)

print("TESTING " + df.upper() +" with model " + str(opts.model))
print("Alpha: ", opts.alpha, " Beta: ", opts.beta, " Lambda :", opts.lambda_)

header = "test"+df
ofile = "\"./Results/ResultsTraning/" + model + "_" + df.upper() + "-" + str(nbits) + "BITS-" + \
        str(opts.alpha) + "ALPHA-" + str(opts.beta) + "BETA-" + str(opts.lambda_) + "LAMBDA.csv\""
print(ofile)
tail = "(model="+str(opts.model)+",ps="+str(opts.ps)+", addvalidation="+str(opts.addvalidation)+",alpha="+str(opts.alpha)+\
       ",beta="+str(opts.beta)+",lambda_="+str(opts.lambda_)+",repetitions="+str(opts.repetitions)+",nbits="+str(nbits)+\
       ",ofilename="+ofile+")"
func = header + tail
eval(func)
