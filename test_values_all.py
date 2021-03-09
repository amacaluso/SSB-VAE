import numpy as np
import sys
from optparse import OptionParser
from test_models_20news import test20news
from test_models_TMC import testtmc
from test_models_cifar import testcifar
from test_models_snippets import testsnippets
from utils import obtain_parameters

op = OptionParser()
op.add_option("-M", "--model", type=int, default=3, help="model type (1,2,3)")
op.add_option("-r", "--repetitions", type=int, default=2, help="repetitions")
op.add_option("-s", "--reseed", type=int, default=0, help="if >0 reseed numpy for each repetition")
op.add_option("-v", "--addvalidation", type=int, default=1, help="if >0 add the validation set to the train set")
op.add_option("-c", "--nbits", type=int, default=16, help="number of bits")
op.add_option("-d", "--ds", type="string", default="20news", help="Dataset to train: 20news, cifar, tmc, snippets")

(opts, args) = op.parse_args()
nbits = opts.nbits
df = str(opts.ds).lower()
supervised_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
model_dict = {1:"VDHS-S", 2: "PHS-GS", 3:"SSBVAE" }
model = model_dict.get(opts.model)

header = "test"+df

for level in supervised_levels:
    alphaVal, betaVal, lambdaVal = obtain_parameters(level, df, nbits)
    for alpha in alphaVal:
        for beta in betaVal:
            for lambda_ in lambdaVal:
                print("TESTING " + df.upper() + " @Level" + str(level))
                print("Alpha: ", alpha, " Beta: ", beta, " Lambda :", lambda_)

                ofile = "\"./Results/ResultsTraning/"+ model +"_"+df.upper()+"-"+str(nbits)+"BITS-"+\
                        str(alpha)+"ALPHA-"+str(beta)+"BETA-"+str(lambda_)+"LAMBDA.csv\""

                tail = "(model="+str(opts.model)+",ps="+str(level)+",addvalidation="+str(opts.addvalidation)+\
                       ",alpha="+str(alpha)+",beta="+str(beta)+",lambda_="+str(lambda_)+",repetitions="+str(opts.repetitions)+",nbits="+\
                       str(nbits)+",ofilename="+ofile+")"
                func = header + tail
                eval(func)






