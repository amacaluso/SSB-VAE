#TEST VDSH-S WITH DIFFERENT SUPERVISION LEVELS FOR 16 BITS

python test_models_20news.py -M 1 -p 0.1 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.2 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.3 -v 1 -a 100000 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.4 -v 1 -a 10.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.5 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.6 -v 1 -a 10.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.7 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.8 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 0.9 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'
python test_models_20news.py -M 1 -p 1.0 -v 1 -a 1.0 -b 0.06250 -g 0.0 -r 5 -l 32  -o 'VDSH_20NEWS-32BITS.csv'

#TEST PSH-GS WITH DIFFERENT SUPERVISION LEVELS FOR 16 BITS

python test_models_20news.py -M 2 -p 0.1 -v 1 -a 0.0001 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.2 -v 1 -a 0.0001 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.3 -v 1 -a 0.001 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.4 -v 1 -a 0.001 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.5 -v 1 -a 0.01 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.6 -v 1 -a 0.01 -b 0.015625 -g 10000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.7 -v 1 -a 10.0 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.8 -v 1 -a 1000.0 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 0.9 -v 1 -a 0.01 -b 0.015625 -g 100000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
python test_models_20news.py -M 2 -p 1.0 -v 1 -a 1.0 -b 0.015625 -g 1000 -r 5 -l 32 -o 'PHS_20NEWS-32BITS.csv'
 
#TEST SSB-VAE WITH DIFFERENT SUPERVISION LEVELS FOR 16 BITS

python test_models_20news.py -M 3 -p 0.1 -v 1 -a 0.001 -b 0.015625 -g 1000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.2 -v 1 -a 0.001 -b 0.015625 -g 1000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.3 -v 1 -a 0.001 -b 0.015625 -g 100000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.4 -v 1 -a 0.001 -b 0.015625 -g 1000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.5 -v 1 -a 100000 -b 0.015625 -g 100000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.6 -v 1 -a 0.1 -b 0.015625 -g 100000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.7 -v 1 -a 100 -b 0.015625 -g 1000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.8 -v 1 -a 1.0 -b 0.015625 -g 10000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 0.9 -v 1 -a 1.0 -b 0.015625 -g 10000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
python test_models_20news.py -M 3 -p 1.0 -v 1 -a 0.01 -b 0.015625 -g 1000 -r 5 -l 32 -o 'SSBVAE_20NEWS-32BITS.csv'
