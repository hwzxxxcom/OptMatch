dataset=$1
nnode=$2
queryno=$3

python -i match.py --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0021 --prelr 0.0018 \
                           --ln --lnnn --predictor cn1 --dataset $dataset  --epochs 100 --runs 10 --model puresum\
                           --hiddim 64 --mplayers 1  --use_xlin  --twolayerlin  --res  --maskinput --savemod\
                           --nnode $nnode --queryno $queryno --k $4