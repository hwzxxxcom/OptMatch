#!/bin/bash

name=${1}
nnode=${2}
noquery=${3}
k=${4}
if [ "$name" == "yeast" ]; then
    nv=3112
    ne=12519
    nl=72
elif [ "$name" == "ciwiki" ]; then
    nv=5201
    ne=198353
    nl=14
elif [ "$name" == "wordnet" ]; then
    nv=40559
    ne=71925
    nl=16
elif [ "$name" == "ciyoutube" ]; then
    nv=1134890
    ne=1987624
    nl=25
elif [ "$name" == "chciteseer" ]; then
    nv=3264
    ne=4536
    nl=6
elif [ "$name" == "dblp" ]; then
    nv=317080
    ne=1049866
    nl=15
elif [ "$name" == "chyago" ]; then
    nv=12811197
    ne=15835677
    nl=40
fi

python -i match.py \
    --queryno $noquery\
    --dataname $name\
    --nnode $nnode\
    --pattern_dir /home/nagy/data/SubgraphCounting2/$name$nnode/patterns \
    --graph_dir /home/nagy/data/SubgraphCounting2/$name$nnode/graphs \
    --metadata_dir /home/nagy/data/SubgraphCounting2/$name$nnode/metadata \
    --save_data_dir ./data/$name$nnode/datasets \
    --save_model_dir ./dumps/$name$nnode \
    --add_rev True \
    --hid_dim 64 --node_pred True --edge_pred False \
    --match_weights node \
    --enc_net Position --enc_base 2 \
    --emb_net Equivariant --share_emb_net True \
    --rep_net DMPNN \
    --rep_num_pattern_layers 3 --rep_num_graph_layers 3 \
    --rep_residual True --rep_dropout 0.0 --share_rep_net True \
    --pred_net SumPredictNet --pred_hid_dim 64 --pred_dropout 0.0 \
    --max_npv ${nnode} --max_npe 100 --max_npvl ${nl} --max_npel ${nl} \
    --max_ngv ${nv} --max_nge ${ne} --max_ngvl ${nl} --max_ngel ${nl} \
    --train_grad_steps 1 --train_batch_size 64 \
    --train_log_steps 10 --eval_batch_size 64 \
    --lr 1e-3 --train_epochs 100 \
    --seed 0 --gpu_id -1 \
    --eval_metric SMSE --bp_loss SMSE\
    --k $k
