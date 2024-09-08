#!/bin/bash

cd reordering

echo "run_reorderings"

#echo ogbn-products
#bash run_reorderings.sh ogbn-products

echo ogbn-arxiv
bash run_reorderings.sh ogbn-arxiv
