# ADE
mkdir logs
CUDA_VISIBLE_DEVICES=1 python ade_trainer.py -s 0 -e 3 > logs/ade_basic_0.txt &
CUDA_VISIBLE_DEVICES=2 python ade_trainer.py -s 3 -e 6 > logs/ade_basic_1.txt &
CUDA_VISIBLE_DEVICES=3 python ade_trainer.py -s 6 -e 10 > logs/ade_basic_2.txt &
CUDA_VISIBLE_DEVICES=1 python ade_trainer.py -s 0 -e 3 -c with_external_knowledge > logs/ade_with_external_knowledge_0.txt &
CUDA_VISIBLE_DEVICES=2 python ade_trainer.py -s 3 -e 6 -c with_external_knowledge > logs/ade_with_external_knowledge_1.txt &
CUDA_VISIBLE_DEVICES=3 python ade_trainer.py -s 6 -e 10 -c with_external_knowledge > logs/ade_with_external_knowledge_2.txt
