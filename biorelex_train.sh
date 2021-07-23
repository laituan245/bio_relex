# BioRelEx
mkdir logs
CUDA_VISIBLE_DEVICES=1 python trainer.py -s 0 > logs/biorelex_basic_0.txt &
CUDA_VISIBLE_DEVICES=2 python trainer.py -s 1 > logs/biorelex_basic_1.txt &
CUDA_VISIBLE_DEVICES=3 python trainer.py -s 2 > logs/biorelex_basic_2.txt &
CUDA_VISIBLE_DEVICES=1 python trainer.py -c with_external_knowledge -s 0 > logs/biorelex_with_external_knowledge_0.txt &
CUDA_VISIBLE_DEVICES=2 python trainer.py -c with_external_knowledge -s 1 > logs/biorelex_with_external_knowledge_1.txt &
CUDA_VISIBLE_DEVICES=3 python trainer.py -c with_external_knowledge -s 2 > logs/biorelex_with_external_knowledge_2.txt
