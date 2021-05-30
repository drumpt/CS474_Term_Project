python3 setup.py install
pip3 install -r requirements.txt
doc2vec ../../dataset --save ../savesec_lr025.hdf5 --load_vocab ./vocab.vocab --train --embedding_size 256 --num_epochs 100 --model dmsec
doc2vec ../../dataset --save ../savedm_lr025.hdf5 --load_vocab ./vocab.vocab --train --embedding_size 256 --num_epochs 100