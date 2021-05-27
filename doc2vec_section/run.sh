python setup.py install
pip install -r requirements.txt
doc2vec ../dataset --save ../savesec_lr025.hdf5 --load_vocab ./vocab.vocab --train --embedding_size 100 --num_epochs 250 --model dmsec
doc2vec ../dataset --save ../savedm_lr025.hdf5 --load_vocab ./vocab.vocab --train --embedding_size 100 --num_epochs 250