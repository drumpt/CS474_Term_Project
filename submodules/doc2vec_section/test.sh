doc2vec ../../dataset --train --embedding_size 256 --num_epochs 150 --model dm --json_num=8 --load_vocab ./total_vocab.vocab --save ./total_150_dm.hdf5
doc2vec ../../dataset --train --embedding_size 256 --num_epochs 150 --model dmsec --json_num=8 --load_vocab ./total_vocab.vocab --save ./total_150_dmsec.hdf5
word2vec ../../dataset --train --embedding_size 256 --num_epochs 150 --model cbow --json_num=8 --load_vocab ./total_vocab.vocab --save ./total_150_cbow.hdf5
word2vec ../../dataset --train --embedding_size 256 --num_epochs 150 --model cbsec --json_num=8 --load_vocab ./total_vocab.vocab --save ./total_150_cbsec.hdf5
