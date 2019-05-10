wget -O model_300.bin https://www.dropbox.com/s/yf70v0xu3offtcq/model_300.bin?dl=1
wget -O model_300.bin.trainables.syn1neg.npy https://www.dropbox.com/s/cc8ijzszk5l6gak/model_300.bin.trainables.syn1neg.npy?dl=1
wget -O model_300.bin.wv.vectors.npy https://www.dropbox.com/s/mca8t841sxerb3w/model_300.bin.wv.vectors.npy?dl=1
python3 hw6_test.py $1 $2 $3
