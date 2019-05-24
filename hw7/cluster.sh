wget -O kmeans_1.pkl https://www.dropbox.com/s/s4l7taw9d4q3mu3/kmeans_1.pkl?dl=1
wget -O autoencoder_epoch100.pth https://www.dropbox.com/s/qgwx0l5fxdt0lnk/autoencoder_epoch100.pth?dl=1
python cluster.py $1 $2 $3
