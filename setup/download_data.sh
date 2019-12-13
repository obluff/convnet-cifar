if [ ! -d "data/" ]; then mkdir data; fi
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/input_data.tar.gz
tar -xvf data/input_data.tar.gz -C data