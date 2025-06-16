mkdir -p dataset
git clone https://github.com/gmichalo/LexSubCon.git
find LexSubCon/dataset -type f -exec cp {} dataset/ \;
rm -rf LexSubCon
rm dataset/.keep
