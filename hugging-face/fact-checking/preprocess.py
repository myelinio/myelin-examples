dailymail_stories = "https://drive.google.com/uc?export=download&confirm=pra0&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"
cnn_stories = "https://drive.google.com/uc?export=download&confirm=eMAz&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"

"""
# Pull and install Huggingface Transformers Repo
git clone https://github.com/huggingface/transformers && cd transformers
pip install .
pip install nltk py-rouge
cd examples/summarization

#------------------------------
# Download original Summarization Datasets. The code downloads from Google drive on Linux
wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=<CONFIRMATION CODE HERE>&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O cnn_stories.tgz

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=<CONFIRMATION CODE HERE>&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -O dailymail_stories.tgz

# Unzip & untar the files
tar -xvf cnn_stories.tgz && tar -xvf dailymail_stories.tgz
rm cnn_stories.tgz dailymail_stories.tgz

# Move the articles to a single location
mkdir bertabs/dataset
mkdir bertabs/summaries_out
cp -r bertabs/cnn/stories dataset
cp -r bertabs/dailymail/stories dataset

# Select a subset of articles to summarize
mkdir bertabs/dataset2
cd bertabs/dataset && find . -maxdepth 1 -type f | head -1000 | xargs cp -t ../dataset2/


"""