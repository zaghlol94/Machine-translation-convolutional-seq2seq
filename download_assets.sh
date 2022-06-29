python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
gdown "17OPCWbWiMacOX86Sh5gysDj-sat1gHvj"
mv conv_seq2seq.zip src/
cd src/
unzip conv_seq2seq.zip
rm conv_seq2seq.zip
