cd /Users/pablo/Dropbox/workspace/darth_linguo/python

#python generateWS.py euro_hundo

python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-grammatical True False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-2-gramWS False False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-3-gramWS False False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-4-gramWS False False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-5-gramWS False False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-6-gramWS False False 80
python splitCorpus.py /Users/pablo/Dropbox/workspace/darth_linguo/Data/euro_hundo/euro_hundo-mix-gramWS False False 80

python experimenter.py euro_hundo 32 32 32 5 0.1 false 2-gramWS
python experimenter.py euro_hundo 32 32 32 5 0.1 false 3-gramWS
python experimenter.py euro_hundo 32 32 32 5 0.1 false 4-gramWS
python experimenter.py euro_hundo 32 32 32 5 0.1 false 5-gramWS
python experimenter.py euro_hundo 32 32 32 5 0.1 false 6-gramWS
python experimenter.py euro_hundo 32 32 32 5 0.1 false mix-gramWS
