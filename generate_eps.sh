python3 main.py working --model='ssc=1500.model' --predict --coupling=10 --save-to-file='1c-1p'
python3 main.py double --model='ssc=2500.model' --predict --coupling=1 --save-to-file='1c-double-2500ssc'
python3 main.py double --model='ssc=1500.model' --predict --coupling=1 --save-to-file='1c-double'
python3 main.py optical_lat --model='ssc=2500.model' --predict --coupling=1 --save-to-file='1c-optical_lat'
python3 main.py working --model='ssc=250.model' --predict --coupling=10 --save-to-file='rsampling250'
python3 main.py working --model='ssc=500.model' --predict --coupling=10 --save-to-file='rsampling500'
python3 main.py working --model='ssc=750.model' --predict --coupling=10 --save-to-file='rsampling750'
python3 main.py working --model='ssc=1000.model' --predict --coupling=10 --save-to-file='rsampling1000'
python3 main.py 2harmonic --two-component --model='ssc=3000.model' --predict --couplings='1,1,1' --save-to-file='2c-harmonic-111'
python3 main.py 2comp --two-component --model='ssc=2000.model' --predict --couplings='103,100,97' --save-to-file='2c-cosine-103-100-97-omega=-1'
python3 main.py 2double --two-component --model='ssc=2000.model' --predict --couplings='1,1,1' --save-to-file='2c-double-111'
python3 main.py working --model='ssc=1500.model' --predict --couplings='1,10,50,100' --overlay --save-to-file='1c-4g-overlay'