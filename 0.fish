#!/bin/false
c channel 24 config.py
python 2.py
c channel 48 config.py
python 2.py
c channel 64 config.py
python 2.py
c channel 96 config.py
python 2.py
c channel 24 config.py

c penalty_num 47 config.py 
python 2.py
c penalty_num 35 config.py 
python 2.py
c penalty_num 23 config.py 
python 2.py
c penalty_num 7 config.py 
python 2.py
c penalty_num 3 config.py 
python 2.py
c penalty_num 63 config.py

c sigma_range True config.py
c sigma 75 config.py
python 2.py
c sigma_range False config.py
c sigma 25 config.py

#c(){
    #sed -r 's/('$1'=).*,/\1'$2',/' config.py
#}
#c channel 24
#fish -c 'conda activate;python 2.py'
#c channel 36
#c penalty_num 36

#sed -i 's/(channel=).*,/\124,/' config.py
#sed -i 's/(channel=).*,/\136,/' config.py
#sed -i 's/(channel=).*,/\164,/' config.py
#sed -i 's/(channel=).*,/\196,/' config.py
#sed -i 's/(penalty_num=).*,/\11,/' config.py
