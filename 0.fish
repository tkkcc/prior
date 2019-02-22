#!/bin/false
function c
    sed -ri 's/(.*'$argv[1]'=).*,/\1'$argv[2]',/' config.py
end
c model \"tnrd\"
c stage 1
c lr 1e-4
c penalty_num 63
./2.py
#c model tnrdi
#./2.py
exit
#c stage 1
#python 2.py
#c stage 2
#python 2.py
c stage 3
python 2.py
c stage 4
python 2.py
c stage 5
python 2.py
#c stage 6
#python 2.py
#c stage 7
#python 2.py
#c stage 8
#python 2.py

exit
c depth 1
python 2.py
c depth 2
python 2.py
c depth 3
python 2.py
c depth 2

c channel 48
python 2.py
c channel 64
python 2.py
c channel 24

c penalty_num 47
python 2.py
c penalty_num 35
python 2.py
c penalty_num 23
python 2.py
c penalty_num 7
python 2.py
c penalty_num 3
python 2.py
c penalty_num 63

c sigma_range True
c sigma 75
python 2.py
c sigma_range False config.py
c sigma 25 config.py

c batch_size 4 config.py
python 2.py
c batch_size 8 config.py
