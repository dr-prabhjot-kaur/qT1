# pdate: Proton Density and T1/T2 Estimation

## Create docker image
docker build -t crl:pdate:0.9.0_0.1.0 .

## Run docker container
docker run -it --rm --name pdate -v /your/local/data/path:/opt/pDate/data -v /your/local/recons/path:/opt/pDate/recons crl/pdate:0.9.0_0.1.0 -m T1 --TI 400,800,1200 -fa 158

### usage: pDate.py [-h] [-V] -m {T1,T2} [--TI TI] [--TE TE] [-fa FLIP_ANGLE]

optional arguments:

  -h, --help            show this help message and exit
  
  -V, --version         show version
  
  -m {T1,T2}, --mapping {T1,T2}
                        specify [T1] or [T2] mapping will be estimated
                        
  --TI TI               specify inverse time (TI) values, separated by comma(,)
  
  --TE TE               specify echo time (TE) values, separated by comma(,)
  
  -fa FLIP_ANGLE, --flip-angle FLIP_ANGLE
                        specify flip angle values in DEGREES, separated by comma(,)
