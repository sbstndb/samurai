##
## pip3 install numpy
## pip3 install scipy
##

python3 ../build/_deps/googlebench-src/tools/compare.py benchmarks results.json results2.json


## Output --> 

#Benchmark                                                 Time             CPU      Time Old      Time New       CPU Old       CPU New
#--------------------------------------------------------------------------------------------------------------------------------------
#CASE_linear_convection_little/repeats:1                +0.0191         +0.0191    6312074392    6432840417    6310644970    6431073513
#CASE_linear_convection_medium/repeats:1                +0.0087         +0.0085    7025446691    7086726651    7023613519    7083608305
#CASE_linear_convection_large/repeats:1                 -0.0040         -0.0039   95704253385   95322449676   95678201869   95300671791
#CASE_advection_2d_little/repeats:1                     +0.0071         +0.0072    2341452289    2358150262    2340765947    2357641123
#CASE_advection_2d_medium/repeats:1                     +0.0036         +0.0037   13611233184   13659722419   13606297431   13656881081
#CASE_advection_2d_large/repeats:1                      +0.0169         +0.0169   48833804641   49658527263   48819079656   49643270478
#OVERALL_GEOMEAN                                        +0.0085         +0.0086            14            14            14            14
