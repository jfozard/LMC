
mkdir Orig_data/
cd Orig_data/

for s in {1..3} ; do
    
    wget "https://seafile.lirmm.fr/d/123f71e12bf24db59d84/files/?p=%2FStudy_${s}.tar.gz&dl=1" -O Study_${s}.tar.gz
    tar xjf Study_${s}.tar.gz
done
