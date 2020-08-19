#!/bin/bash
function openface(){
    for file in *
    do
        if test -f $file
        then
            sudo /home/ytikewk/python_project/OpenFace-master/build/bin/FeatureExtraction -f $file -out_dir "/home/ytikewk/python_project/daisee_detect/validation_process"
            #echo $file
        elif test -d $file
        then
            #echo $file
            cd $file
            openface
            cd ..
        fi
    done
}
 
openfaces