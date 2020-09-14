#
#!/bin/bash
sudo apt install tmux

dlinks=("https://zenodo.org/record/3723295/files/subset0.zip" "https://zenodo.org/record/3723295/files/subset1.zip")
length=${#dlinks[@]}
echo $length

for (( i = 0; i <= length; i++ )); do
        file1=${dlinks[i]}
        echo $file1
        tname=$(echo $file1 | cut -d '.' -f2 | rev | cut -d '/' -f1 | rev)
        filename=$(echo $file1 | rev | cut -d '/' -f1 | rev)

        tmux kill-session -t $tname
        tmux new-session -d -s $tname && tmux send-keys -t $tname C-z "mkdir $tname; cd $tname;wget $file1; 7z e $filename" Enter
    done

link_="https://zenodo.org/record/3723295/files/"
impfiles=("annotations.csv" "candidates.csv" "candidates_V2.zip" "evaluationScript.zip" "sampleSubmission.csv" "seg-lungs-LUNA16.zip")
length1=${#impfiles[@]}

for (( i = 0; i <= length1; i++ )); do
        file1=${impfiles[i]}
        downlink=$link_$file1
        wget $downlink

    done

