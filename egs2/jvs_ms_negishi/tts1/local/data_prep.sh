#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

num_dev=5
num_eval=5
train_set="tr_no_dev"
dev_set="dev"
eval_set="eval1"

db=$1
spk=$2
org_data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <corpus_dir> <target_spk> <data_dir>"
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e ${db}/${spk} ] && echo "${spk} does not exist." >&2 && exit 1;

# NOTE: we use only parallel100 and nonpara30 for TTS
for name in parallel100 nonpara30; do
    data_dir=${org_data_dir}_${name}

    # check directory existence
    [ ! -e ${data_dir} ] && mkdir -p ${data_dir}

    # set filenames
    scp=${data_dir}/wav.scp
    utt2spk=${data_dir}/utt2spk
    spk2utt=${data_dir}/spk2utt
    text=${data_dir}/text
    segments=${data_dir}/segments

    # check file existence
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${text} ] && rm ${text}
    [ -e ${segments} ] && rm ${segments}

    # make scp, utt2spk, and spk2utt
    find ${db}/${spk}/${name} -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${spk}_${id} ${filename}" >> ${scp}
        echo "${spk}_${id} ${spk}" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
    echo "finished making wav.scp, utt2spk, spk2utt."

    # make text
    sed -e "s/:/ /g" "${db}/${spk}/${name}/transcripts_utf8.txt" | sed -e "s/^/${spk}_/g" > ${text}
    echo "finished making text."

    # make segments
    find ${db}/${spk}/${name}/lab/mon -follow -name "*.lab" | sort | while read -r filename; do
        start=$(head -n 1 ${filename} | cut -d " " -f 2)
        end=$(tail -n 1 ${filename} | cut -d " " -f 1)
        id="${spk}_$(basename ${filename} .lab)"
        echo "${id} ${id} ${start} ${end}" >> ${segments}
    done
    echo "finished making segments."

    # check
    utils/fix_data_dir.sh ${data_dir}

done

num_deveval=$((num_dev + num_eval))
num_train=$((30 - num_deveval))
utils/subset_data_dir.sh --first "data/${spk}_nonpara30" ${num_train} "data/${spk}_nonpara30_train"
utils/subset_data_dir.sh --last "data/${spk}_nonpara30" ${num_deveval} "data/${spk}_nonpara30_deveval"
utils/subset_data_dir.sh --first "data/${spk}_nonpara30_deveval" ${num_dev} "data/${spk}_${dev_set}"
utils/subset_data_dir.sh --last "data/${spk}_nonpara30_deveval" ${num_eval} "data/${spk}_${eval_set}"
utils/combine_data.sh "data/${spk}_${train_set}" "data/${spk}_parallel100" "data/${spk}_nonpara30_train"

rm -rf data/${spk}_nonpara30*
rm -rf data/${spk}_parallel100

echo "Successfully finished data preparation."
