#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${JVS}" ]; then
   log "Fill the value of 'JVS' of db.sh"
   exit 1
fi
db_root=${JVS}

train_set="tr_no_dev"
dev_set="dev"
eval_set="eval1"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/data_download.sh"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    
    train_data_dirs=""
    dev_data_dirs=""
    eval_data_dirs=""
    spks=$(find "${db_root}/jvs_ver1" -maxdepth 1 -name "jvs*" -exec basename {} \; | sort | grep -vE 'jvs006|jvs028')
    for spk in ${spks}; do
        local/data_prep.sh "${db_root}"/jvs_ver1 "${spk}" "data/${spk}"
    
        train_data_dirs+=" data/${spk}_${train_set}"
        dev_data_dirs+=" data/${spk}_${dev_set}"
        eval_data_dirs+=" data/${spk}_${eval_set}"
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/combine_data.sh"

    spks=$(find "${db_root}/jvs_ver1" -maxdepth 1 -name "jvs[0-9]*" -exec basename {} \; | sort | grep -vE 'jvs006|jvs028')
    for spk in ${spks}; do
        train_data_dirs+=" data/${spk}_${train_set}"
        dev_data_dirs+=" data/${spk}_${dev_set}"
        eval_data_dirs+=" data/${spk}_${eval_set}"
    done

    utils/combine_data.sh data/${train_set} ${train_data_dirs}
    utils/combine_data.sh data/${dev_set} ${dev_data_dirs}
    utils/combine_data.sh data/${eval_set} ${eval_data_dirs}

    rm -rf data/jvs[0-9]*
fi



log "Successfully finished. [elapsed=${SECONDS}s]"
