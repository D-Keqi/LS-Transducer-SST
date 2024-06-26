#!/usr/bin/env bash
# simuleval requires .wav format
# source file should have 1 path per line, with no uttid
# cut -d' ' -f2- dump_wav/raw/tst-COMMON.en-de/wav.scp > dump_wav/raw/tst-COMMON.en-de/wav.scp.noid
export PYTHONPATH=./path/to/SimulEval:${PYTHONPATH}

source=dump_wav/raw/tst-COMMON.en-de/wav.scp.noid
target=dump_wav/raw/tst-COMMON.en-de/ref.trn.detok
agent=local/lst_simuleval_agent.py
nj=1
python=python3

batch_size=1
ngpu=1
exp=./path/to/exp_dir
inference_st_model=valid.acc.ave.pth 
inference_config=decode_st_online.yaml
inference_tag="$(basename "${inference_config}" .yaml)"
inference_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

lang="de"
rnnt=false
aed_type="waitk"
disable_repetition_detection=true
beam_size=1
sim_chunk_length=20480 #64*0.02*16000  # ms * fs; this controls the rate at which espnet calls beamsearch
ctc_weight=0.0
backend=streaming  
incremental_decode=true
penalty=0.0
latency_metrics="LAAL AL AP DAL"
hugging_face_decoder=true
source_segment_size=1   # ms; this controls the rate at which simuleval calls espnet; set as a factor of sim_chunk_length
recompute=true
token_delay=false
target_type=text
hold_n=0
chunk_decay=1.0
time_sync=false
blank_penalty=1.0

# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

output="$exp/simuleval_${inference_tag}"

st_train_config=$exp/config.yaml
st_model_file=$exp/$inference_st_model

echo $output
mkdir -p $output
mkdir -p $output/split$nj

${python} local/split_text_scp.py --text $target --scp $source --dst $output/split$nj --nj $nj

_cmd=$cuda_cmd
${_cmd} --gpu "${ngpu}" JOB=1:"${nj}" "${output}"/simuleval.JOB.log \
    simuleval --source $output/split$nj/source.JOB \
        --target $output/split$nj/target.JOB \
        --agent $agent \
        --batch_size $batch_size \
        --ngpu $ngpu \
        --st_train_config $st_train_config \
        --st_model_file $st_model_file \
        --disable_repetition_detection $disable_repetition_detection \
        --beam_size $beam_size \
        --ctc_weight $ctc_weight \
        --sim_chunk_length $sim_chunk_length \
        --chunk_decay $chunk_decay \
        --hold_n $hold_n \
        --backend $backend \
        --incremental_decode $incremental_decode \
        --penalty $penalty \
        --latency-metrics $latency_metrics \
        --hugging_face_decoder $hugging_face_decoder \
        --source-segment-size $source_segment_size \
        --recompute $recompute \
        --time_sync $time_sync \
        --blank_penalty $blank_penalty \
        --output $output/out.JOB \
        --token_delay $token_delay \
        --target-type $target_type \
        --rnnt $rnnt \
        --lang $lang \
        --aed_type $aed_type

${python} pyscripts/utils/merge_simuleval_logs.py --src $output/ --nj $nj --dst $output/instances.log

simuleval --score-only \
    --output $output \
    --latency-metrics $latency_metrics \
    --source-type speech \
    --target-type text >> $output/results.txt

cat $output/results.txt
