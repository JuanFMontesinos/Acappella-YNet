readonly  exp_folder=/media/jfm/SlaveEVO970/AcapellaEx
declare -i device=1

export CUDA_VISIBLE_DEVICES=$device

python3 run.py -m u_net --workname white_metrics_2sa --remix --testing --white_metrics --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
python3 run.py -m u_net --workname white_metrics_1sa         --testing --white_metrics --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed