readonly  exp_folder=/media/jfm/SlaveEVO970/AcapellaEx
declare -i device=1
declare -i n_layers=5

export CUDA_VISIBLE_DEVICES=$device

#python3 run.py -m y_net_r_legacy  --workname test_y_net_r${n_layers}_2sa --remix --legacy --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
#python3 run.py -m y_net_r_legacy  --workname test_y_net_r${n_layers}_1sa --legacy --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
#
#python3 run.py -m y_net_m_legacy  --workname test_y_net_m${n_layers}_2sa --remix --legacy --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
#python3 run.py -m y_net_m_legacy  --workname test_y_net_m${n_layers}_1sa --legacy --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed

python3 run.py -m llcp  --workname test_llcp${n_layers}_2sa --remix --testing --force_dumping --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/llcp_r/metadata/3/0297_best_checkpoint.pth --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
python3 run.py -m llcp  --workname test_llcp${n_layers}_1sa --testing --force_dumping --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/llcp_r/metadata/3/0297_best_checkpoint.pth --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
