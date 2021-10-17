readonly  exp_folder=/media/jfm/SlaveEVO970/AcapellaEx
declare -i device=1
declare -i n_layers=7

export CUDA_VISIBLE_DEVICES=$device
if [ $n_layers == 7 ]
then

python3 run.py -m y_net_g  --workname test_y_net_ger${n_layers}_2sa --remix --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/y_net_ger7/metadata/1/1002_best_checkpoint.pth --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
python3 run.py -m y_net_g  --workname test_y_net_ger${n_layers}_1sa --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/y_net_ger7/metadata/1/1002_best_checkpoint.pth --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed

elif [ $n_layers == 5 ]
then
python3 run.py -m y_net_g  --workname test_y_net_ger${n_layers}_2sa --remix --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/venkatesh/y-net-gr7.pth --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed
python3 run.py -m y_net_g  --workname test_y_net_ger${n_layers}_1sa --pretrained_from /media/jfm/SlaveEVO970/AcapellaEx/venkatesh/y-net-gr7.pth --testing --force_dumping --arxiv_path $exp_folder --test_in test_seen test_seen_english_mixed test_seen_spanish_mixed test_seen_hindi_mixed test_seen_others_mixed test_unseen test_unseen_english_mixed test_unseen_spanish_mixed test_unseen_hindi_mixed test_unseen_others_mixed

fi
