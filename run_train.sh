python text_classifier.py --dataset yelp --save_path saved/yelp_vda.pt --max_length 512 --batch_size 12
python text_classifier_freelb.py --dataset yelp --save_path saved/yelp_vda_freelb.pt --max_length 512 --batch_size 8
python text_classifier_smix.py --dataset yelp --save_path saved/yelp_vda_smix.pt --max_length 512 --batch_size 6
python text_classifier_smart.py --dataset yelp --save_path saved/yelp_vda_smart.pt --max_length 512 --batch_size 6

python text_pair_classifier.py --dataset qnli --save_path saved/qnli_vda.pt
python text_pair_classifier_freelb.py --dataset qnli --save_path saved/qnli_vda_freelb.pt
python text_pair_classifier_smart.py --dataset qnli --save_path saved/qnli_vda_smart.pt
python text_pair_classifier_smix.py --dataset qnli --save_path saved/qnli_vda_smix.pt
