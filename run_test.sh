python bert_robust.py --data_path Robust/yelp.txt --tgt_path saved/yelp_vda.pt
python bert_robust.py --data_path Robust/yelp.txt --tgt_path saved/yelp_vda_freelb.pt
python bert_robust.py --data_path Robust/yelp.txt --tgt_path saved/yelp_vda_smix.pt
python bert_robust.py --data_path Robust/yelp.txt --tgt_path saved/yelp_vda_smart.pt

python bert_pair_robust.py --data_path Robust/qnli.txt --tgt_path saved/qnli_vda.pt
python bert_pair_robust.py --data_path Robust/qnli.txt --tgt_path saved/qnli_vda_freelb.pt
python bert_pair_robust.py --data_path Robust/qnli.txt --tgt_path saved/qnli_vda_smart.pt
python bert_pair_robust.py --data_path Robust/qnli.txt --tgt_path saved/qnli_vda_smix.pt
