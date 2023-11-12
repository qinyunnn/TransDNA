python target_train.py \
--train_data ./data/target_data/reads.txt \
--ground_truth ./data/target_data/reference.txt \
--pretrained_encoder PATH/TO/PRETRAINED_ENCODER \
--pretrained_decoder PATH/TO/PRETRAINED_DECODER \
--source_padding_length 154 \
--target_padding_length 96 \
--model_dir ./ \
--batch_size 32 \
--epoch 10
