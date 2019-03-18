# Training Steps For COQG
We are using open_NMT implementation of copy atttention encoder decoder model. The training process is completed using the following script:
```
 python OpenNMT-py/train.py -data data/data.feat.coqg -save_model seq2seq_models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/data.feat.coqg.embed.enc.pt -pre_word_vecs_dec data/data.feat.coqg.embed.dec.pt -epochs 50 -gpuid 0 -seed 123

```

## Difference between COQA Conversational/RC-Models and COQG?
COQA dataset is released for conversational question answering, where as COQG tries to revert the process by generating conversational questions. Following are the key distinctions between COQA and COQG:
1. Both use the COQA dataset but in a different context. In reading comprehension models or conversational models for COQA, the target is to answer questions on a given paragraph. However, in contrast, COQG is focused on conversational models and the target is to take the paragraph, rationale and generate questions that could be used in a conversational context.
2. The deep learning encoder decoder architecture used by both models is same.
3. Training, Validation process still is the same