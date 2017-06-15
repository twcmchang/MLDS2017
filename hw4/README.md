# HW4 - Seq2seq & RL

## Testing
```
bash run.sh [S2S, RL, BEST] [INPUT_FILE] [OUTPUT_FILE]
```
Trained Models and vocabularies would be downloaded.

## Training
```
bash train.sh [<workspace>] [S2S, BiS2S] [RL] [SS]
```
Before training, a work space should be created in works/<workspace>, and training data should be put in works/<workspace>/data/train/ and named and packaged as chat.txt.gz.
- [S2S, BiS2S]: Attention-based Seq2seq (S2S) or Bidirectional-encoder Seq2seq (BiS2S)
- [RL]: True for using reinforcement learning
- [SS]: True for using scheduled sampling for training model

# Features
- Sequence-to-sequence Model
- Attention Mechanism
- Bidirectional Encoder for First Layer
- Residual Connections
- Scheduled Sampling
- Reinforcement Learning
- Beam Search
- AntiLM

## Datasets
- [Cornell Movie-Dialogs Corpus][CMDS]
- [OpenSubtitles][OS]
- [Marsan-Ma chat-corpus][MMCC]

## References
- [Tf Chatbot Seq2seq AntiLM][MM]
- [Search QA][SQA]
- [Deep learning based chatbot][DLBC]
- [Deep Reinforcement Learning for Dialogue Generation][DRLDG]

[MM]:https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm
[slide]: https://docs.google.com/presentation/d/1e-9a7MmHDi1OfXrSFh_NOuyXjK2cN640JcZ5D08MBEk/edit#slide=id.g1efeb48205_0_0
[CMDS]: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
[OS]: http://opus.lingfil.uu.se/OpenSubtitles.php
[MMCC]: https://github.com/Marsan-Ma/chat_corpus
[CBC]: https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data
[CH]: http://talkbank.org/access/CABank/CallHome/eng.html
[SQA]: https://github.com/nyu-dl/SearchQA
[DLBC]: https://github.com/Conchylicultor/DeepQA
[DRLDG]: https://github.com/liuyuemaicha/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-tensorflow
