from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
<<<<<<< HEAD
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict, sanitize
=======
from allennlp.predictors import SentenceTaggerPredictor
>>>>>>> e9ded0d684e84953dcc510df2eb4f3498e03b9a6

torch.manual_seed(1)

@DatasetReader.register("linguo_reader")
class LinguoDatasetReader(DatasetReader):
    """Dataset reader for preprocessed sentences """
    GRAMMATICALITY_labels = ["ungrammatical", "grammatical"]
    UG_TYPE_labels = ["WS", "VA", "AA", "RV", "G"]

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self,
                         tokens: List[Token],
                         glabel: int=None,
                         ugType: str=None):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if glabel:
            glabel_field = LabelField(label=glabel,
                                      label_namespace="grammaticality_labels")
            fields["g_label"] = glabel_field
        if ugType:
            ugType_field = LabelField(label=ugType,
                                      label_namespace="ugtype_labels")
            fields["ug_type"] = ugType_field
        return Instance(fields)

    def _read(self,
              file_path: str,
              label: str=None,
              ugType: str=None) -> Iterator[Instance]:
        with open(file_path) as infile:
            for line in infile:
                elements = line.strip().split()
                label = self.GRAMMATICALITY_labels[int(elements[0])]
                if label == self.GRAMMATICALITY_labels[0]:
                    ugType = elements[1]
                else:
                    ugType = "G"
                sentence = elements[2:]
                yield self.text_to_instance([Token(word) for word in sentence],label,ugType)


@Model.register("linguo")
class AllenLinguo(Model):

    def __init__(self,word_embeddings : TextFieldEmbedder,
                encoder : Seq2VecEncoder,
                vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2decision = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                              out_features=vocab.get_vocab_size("grammaticality_labels"))
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        self.vocab = vocab
        self.specificAccuracies = {}
        for ind in range(vocab.get_vocab_size(namespace="ugtype_labels")):
            self.specificAccuracies[ind] = CategoricalAccuracy()

    def forward(self,
               sentence: Dict[str, torch.Tensor],
               g_label: torch.Tensor = None,
               ug_type: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2decision(encoder_out)

        output = {"tag_logits": tag_logits}

        if g_label is not None:
            # Update general accuracy and compute loss
            self.accuracy(tag_logits, g_label)
            output["loss"] = self.loss_function(tag_logits, g_label)
            # update specific accuracies
            n_ugtypes = self.vocab.get_vocab_size(namespace="ugtype_labels")
            specific_gold = {n: [] for n in range(n_ugtypes)}
            specific_pred = {n: [] for n in range(n_ugtypes)}
            for ind in range(len(g_label)):
                g_lab = g_label[ind].item()
                logit = [tag_logits[ind][0].item(), tag_logits[ind][1].item()]
                spec_label = ug_type[ind].item()
                specific_gold[spec_label].append(g_lab)
                specific_pred[spec_label].append(logit)
            for ind in self.specificAccuracies:
                if specific_pred[ind]:
                    preds = torch.tensor(specific_pred[ind])
                    labels = torch.tensor(specific_gold[ind])
                    self.specificAccuracies[ind](preds, labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(reset)}
        #print(len(self.specificAccuracies))
        for ind in self.specificAccuracies:
            name = self.vocab.get_token_from_index(ind, namespace="ugtype_labels")
            name += "_accuracy"
            metrics[name] = self.specificAccuracies[ind].get_metric(reset)
        #print(metrics)
        return metrics

@Predictor.register('linguo-predictor')
class GrammaticalityJudge(Predictor):
    def __init__(self, model:Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
    
    def predict_instance(self, instance: Instance) -> JsonDict:

        outputs = self._model.forward_on_instance(instance)
        sanitized = sanitize(outputs)
        sanitized["sentence"] = " ".join([ str(token) 
            for token in instance["sentence"].tokens])
        sanitized["ug_type"] = instance["ug_type"].label
        sanitized["predicted_label"]
        print("I got {}".format(str(sanitized)))
        return sanitized 
