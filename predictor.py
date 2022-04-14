import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
from bert_classification import NewsDataset, BertNewsClassifier, BertTokenizer


class Predictor:
    def __init__(self, model_dir):

        model_path = os.path.join(model_dir, 'model.pt')
        tokenizer_dir = os.path.join(model_dir, 'tokenizer')
        params_path = os.path.join(model_dir, 'params.json')

        dict_args = json.load(open(params_path, 'r'))
        self.model = BertNewsClassifier(**dict_args)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = 100
        self.batch_size = 1
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, reviews):
        targets = np.array([0] * len(reviews))
        dataset = NewsDataset(
            reviews, targets, self.tokenizer, self.max_length)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=1
        )

        results = []
        for batch in data_loader:

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            output = self.model.forward(input_ids, attention_mask)
            _, y_hat = torch.max(output, dim=1)
            results.append(y_hat.item())
        return results


class TextClassificationWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = context.artifacts["model_dir"]
        self.predictor = Predictor(model_dir)

        pass

    def predict(self, context, model_input):
        labels = self.predictor.predict(model_input.description.to_numpy())
        return {"results": labels}
