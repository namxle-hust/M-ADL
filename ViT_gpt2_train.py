import os

import evaluate
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

import nltk
from nltk.tokenize import sent_tokenize
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score

rouge = evaluate.load("rouge")
nltk.download('punkt_tab')
smoothie = SmoothingFunction().method4


if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Adding some constants
data_path = "data"

# Declare configurations
class config : 
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 5
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95



def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


# Compute metrics function for validation dataset during training process
def compute_metrics(pred):
    # Extract predictions and labels
    pred_ids = pred.predictions
    labels_ids = pred.label_ids

    # Replace -100 (ignored index) with pad_token_id for decoding
    labels_ids = np.where(labels_ids == -100, tokenizer.pad_token_id, labels_ids)

    # Decode predicted and reference sequences
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Compute ROUGE (only returns F1 scores)
    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True  # Optional but improves matching
    )

    # Extract and round the ROUGE-2 F1 score
    return {
        "rouge2_fmeasure": round(rouge_output["rouge2"], 4),
    }




class ImgDataset(Dataset):
    def __init__(self, df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50
    def __len__(self,):
        return len(self.df)
    def __getitem__(self,idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir , image)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img= self.transform(img)
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,
                                 padding='max_length',
                                 max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding
        
        
def get_scores(model, val_df):
    # Load model if available
    # model = VisionEncoderDecoderModel.from_pretrained(model_path)
    batch_size = 16  # you can increase if GPU allows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set the model to evaluation mode and move to device
    model.eval()
    model.to(device)


    # Storage for final outputs
    refs_dict, preds_dict, full_preds_dict = {}, {}, {}

    # Preload all image tensors and captions
    images = []
    references = []
    indexes = []

    count = 0

    for index, row in val_df.iterrows():
        print(count)
        img_path = f"{data_path}/flickr8k/Images/" + row["image"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = feature_extractor(img, return_tensors="pt").pixel_values[0]
        images.append(img_tensor)
        references.append(row["caption"])
        indexes.append(index)
        count += 1

    # Stack tensors into a single batch tensor list
    aimages = torch.stack(images)  # shape: (N, 3, 224, 224)

    # Run in batches
    for i in tqdm(range(0, len(aimages), batch_size)):
        batch = aimages[i:i+batch_size].to(device)
        print("Batch ", i)

        with torch.no_grad():
            outputs = model.generate(batch, max_length=50, num_beams=4)

        decoded_captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, caption in enumerate(decoded_captions):
            index = indexes[i + j]
            cleaned_caption = sent_tokenize(caption.replace("<|endoftext|>", "")[:120].strip())[0]
            refs_dict[str(index)] = [references[i + j]]
            preds_dict[str(index)] = [cleaned_caption]
            full_preds_dict[str(index)] = [caption]

    # Cider score
    cider = Cider()
    # BLEU score
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}

    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []

    for idx in preds_dict:
        pred = preds_dict[idx][0].split()
        ref = [refs_dict[idx][0].split()]  # wrap in list for multiple references

        bleu_1 = sentence_bleu(ref, pred, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu(ref, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 = sentence_bleu(ref, pred, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)


    # Average BLEU scores
    print("BLEU-1:", round(sum(bleu_1_scores)/len(bleu_1_scores), 4))
    print("BLEU-2:", round(sum(bleu_2_scores)/len(bleu_2_scores), 4))
    print("BLEU-3:", round(sum(bleu_3_scores)/len(bleu_3_scores), 4))
    print("BLEU-4:", round(sum(bleu_4_scores)/len(bleu_4_scores), 4))

    # Cider score
    cider_score, _ = cider.compute_score(refs_dict, preds_dict)
    print("CIDEr:", round(cider_score, 4))


# Build autotokenize inputs with special tokens
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

transforms = transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5, 
            std=0.5
        )
    ]
)

def main():
    # Load dataset
    df=  pd.read_csv(f"{data_path}/flickr8k/captions.txt")
    train_df , val_df = train_test_split(df, test_size = 0.2)
    # train_df = train_df.head(5)
    # val_df = val_df.head(5)
    train_dataset = ImgDataset(train_df, root_dir = f"{data_path}/flickr8k/Images",tokenizer=tokenizer,feature_extractor = feature_extractor ,transform = transforms)
    val_dataset = ImgDataset(val_df , root_dir = f"{data_path}/flickr8k/Images",tokenizer=tokenizer,feature_extractor = feature_extractor , transform  = transforms)
    

    # Load pretrained model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
    # Adding some config
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Add training args
    training_args = Seq2SeqTrainingArguments(
        output_dir='ViT-gpt2',
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VAL_BATCH_SIZE,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024,  
        save_steps=2048, 
        warmup_steps=1024,  
        learning_rate = 5e-5,
        #max_steps=1500, # delete for full training
        num_train_epochs = config.EPOCHS, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=1,
    )

    # Instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


    # Save model
    trainer.save_model('ViT_gpt2')

    # Get scores
    get_scores(model, val_df)

if __name__ == '__main__':
    main()