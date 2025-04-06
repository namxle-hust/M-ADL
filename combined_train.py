import os
import csv
import nltk
import torch
import pickle
import numpy as np
import time
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score_score

nltk.download('punkt')

######################
# 1. Tiền xử lý dữ liệu và từ vựng
######################
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.lower() for tok in nltk.tokenize.word_tokenize(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized_text]
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        """
        Args:
            root_dir: đường dẫn folder chứa ảnh.
            captions_file: file CSV có 2 cột "image", "caption"
            vocab: đối tượng Vocabulary
            transform: các transform cho ảnh
        """
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        self.imgs = []
        self.captions = []
        with open(captions_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.imgs.append(row["image"])
                self.captions.append(row["caption"])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])
        return image, torch.tensor(numericalized_caption)

def collate_fn(data):
    """
    Hàm collate cho DataLoader, padding các caption về độ dài bằng nhau.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
    return images, padded_captions, lengths

######################
# 2. Encoder với Full-Memory Transformer và Skip-Connections
######################
class FullMemoryTransformer(nn.Module):
    def __init__(self, encoder_dim, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super(FullMemoryTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features):
        features = features.permute(1, 0, 2)  
        transformed = self.transformer(features)  
        transformed = transformed.permute(1, 0, 2)  
        return transformed

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.full_memory_transformer = FullMemoryTransformer(encoder_dim=2048, nhead=8, num_layers=2)
        
    def forward(self, images):
        features = self.resnet(images)  
        features = self.adaptive_pool(features)  
        features = features.permute(0, 2, 3, 1)
        batch_size, H, W, encoder_dim = features.size()
        features_flat = features.view(batch_size, H * W, encoder_dim)  
        transformer_features = self.full_memory_transformer(features_flat)
        combined_features = transformer_features + features_flat  
        return combined_features  

######################
# 3. Các module phụ: Attention, Hash Memory
######################
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)          
        att2 = self.decoder_att(decoder_hidden)         
        att2 = att2.unsqueeze(1)                        
        att = self.full_att(self.relu(att1 + att2))       
        att = att.squeeze(2)                            
        alpha = self.softmax(att)                       
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class HashMemory(nn.Module):
    def __init__(self, hidden_dim, memory_size=128):
        super(HashMemory, self).__init__()
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
    def forward(self, x):
        similarity = torch.matmul(x, self.memory_bank.t())  
        att_weights = torch.softmax(similarity, dim=1)
        memory_out = torch.matmul(att_weights, self.memory_bank)
        out = x + memory_out
        return out

######################
# 4. Decoder (Show, Attend and Tell + HashMemory)
######################
class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=2048, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.hash_memory = HashMemory(decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)
        
        embeddings = self.embedding(encoded_captions)  
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = [l - 1 for l in caption_lengths]
        
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            h = self.hash_memory(h)
            preds = self.fc(self.dropout_layer(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, decode_lengths, alphas

    def generate_caption(self, encoder_out, vocab, max_len=20):
        assert encoder_out.size(0) == 1, "Chỉ hỗ trợ sinh caption từng ảnh một."
        h, c = self.init_hidden_state(encoder_out)
        word = torch.tensor([vocab.stoi["<START>"]]).to(encoder_out.device)
        caption = []
        for t in range(max_len):
            embedding = self.embedding(word).unsqueeze(0)
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            lstm_input = torch.cat([embedding.squeeze(1), attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            h = self.hash_memory(h)
            preds = self.fc(self.dropout_layer(h))
            predicted = preds.argmax(1)
            word = predicted
            token = vocab.itos[predicted.item()]
            if token == "<END>":
                break
            caption.append(token)
        return caption

######################
# 5. Evaluation Metrics: BLEU, CIDEr, ROUGE, SPICE, BERTScore, and Avg Inference Time
######################
def compute_bleu(refs_dict, hyps_dict):
    smoothie = SmoothingFunction().method4
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    for key in refs_dict.keys():
        ref_tokens = refs_dict[key][0].split()
        hyp_tokens = hyps_dict[key][0].split()
        bleu_scores[1].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_scores[2].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_scores[3].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu_scores[4].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
    bleu1 = np.mean(bleu_scores[1])
    bleu2 = np.mean(bleu_scores[2])
    bleu3 = np.mean(bleu_scores[3])
    bleu4 = np.mean(bleu_scores[4])
    return bleu1, bleu2, bleu3, bleu4

def compute_cider(refs_dict, hyps_dict):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs_dict, hyps_dict)
    return score

#def compute_rouge(refs_dict, hyps_dict):
    rouge_scorer = Rouge()
    score, _ = rouge_scorer.compute_score(refs_dict, hyps_dict)
    return score

#def compute_spice(refs_dict, hyps_dict):
    spice_scorer = Spice()
    score, _ = spice_scorer.compute_score(refs_dict, hyps_dict)
    return score

def compute_bertscore(refs, hyps, lang="en"):
    # refs, hyps: list of strings
    P, R, F1 = bert_score_score(hyps, refs, lang=lang, verbose=False)
    return torch.mean(F1).item()

def evaluate_model(encoder, decoder, dataloader, vocab, device):
    encoder.eval()
    decoder.eval()
    refs_dict = {}
    hyps_dict = {}
    total_inference_time = 0.0
    count = 0
    idx = 0
    with torch.no_grad():
        for i, (imgs, captions, lengths) in enumerate(dataloader):
            imgs = imgs.to(device)
            encoder_out = encoder(imgs)
            for j in range(imgs.size(0)):
                start_time = time.time()
                enc_out = encoder_out[j].unsqueeze(0)
                generated_tokens = decoder.generate_caption(enc_out, vocab)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                count += 1
                generated_sentence = " ".join(generated_tokens)
                target_tokens = [vocab.itos[token.item()] for token in captions[j] 
                                 if token.item() not in [vocab.stoi["<START>"], vocab.stoi["<END>"], vocab.stoi["<PAD>"]]]
                reference_sentence = " ".join(target_tokens)
                refs_dict[str(idx)] = [reference_sentence]
                hyps_dict[str(idx)] = [generated_sentence]
                idx += 1
    avg_inference_time = total_inference_time / count if count > 0 else 0.0
    return refs_dict, hyps_dict, avg_inference_time

######################
# 6. Main Training & Evaluation
######################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "archive (3)/Images"      #thay bằng đường dẫn của bạn
    captions_file = "archive (3)/captions.txt"   #thay bằng đường dẫn của bạn 
    freq_threshold = 5
    embed_dim = 256
    attention_dim = 256
    decoder_dim = 512
    dropout = 0.5
    num_epochs = 25
    batch_size = 32
    learning_rate = 1e-4

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    all_captions = []
    with open(captions_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_captions.append(row["caption"])
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(all_captions)
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    dataset = Flickr8kDataset(root_dir, captions_file, vocab, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(attention_dim, embed_dim, decoder_dim, vocab_size, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Huấn luyện
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.
        for imgs, captions, lengths in train_loader:
            imgs, captions = imgs.to(device), captions.to(device)
            optimizer.zero_grad()
            encoder_out = encoder(imgs)
            preds, decode_lengths, alphas = decoder(encoder_out, captions, lengths)
            targets = captions[:, 1:]
            preds_list, targets_list = [], []
            for j in range(len(decode_lengths)):
                preds_list.append(preds[j, :decode_lengths[j], :])
                targets_list.append(targets[j, :decode_lengths[j]])
            preds_concat = torch.cat(preds_list, dim=0)
            targets_concat = torch.cat(targets_list, dim=0)
            loss = criterion(preds_concat, targets_concat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Đánh giá
    refs_dict, hyps_dict, avg_inference_time = evaluate_model(encoder, decoder, test_loader, vocab, device)
    bleu1, bleu2, bleu3, bleu4 = compute_bleu(refs_dict, hyps_dict)
    cider_score = compute_cider(refs_dict, hyps_dict)
    #rouge_score = compute_rouge(refs_dict, hyps_dict)
    #spice_score = compute_spice(refs_dict, hyps_dict)
    refs_list = [refs_dict[k][0] for k in refs_dict.keys()]
    hyps_list = [hyps_dict[k][0] for k in hyps_dict.keys()]
    bertscore_f1 = compute_bertscore(refs_list, hyps_list, lang="en")
    
    print("BLEU-1: {:.4f}".format(bleu1))
    print("BLEU-2: {:.4f}".format(bleu2))
    print("BLEU-3: {:.4f}".format(bleu3))
    print("BLEU-4: {:.4f}".format(bleu4))
    print("CIDEr: {:.4f}".format(cider_score))
    #print("ROUGE-L: {:.4f}".format(rouge_score))
    #print("SPICE: {:.4f}".format(spice_score))
    print("BERTScore F1: {:.4f}".format(bertscore_f1))
    print("Avg Inference Time per Image: {:.4f} seconds".format(avg_inference_time))
    
    # Lưu model
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

if __name__ == '__main__':
    main()