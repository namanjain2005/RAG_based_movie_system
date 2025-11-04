from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
# from torchvision import transforms
import json
import os

class MultimodalSearch():
    def __init__(self,docs, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = [] #should save this in pkl
        self.texts:list[str] = []
        self.docs = docs
        self.build()

    def build(self):
        for movie in self.docs:
            self.texts.append(f"{movie['title']}: {movie['description']}")
        if(os.path.exists("./cache/text_embedding.pt")):
            self.text_embeddings = torch.load("./cache/text_embedding.pt")
        else:
            self.text_embeddings = self.model.encode(self.texts,show_progress_bar=True,convert_to_tensor=True)
            torch.save(self.text_embeddings,"./cache/text_embedding.pt")

    def search_with_image(self,img_path,limit = 5):
        image_embedding = self.embed_image(img_path)
        print(image_embedding)
        text_scores = {}
        for i,text_emb in enumerate(self.text_embeddings):
            text_scores[i] =  self._calculate_cosine_similarity(image_embedding,text_emb)
        
        sorted_docs = sorted(text_scores.items(),key=lambda x : x[1],reverse=True)[:limit]
        sorted_ans = []
        for doc in sorted_docs:
            curr_doc = self.docs[doc[0]]
            sorted_ans.append({
                "doc_id":doc[0],
                "title" : curr_doc['title'],
                "desc" : curr_doc["description"],
                "sim_score" : doc[1]                
            })
        return sorted_ans

    def _calculate_cosine_similarity(self,img_emb,text_emb):
        return torch.nn.functional.cosine_similarity(img_emb.unsqueeze(0), text_emb.unsqueeze(0)).item()
    
    def embed_image(self, image_path):

        with Image.open(image_path) as img:
            img = img.convert("RGB")
        with torch.no_grad():
            emb = self.model.encode([img], convert_to_tensor=True)
        return emb[0]
    
    def embed_text(self, text):
        """Embed text queries"""
        with torch.no_grad():
            emb = self.model.encode([text], convert_to_tensor=True)
        return emb[0]
    
    def normalize_embedding(self, emb):
        """Normalize embedding to unit length"""
        return torch.nn.functional.normalize(emb, p=2, dim=-1)
    
    
    def verify_image_embedding(self, image_path):
        """Verify that image embedding works correctly"""
        try:
            emb = self.embed_image(image_path)
            print(f"✓ Image embedding successful")
            print(f"  Shape: {emb.shape}")
            print(f"  Device: {emb.device}")
            return True
        except Exception as e:
            print(f"✗ Image embedding failed: {e}")
            return False

