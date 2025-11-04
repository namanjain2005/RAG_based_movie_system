from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
# from torchvision import transforms
import json

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = {}
        self.doc_map = {}

    def build(self):
        with open('./data/course-rag-movies.json','r') as f:
            movies = json.load(f)
        movies = movies['movies']
        for movie in movies:
            self.doc_map[movie['id']] = movie
        

    
    def embed_image(self, image_path):

        with Image.open(image_path) as img:
            img = img.convert("RGB")
        # to_tensor_transform = transforms.ToTensor()
        # tensor_image = to_tensor_transform(img)
        # print(tensor_image)
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



# Example usage
if __name__ == "__main__":
    search = MultimodalSearch()

    search.verify_image_embedding("./../../data/paddington.jpeg")
    
    img_emb = search.embed_image("./../../data/paddington.jpeg")
    text_emb = search.embed_text("paddington")    

    # print(img_emb)
    # print(text_emb)
    print(text_emb.shape[0])
    print(img_emb.shape[0])    
    img_emb = search.normalize_embedding(img_emb)
    text_emb = search.normalize_embedding(text_emb)

    
    # print(img_emb)
    # print(text_emb)

    similarity = torch.nn.functional.cosine_similarity(img_emb.unsqueeze(0), text_emb.unsqueeze(0))    
    print(f"Similarity: {similarity.item()}")

