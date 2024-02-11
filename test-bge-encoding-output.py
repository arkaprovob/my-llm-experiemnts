from sentence_transformers import SentenceTransformer

sentences_1 = ["Yo Man! howdy, the weather is awesome today."]
cache_dir = "/media/arko/work/.hfcache"
device = "cuda:0"
model = SentenceTransformer('BAAI/bge-large-zh-v1.5',
                            cache_folder=cache_dir, device=device)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=False, convert_to_tensor=True,
                            show_progress_bar=True)
print(f"the type of output is {type(embeddings_1)} and length is {len(embeddings_1)}")
#print(embeddings_1)


model = SentenceTransformer('togethercomputer/m2-bert-80M-8k-retrieval',
                            cache_folder=cache_dir, device=device)
print(model)