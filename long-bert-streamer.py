from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, models

cache_dir = "/media/arko/work/.hfcache"
model = None
tokenizer = None


def process_string(input_text: str, max_seq_length: int):
    global model, tokenizer
    if model is None or tokenizer is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-8k-retrieval",
            trust_remote_code=True, cache_dir=cache_dir, device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=max_seq_length, cache_dir=cache_dir,
        )
    _input = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        return_token_type_ids=False,
        truncation=True,
        max_length=max_seq_length,
    ).to("cuda:0")
    outputs = model(**_input)
    print(f"the type of output is {type(outputs)} and length is {len(outputs)}")
    return outputs


# processed_output = process_string(input_text="Yo Man! howdy, the weather is awesome today.",
#                                  max_seq_length=8190)
# print(processed_output["sentence_embedding"])

device = "cuda:0"
word_embedding_model = models.Transformer(
    'togethercomputer/m2-bert-80M-8k-retrieval',
    cache_dir=cache_dir, max_seq_length=8190)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode='cls',
                               pooling_mode_cls_token=True,
                               pooling_mode_mean_tokens=False)
new_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device,
                                cache_folder=cache_dir)
embedding = new_model.encode("Hello world", normalize_embeddings=False,
                             convert_to_tensor=True, show_progress_bar=True)
print(embedding)
