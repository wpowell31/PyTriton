import logging
 
import numpy as np
from transformers import BertTokenizer, FlaxBertModel
 
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
 
logger = logging.getLogger("examples.huggingface_bert_jax.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("bert-base-uncased")
 
 
@batch
def _infer_fn(**inputs: np.ndarray):
	(sequence_batch,) = inputs.values()
 
	# need to convert dtype=object to bytes first
	# end decode unicode bytes
	sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
 
	last_hidden_states = []
	for sequence_item in sequence_batch:
    		tokenized_sequence = tokenizer(sequence_item.item(), return_tensors="jax")
    		results = model(**tokenized_sequence)
    		last_hidden_states.append(results.last_hidden_state)
	last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
	return [last_hidden_states]
 

with Triton() as triton:
    logger.info("Loading BERT model.")
    triton.bind(
        model_name="BERT",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequence", dtype=np.bytes_, shape=(1,)),
        ],
        outputs=[
            Tensor(name="last_hidden_state", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=16),
    )
    logger.info("Serving inference")
    triton.serve()