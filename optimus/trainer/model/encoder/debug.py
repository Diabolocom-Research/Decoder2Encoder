from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM as Base
from optimus.trainer.model.encoder.bigemma3 import Gemma3ForCausalLM as Packed

from transformers import AutoTokenizer
import torch

model = "google/gemma-3-270m"

config = Gemma3TextConfig.from_pretrained(model)
config.use_bidirectional_attention = True
# config._attn_implementation = "flash_attention_2"

base = Base.from_pretrained(model, config=config).to("mps")
pack = Packed.from_pretrained(model, config=config).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model)

phrases = [
    "Ceci est la première phrase, elle est courte.",
    "La deuxième phrase est beaucoup plus longue pour illustrer le padding et le masque d'attention.",
    "Je fais un test de séquences de longueurs différentes.",
]

resultat_classique = tokenizer(
    phrases, 
    padding=True, 
    return_tensors="pt", 
    truncation=True,
)
attention_mask = resultat_classique["attention_mask"]
token_batch = resultat_classique["input_ids"]


resultat_nopad = tokenizer(phrases, padding=False, truncation=True)
input_ids_list = [
    torch.tensor(ids, dtype=torch.int64) for ids in resultat_nopad["input_ids"]
]
seq_lens = torch.tensor([len(ids) for ids in input_ids_list])
token_pack = torch.cat(input_ids_list)
cu_seqlen = torch.cat((torch.tensor([0]), seq_lens.cumsum(dim=0)))
max_seqlen = seq_lens.max()

print(base(input_ids=token_batch, attention_mask=attention_mask)[0][2][-1])
print(pack(x=token_pack, cu_seqlens=cu_seqlen, max_seqlen=max_seqlen)[1][-1])