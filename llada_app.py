import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import math

# ==============================================================================
# ADIM 1: LLaDA MODEL SINIFLARINIZ (Değişiklik yok)
# ==============================================================================
class LLaDAConfig:
    def __init__(self, vocab_size=50000, max_seq_len=512, d_model=512, n_layers=16, n_heads=4, dropout=0.1):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ffn = 4 * d_model
        self.dropout = dropout

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
    def forward(self, x):
        return self.cos[:x.shape[2], :], self.sin[:x.shape[2], :]

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1) * sin)
    k_embed = (k * cos) + (torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.wq = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.wo = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        self.rotary_emb = RotaryPositionalEmbedding(self.d_head, config.max_seq_len)
    def forward(self, x: torch.Tensor):
        b, seq_len, _ = x.size()
        q = self.wq(x).view(b, seq_len, self.n_heads, self.d_head).transpose(1,2)
        k = self.wk(x).view(b, seq_len, self.n_heads, self.d_head).transpose(1,2)
        v = self.wv(x).view(b, seq_len, self.n_heads, self.d_head).transpose(1,2)
        cos, sin = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1,2).contiguous().view(b, seq_len, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ffn, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn  = FeedForward(config)
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm  = RMSNorm(config.d_model)
    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))

class LLaDA_Model(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm   = RMSNorm(config.d_model)
        self.head   = nn.Linear(config.d_model, config.vocab_size, bias=False)
    def forward(self, tokens):
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

# ==============================================================================
# ADIM 2: Hugging Face’dan Çekecek Şekilde Güncellenmiş Loader
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    repo_id = "ackermanBaki/llada-turkish"  # ← kendi HF repo’nuzu yazın
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(repo_id)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    config = LLaDAConfig(vocab_size=tokenizer.vocab_size)
    return model, tokenizer, device, config

def get_color_for_confidence(score):
    hue = score * 120
    return f"hsl({hue}, 80%, 40%)"

def render_visualization_step(placeholder, step_info):
    step, total, seq, prompt_len, tokenizer, probs = step_info
    confs, _ = torch.max(probs, dim=-1)
    decoded = tokenizer.convert_ids_to_tokens(seq[0].tolist())
    html = []
    for i in range(prompt_len, len(decoded)):
        tok = decoded[i]
        if tok == tokenizer.mask_token:
            html.append(f'<span style="background:#333;color:#fff;padding:2px;border-radius:4px;">[MASK]</span>')
        elif tok == tokenizer.eos_token:
            html.append(f'<span style="background:#8A2BE2;color:#fff;padding:2px;border-radius:4px;">[EOS]</span>')
        else:
            score = confs[0, i-prompt_len].item()
            color = get_color_for_confidence(score)
            html.append(f'<span title="Güven: {score:.2f}" style="background:{color};padding:2px;border-radius:4px;">{tok}</span>')
    placeholder.markdown(f"**Adım {total-step+1}/{total}**<br>" + "".join(html), unsafe_allow_html=True)

def _run_reverse_process_st(ctx, block_len, model, tokenizer, device, steps):
    mask_id = tokenizer.mask_token_id
    prompt_len = ctx.shape[1]
    current = torch.full((1, block_len), mask_id, dtype=torch.long, device=device)
    known = torch.zeros_like(current, dtype=torch.bool)
    timesteps = torch.linspace(1.0, 0.0, steps+1, device=device)
    for i in range(steps):
        seq = torch.cat([ctx, current], dim=1)
        with torch.no_grad():
            logits = model(seq).logits if hasattr(model, "logits") else model(seq)
        probs = F.softmax(logits, dim=-1)[:, prompt_len:]
        confs, preds = torch.max(probs, dim=-1)
        yield ("viz", (steps-i, steps, seq, prompt_len, tokenizer, probs))
        if timesteps[i+1] == 0:
            current = preds; break
        target_known = math.ceil((1-timesteps[i+1])*block_len)
        to_unmask = max(0, target_known - known.sum().item())
        if to_unmask>0:
            cand = confs.clone(); cand[known]= -float("inf")
            _, idxs = torch.topk(cand, k=to_unmask, dim=-1)
            new_toks = torch.gather(preds, 1, idxs)
            current.scatter_(1, idxs, new_toks)
            known.scatter_(1, idxs, True)
    return current

def generate_llada_st(prompt, model, tokenizer, device, block_size, num_steps, max_blocks):
    eos_id = tokenizer.eos_token_id
    specials = set(tokenizer.all_special_ids)
    enc = tokenizer(prompt, add_special_tokens=False)
    ctx = torch.tensor([enc["input_ids"]], device=device)
    for b in range(max_blocks):
        gen = _run_reverse_process_st(ctx, block_size, model, tokenizer, device, num_steps)
        final = None
        while True:
            try:
                typ, data = next(gen)
                if typ=="viz": yield ("viz", data)
            except StopIteration as e:
                final = e.value; break
        block = final[0].cpu().tolist()
        # debug
        yield ("debug", (b+1, block, tokenizer.decode(block)))
        if eos_id in block:
            idx = block.index(eos_id)
            block = block[:idx]
            done = True
        else:
            done = False
        ids = [i for i in block if i not in specials]
        if ids:
            yield ("stream", tokenizer.decode(ids))
        if done: break
        ctx = torch.cat([ctx, final], dim=1)

# ==============================================================================
# ADIM 3: STREAMLIT ARAYÜZÜNÜN ANA KODU
# ==============================================================================
st.set_page_config(page_title="LLaDA Chatbot", layout="wide")
st.title("LLaDA Difüzyon Modeli ile Sohbet")

model, tokenizer, device, config = load_model_and_tokenizer()

with st.sidebar:
    st.header("Üretim Ayarları")
    block_size           = st.slider("Blok Boyutu", 2, 32, 8, 2)
    num_sampling_steps   = st.slider("Örnekleme Adımları (NFE)", 4, 128, 16, 4)
    max_blocks           = st.slider("Maksimum Blok Sayısı", 5, 100, 50, 5)
    visualize_generation = st.toggle("Üretim Adımlarını Göster", value=True)
    visualization_delay  = st.slider("Görselleştirme Gecikmesi (sn)", 0.0, 1.0, 0.1, 0.05)

if "messages" not in st.session_state:
    st.session_state.messages = [ {"role":"assistant","content":"Merhaba! LLaDA modeliyle sohbet edebilirsiniz."} ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        viz = st.empty()
        debug_log = []; debug_pl = None
        with st.expander("Geliştirici: Ham Üretim Günlüğü", expanded=False):
            debug_pl = st.empty()
        if not visualize_generation:
            viz.status("Cevap üretiliyor...")
        history = ""
        for m in st.session_state.messages[:-1]:
            speaker = "Kullanıcı" if m["role"]=="user" else "Model"
            history += f"{speaker}: {m['content']}\n"
        full_prompt = history + f"Kullanıcı: {prompt}\nModel: "
        def streamer():
            for typ, data in generate_llada_st(
                prompt=full_prompt,
                model=model, tokenizer=tokenizer, device=device,
                block_size=block_size, num_steps=num_sampling_steps, max_blocks=max_blocks
            ):
                if typ=="viz" and visualize_generation:
                    render_visualization_step(viz, data)
                    time.sleep(visualization_delay)
                elif typ=="stream":
                    yield data
                elif typ=="debug":
                    bn, ids, txt = data
                    debug_log.append(f"**Blok {bn}:**\n- ID'ler: `{ids}`\n- Çözümlenmiş: `{txt}`")
                    debug_pl.markdown("\n\n".join(debug_log))
        full_resp = st.write_stream(streamer())
        viz.empty()
        st.session_state.messages.append({"role":"assistant","content":full_resp})
