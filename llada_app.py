import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
import time
import math

# ==============================================================================
# ADIM 1: LLaDA MODEL SINIFLARINIZ (Değişiklik yok)
# ==============================================================================
class LLaDAConfig:
    def __init__(self, vocab_size=50000, max_seq_len=512, d_model=128, n_layers=16, n_heads=8, dropout=0.1):
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
    q_embed = (q * cos) + (torch.cat([-q[...,1::2], q[...,::2]], dim=-1) * sin)
    k_embed = (k * cos) + (torch.cat([-k[...,1::2], k[...,::2]], dim=-1) * sin)
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
# ADIM 2: HF SNAPSHOT_DOWNLOAD İLE LOADER (GÜNCELLENDİ)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # 1) HF’den indir
    repo_id = "kullanici_adiniz/llada-turkish"
    local_dir = snapshot_download(repo_id)

    # 2) Tokenizer
    tok_path = f"{local_dir}/tokenizer.json"
    tokenizer = Tokenizer.from_file(tok_path)

    # 3) Model + state_dict
    config = LLaDAConfig(vocab_size=tokenizer.get_vocab_size())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaDA_Model(config).to(device)

    # 4) Dosyayı yükle
    ckpt = torch.load(f"{local_dir}/pytorch_model.bin", map_location=device)
    # Eğer tam bir dict ise önce doğru alt-sözlüğü alın
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 5) Gerekirse "module." prefix’lerini kaldır
    def strip_prefix(sd):
        return {k[len("module."):]: v if k.startswith("module.") else (k, v)[0]: v for k, v in sd.items()}
    # Uygula:
    if any(k.startswith("module.") for k in state_dict):
        state_dict = strip_module_prefix(state_dict)

    # 6) Model’e yükle
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device, config

def get_color_for_confidence(score):
    hue = score * 120
    return f"hsl({hue}, 80%, 40%)"

def render_visualization_step(placeholder, step_info):
    step, total, seq, prompt_len, tokenizer, probs = step_info
    confs, _ = torch.max(probs, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(seq[0].tolist())
    html_parts = []
    for i in range(prompt_len, len(tokens)):
        tok = tokens[i]
        if tok == tokenizer.mask_token:
            html_parts.append(
                '<span style="background:#333;color:#fff;padding:2px;border-radius:4px;">[MASK]</span>')
        elif tok == tokenizer.eos_token:
            html_parts.append(
                '<span style="background:#8A2BE2;color:#fff;padding:2px;border-radius:4px;">[EOS]</span>')
        else:
            score = confs[0, i-prompt_len].item()
            color = get_color_for_confidence(score)
            html_parts.append(
                f'<span title="Güven: {score:.2f}" style="background:{color};padding:2px;border-radius:4px;">{tok}</span>')
    placeholder.markdown(f"**Adım {total-step+1}/{total}**<br>" + "".join(html_parts),
                         unsafe_allow_html=True)

def _run_reverse_process_st(ctx, block_len, model, tokenizer, device, steps):
    mask_id = tokenizer.token_to_id("[MASK]")
    prompt_len = ctx.shape[1]
    current = torch.full((1, block_len), mask_id, dtype=torch.long, device=device)
    known = torch.zeros_like(current, dtype=torch.bool)
    timesteps = torch.linspace(1.0, 0.0, steps+1, device=device)

    for i in range(steps):
        seq = torch.cat([ctx, current], dim=1)
        with torch.no_grad():
            logits = model(seq)
        probs = F.softmax(logits, dim=-1)[:, prompt_len:]
        confs, preds = torch.max(probs, dim=-1)

        yield ("viz", (steps-i, steps, seq, prompt_len, tokenizer, probs))

        if timesteps[i+1] == 0:
            current = preds
            break

        target_known = math.ceil((1 - timesteps[i+1]) * block_len)
        newly = max(0, target_known - known.sum().item())
        if newly > 0:
            cand = confs.clone()
            cand[known] = -float("inf")
            _, idxs = torch.topk(cand, newly, dim=-1)
            new_toks = torch.gather(preds, 1, idxs)
            current.scatter_(1, idxs, new_toks)
            known.scatter_(1, idxs, True)

    return current

def generate_llada_st(prompt, model, tokenizer, device, block_size, num_steps, max_blocks):
    eos_id = tokenizer.token_to_id("[EOS]")
    specials = set(tokenizer.all_special_ids)
    enc = tokenizer.encode(prompt)
    ctx = torch.tensor([enc.ids], device=device)

    for b in range(max_blocks):
        gen = _run_reverse_process_st(ctx, block_size, model, tokenizer, device, num_steps)
        final = None
        while True:
            try:
                typ, data = next(gen)
                if typ == "viz":
                    yield ("viz", data)
            except StopIteration as e:
                final = e.value
                break

        block = final[0].cpu().tolist()
        yield ("debug", (b+1, block, tokenizer.decode(block)))

        done = False
        if eos_id in block:
            idx = block.index(eos_id)
            block = block[:idx]
            done = True

        ids = [i for i in block if i not in specials]
        if ids:
            yield ("stream", tokenizer.decode(ids))

        if done:
            break
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
    st.session_state.messages = [{"role":"assistant","content":"Merhaba! LLaDA ile sohbet edebilirsiniz."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Mesajınızı yazın…"):
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        viz_pl = st.empty()
        debug_log = []
        with st.expander("Geliştirici: Ham Üretim Günlüğü", expanded=False):
            debug_pl = st.empty()
        if not visualize_generation:
            viz_pl.status("Cevap üretiliyor…")

        history = ""
        for m in st.session_state.messages[:-1]:
            who = "Kullanıcı" if m["role"]=="user" else "Model"
            history += f"{who}: {m['content']}\n"
        full_prompt = history + f"Kullanıcı: {user_input}\nModel: "

        def streamer():
            for typ, data in generate_llada_st(
                prompt=full_prompt,
                model=model, tokenizer=tokenizer, device=device,
                block_size=block_size, num_steps=num_sampling_steps, max_blocks=max_blocks
            ):
                if typ=="viz" and visualize_generation:
                    render_visualization_step(viz_pl, data)
                    time.sleep(visualization_delay)
                elif typ=="stream":
                    yield data
                elif typ=="debug":
                    bn, ids, txt = data
                    debug_log.append(f"**Blok {bn}:**\n- ID’ler: `{ids}`\n- Çözümlenmiş: `{txt}`")
                    debug_pl.markdown("\n\n".join(debug_log))

        response = st.write_stream(streamer())
        viz_pl.empty()
        st.session_state.messages.append({"role":"assistant","content":response})
