import streamlit as st
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import time
import math

# ==============================================================================
# ADIM 1: (Opsiyonel) Eğer kendi LLaDA_Model’inizi kullanacaksanız buraya gelir.
#           Fakat trust_remote_code=True ile Hub’daki modeling_llada.py yüklenecek.
# ==============================================================================

# ==============================================================================
# ADIM 2: Hugging Face’dan Çekecek Şekilde Güncellenmiş Loader
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    repo_id = "ackermanBaki/llada-turkish"   # ← kendi HF repo’nuzu yazın

    # 1) Config’u indir ve custom config sınıfını kullan
    config = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=True
    )

    # 2) Tokenizer’ı indir
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True,
        trust_remote_code=True
    )

    # 3) Modeli indir
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        config=config,
        trust_remote_code=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    return model, tokenizer, device, config


def get_color_for_confidence(score: float) -> str:
    hue = int(score * 120)
    return f"hsl({hue}, 80%, 40%)"


def render_visualization_step(placeholder, step_info):
    step, total, seq_tensor, prompt_len, tokenizer, probs = step_info
    confidences, _ = torch.max(probs, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(seq_tensor[0].tolist())

    spans = []
    for i in range(prompt_len, len(tokens)):
        tok = tokens[i]
        if tok == tokenizer.mask_token:
            spans.append(
                '<span style="background:#333;color:#fff;padding:2px;border-radius:4px;">[MASK]</span>'
            )
        elif tok == tokenizer.eos_token:
            spans.append(
                '<span style="background:#8A2BE2;color:#fff;padding:2px;border-radius:4px;">[EOS]</span>'
            )
        else:
            conf = confidences[0, i - prompt_len].item()
            color = get_color_for_confidence(conf)
            spans.append(
                f'<span title="Güven: {conf:.2f}" '
                f'style="background:{color};padding:2px;border-radius:4px;">{tok}</span>'
            )

    html = f"**Adım {total-step+1}/{total}**<br>" + "".join(spans)
    placeholder.markdown(html, unsafe_allow_html=True)


def _run_reverse_process_st(ctx, block_len, model, tokenizer, device, steps):
    mask_id = tokenizer.mask_token_id
    prompt_len = ctx.shape[1]
    current = torch.full((1, block_len), mask_id, device=device, dtype=torch.long)
    known   = torch.zeros_like(current, dtype=torch.bool)
    timesteps = torch.linspace(1.0, 0.0, steps+1, device=device)

    for i in range(steps):
        seq = torch.cat([ctx, current], dim=1)
        with torch.no_grad():
            out = model(seq)
            logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=-1)[:, prompt_len:]
        confs, preds = torch.max(probs, dim=-1)

        # görselleştirme
        yield ("viz", (steps-i, steps, seq, prompt_len, tokenizer, probs))

        # son adım
        if timesteps[i+1] == 0:
            current = preds
            break

        # kaç token açılacak?
        target = math.ceil((1-timesteps[i+1]) * block_len)
        to_unmask = max(0, target - known.sum().item())
        if to_unmask > 0:
            cand = confs.clone()
            cand[known] = -float("inf")
            _, idxs = torch.topk(cand, k=to_unmask, dim=-1)
            new_toks = torch.gather(preds, 1, idxs)
            current.scatter_(1, idxs, new_toks)
            known.scatter_(1, idxs, True)

    return current


def generate_llada_st(prompt, model, tokenizer, device,
                      block_size, num_steps, max_blocks):
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
                if typ == "viz":
                    yield ("viz", data)
            except StopIteration as e:
                final = e.value
                break

        block_ids = final[0].cpu().tolist()
        # debug log
        yield ("debug", (b+1, block_ids, tokenizer.decode(block_ids)))

        # EOS kontrolü
        done = False
        if eos_id in block_ids:
            idx = block_ids.index(eos_id)
            block_ids = block_ids[:idx]
            done = True

        # akışa alacağı token’lar
        ids = [i for i in block_ids if i not in specials]
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
    visualization_delay  = st.slider("Gecikme (sn)", 0.0, 1.0, 0.1, 0.05)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! LLaDA modeliyle sohbet edebilirsiniz."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        viz_pl   = st.empty()
        debug_log = []
        debug_pl = st.expander("Geliştirici: Ham Üretim Günlüğü", expanded=False).empty()
        if not visualize_generation:
            viz_pl.status("Cevap üretiliyor...")

        history = ""
        for m in st.session_state.messages[:-1]:
            who = "Kullanıcı" if m["role"] == "user" else "Model"
            history += f"{who}: {m['content']}\n"
        full_prompt = history + f"Kullanıcı: {prompt}\nModel: "

        def streamer():
            for typ, data in generate_llada_st(
                full_prompt, model, tokenizer, device,
                block_size, num_sampling_steps, max_blocks
            ):
                if typ == "viz" and visualize_generation:
                    render_visualization_step(viz_pl, data)
                    time.sleep(visualization_delay)
                elif typ == "stream":
                    yield data
                elif typ == "debug":
                    blk, ids, txt = data
                    debug_log.append(f"**Blok {blk}:**\n- ID: `{ids}`\n- Metin: `{txt}`")
                    debug_pl.markdown("\n\n".join(debug_log))

        full_resp = st.write_stream(streamer())
        viz_pl.empty()
        st.session_state.messages.append({"role": "assistant", "content": full_resp})
