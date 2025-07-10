import streamlit as st
import torch
import torch.nn.functional as F
# YENİ: Artık transformers kütüphanesini ana bileşen olarak kullanıyoruz
from transformers import AutoTokenizer, AutoModel
import time
import math
import re


# ==============================================================================
# ADIM 1: MODEL YÜKLEME VE YARDIMCI FONKSİYONLAR
# ==============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """
    GÜNCELLENDİ: Modeli ve tokenizer'ı doğrudan Hugging Face Hub'dan yükler.
    Artık yerel dosyalara veya özel model sınıflarına ihtiyaç yok.
    """
    # YENİ: Gradio uygulamasında kullanılan modelin kimliği
    HF_REPO_ID = "GSAI-ML/LLaDA-8B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.info(f"{HF_REPO_ID} modeli yükleniyor... Bu işlem biraz zaman alabilir.")

    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, trust_remote_code=True)

    # Modeli yükle (bfloat16 ile daha az bellek kullanımı için)
    model = AutoModel.from_pretrained(
        HF_REPO_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Modern GPU'lar için bellek optimizasyonu
    ).to(device)

    model.eval()
    st.success("Model başarıyla yüklendi!")

    return model, tokenizer, device


# --- Gradio Kodundan Alınan Yardımcı Fonksiyonlar (Değişiklik yok) ---

def parse_constraints(constraints_text, tokenizer):
    """Kısıtlama metnini ayrıştırır ve token ID'lerine çevirir."""
    constraints = {}
    if not constraints_text:
        return constraints

    parts = re.findall(r'(\d+)\s*:\s*([^,]+)', constraints_text)

    for pos_str, word in parts:
        try:
            pos = int(pos_str.strip())
            # GÜNCELLENDİ: transformers tokenizer'ın encode metodu direkt liste döndürür
            tokens = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
            for i, token_id in enumerate(tokens):
                constraints[pos + i] = token_id
        except (ValueError, IndexError):
            continue
    return constraints


def add_gumbel_noise(logits, temperature):
    """Örnekleme için Gumbel gürültüsü ekler."""
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return logits + gumbel_noise * temperature


# ==============================================================================
# ADIM 2: ÜRETİM VE GÖRSELLEŞTİRME MANTIĞI
# ==============================================================================

def render_visualization_step(placeholder, step_info):
    """Her bir üretim adımını renkli token'lar olarak render eder."""
    step, total_steps, viz_tokens, prompt_len = step_info

    html_parts = []
    prompt_html = f"<div style='margin-bottom: 10px; padding: 5px; border-left: 3px solid #ccc;'>{st.session_state.get('current_prompt_text', '')}</div>"

    for token_str, color, confidence in viz_tokens:
        token_str_display = token_str.replace(' ', ' ').replace('<', '&lt;').replace('>', '&gt;')
        title = f"Güven: {confidence:.2f}" if confidence is not None else "Sabit Token"
        html_parts.append(
            f'<span title="{title}" style="background-color: {color}; color: white; padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block;">{token_str_display}</span>'
        )

    full_html = f"**Adım {step}/{total_steps}**<br>" + "".join(html_parts)
    placeholder.markdown(full_html, unsafe_allow_html=True)


def _run_llada_diffusion_st(context, gen_length, model, tokenizer, device, steps, constraints, temperature, cfg_scale, remasking):
    # <<<--- HATA İÇİN DÜZELTME: GRADIO KODUNDAKİ GİBİ SABİT ID KULLANILIYOR ---<<<
    mask_token_id = 126336
    prompt_len = context.shape[1]

    x = torch.cat([
        context,
        torch.full((1, gen_length), mask_token_id, dtype=torch.long, device=device)
    ], dim=1)

    for pos, token_id in constraints.items():
        if prompt_len + pos < x.shape[1]:
            x[:, prompt_len + pos] = token_id

    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_len] = True

    for i in range(steps):
        mask_index = (x == mask_token_id)
        if not mask_index.any():
            break

        with torch.no_grad():
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature)
        predicted_x0 = torch.argmax(logits_with_noise, dim=-1)

        probabilities = F.softmax(logits.to(torch.float64), dim=-1)
        if remasking == 'low_confidence':
            confidences = torch.gather(probabilities, -1, predicted_x0.unsqueeze(-1)).squeeze(-1)
        else:
            confidences = torch.rand_like(probabilities[:, 0])

        confidences = torch.where(mask_index, confidences, -torch.inf)

        num_to_unmask = math.ceil(((i + 1) / steps) * gen_length)
        num_already_unmasked = (x[:, prompt_len:] != mask_token_id).sum()
        num_newly_unmasked = max(0, num_to_unmask - num_already_unmasked)

        if num_newly_unmasked == 0 and i < steps - 1:
            continue

        if i < steps - 1:
            if int(num_newly_unmasked) > 0:
                _, indices_to_unmask = torch.topk(confidences.view(-1), k=int(num_newly_unmasked))
                new_tokens = torch.gather(predicted_x0.view(-1), 0, indices_to_unmask)
                x.view(-1)[indices_to_unmask] = new_tokens
        else:
            x = torch.where(mask_index, predicted_x0, x)

        for pos, token_id in constraints.items():
            if prompt_len + pos < x.shape[1]:
                x[:, prompt_len + pos] = token_id

        viz_tokens = []
        response_tokens = x[0, prompt_len:]
        response_confidences = confidences[0, prompt_len:]

        for j in range(gen_length):
            token_id = response_tokens[j].item()
            if token_id == mask_token_id:
                viz_tokens.append(('[MASK]', '#444444', 0.0))
            else:
                token_str = tokenizer.decode([token_id])
                confidence = response_confidences[j].item()
                color = f"hsl({confidence * 120}, 80%, 40%)"
                if j in constraints:
                    color = "#8A2BE2"
                viz_tokens.append((token_str, color, confidence if confidence > -1 else None))

        yield ('viz', (i + 1, steps, viz_tokens, prompt_len))

    return x[:, prompt_len:]


def generate_llada_st(messages, model, tokenizer, device, gen_length, steps, constraints_str, temperature, cfg_scale,
                      remasking):
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    st.session_state['current_prompt_text'] = prompt_text

    prompt_ids = tokenizer.encode(prompt_text)
    context = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    constraints = parse_constraints(constraints_str, tokenizer)

    reverse_process_generator = _run_llada_diffusion_st(
        context, gen_length, model, tokenizer, device,
        steps=steps, constraints=constraints, temperature=temperature,
        cfg_scale=cfg_scale, remasking=remasking
    )

    final_block = None
    while True:
        try:
            type, data = next(reverse_process_generator)
            if type == 'viz':
                yield ('viz', data)
        except StopIteration as e:
            final_block = e.value
            break

    if final_block is None: return

    # GÜNCELLENDİ: Özel token ID'leri direkt tokenizer'dan alınıyor
    eos_token_id = tokenizer.eos_token_id
    special_tokens_ids = {
        tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.cls_token_id,
        tokenizer.sep_token_id, tokenizer.unk_token_id, tokenizer.eos_token_id
    }

    block_ids = final_block[0].cpu().tolist()

    decoded_block = tokenizer.decode(block_ids, skip_special_tokens=False)
    yield ('debug', (1, block_ids, decoded_block))

    if eos_token_id in block_ids:
        eos_index = block_ids.index(eos_token_id)
        block_ids = block_ids[:eos_index]

    # Sonuç direkt olarak decode edilebilir
    new_text = tokenizer.decode(block_ids, skip_special_tokens=True)
    yield ('stream', new_text)


# ==============================================================================
# ADIM 3: STREAMLIT ARAYÜZÜ (Değişiklik yok)
# ==============================================================================

st.set_page_config(page_title="LLaDA Chatbot", layout="wide")
st.title("LLaDA Difüzyon Modeli ile Sohbet (Gelişmiş Kontroller)")

# GÜNCELLENDİ: Artık config nesnesi dönmüyor
model, tokenizer, device = load_model_and_tokenizer()

with st.sidebar:
    st.header("Üretim Ayarları")
    gen_length = st.slider("Üretim Uzunluğu", 32, 256, 64, 8)
    steps = st.slider("Örnekleme Adımları (NFE)", 4, 128, 16, 4)
    temperature = st.slider("Sıcaklık (Temperature)", 0.0, 2.0, 0.8, 0.05,
                            help="Daha yüksek değerler daha rastgele sonuçlar üretir.")
    cfg_scale = st.slider("CFG Ölçeği", 0.0, 5.0, 1.5, 0.1, help="Prompt'a ne kadar sadık kalınacağını belirler.")

    st.markdown("---")
    constraints_input = st.text_input("Kelime Kısıtlamaları (opsiyonel)", placeholder="0:Merhaba, 10:yapay zeka",
                                      help="Format: `pozisyon:kelime, pozisyon2:kelime2`")
    remasking_strategy = st.radio("Yeniden Maskeleme Stratejisi", ["low_confidence", "random"], index=0)

    st.markdown("---")
    visualize_generation = st.toggle("Üretim Adımlarını Göster", value=True)
    visualization_delay = st.slider("Görselleştirme Gecikmesi (sn)", 0.0, 1.0, 0.1, 0.05)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Merhaba! LLaDA modeliyle sohbet edebilirsiniz. Kenar çubuğundaki gelişmiş ayarları kullanabilirsiniz."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        viz_placeholder = st.empty()
        debug_log = []
        with st.expander("Geliştirici: Ham Üretim Günlüğü", expanded=False):
            debug_placeholder = st.empty()

        if not visualize_generation:
            viz_placeholder.status("Cevap üretiliyor...")

        chat_history_for_model = []
        for msg in st.session_state.messages:
            chat_history_for_model.append({"role": msg["role"], "content": msg["content"]})


        def stream_response():
            generator = generate_llada_st(
                messages=chat_history_for_model,
                model=model, tokenizer=tokenizer, device=device,
                gen_length=gen_length, steps=steps,
                constraints_str=constraints_input,
                temperature=temperature, cfg_scale=cfg_scale,
                remasking=remasking_strategy
            )
            for type, data in generator:
                if type == 'viz' and visualize_generation:
                    render_visualization_step(viz_placeholder, data)
                    time.sleep(visualization_delay)
                elif type == 'stream':
                    yield data
                elif type == 'debug':
                    block_num, block_ids, decoded_block = data
                    debug_log.append(f"**Blok {block_num}:**\n - Çözümlenmiş: `{decoded_block}`")
                    debug_placeholder.markdown("\n\n".join(debug_log))


        full_response = st.write_stream(stream_response)
        viz_placeholder.empty()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
