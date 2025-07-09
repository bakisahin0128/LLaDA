import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import time
import math
from huggingface_hub import hf_hub_download



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


# ... Diğer model sınıflarınız (RMSNorm, Attention, LLaDA_Model vs.) buraya gelecek ...
# Önceki koddan bu kısımları aynen kopyalayın.

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        cos, sin = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.wo(output)


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
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLaDA_Model(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output_head(x)
        return logits


# ==============================================================================
# ADIM 2: STREAMLIT İÇİN GEREKLİ FONKSİYONLAR
# ==============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    # Hugging Face kullanıcı adınızı ve depo adınızı buraya yazın
    HF_REPO_ID = "ackermanBaki/llada-turkce-model"  # ÖRNEK: "ahmet/llada-turkce-model"

    # Dosyaları Hugging Face Hub'dan indir (veya cache'den yükle)
    tokenizer_path = hf_hub_download(repo_id=HF_REPO_ID, filename="turkce_bpe_tokenizer.json")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="llada_sft_model_final.pt")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    config = LLaDAConfig(vocab_size=tokenizer.get_vocab_size())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Streamlit Cloud'da genellikle GPU olmaz, bu yüzden map_location='cpu' önemlidir.
    model = LLaDA_Model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer, device, config


def get_color_for_confidence(score):
    hue = score * 120
    return f"hsl({hue}, 80%, 40%)"


def render_visualization_step(placeholder, step_info):
    # step_info'dan gelen verileri aç
    step, total_steps, sequence_tensor, prompt_len, tokenizer, probabilities = step_info

    # olasılıklardan güven skorlarını (confidences) hesapla
    confidences, _ = torch.max(probabilities, dim=-1)

    mask_token_id = tokenizer.token_to_id('[MASK]')
    eos_token_id = tokenizer.token_to_id('[EOS]')

    # sequence_tensor, prompt + blok içerir.
    decoded_tokens = [tokenizer.id_to_token(tid) for tid in sequence_tensor[0].tolist()]
    html_parts = []

    # Döngü hala global indeksleri kullanıyor, bu doğru.
    for i in range(prompt_len, len(decoded_tokens)):
        token_id = sequence_tensor[0, i]
        token_str = decoded_tokens[i].replace(' ', ' ')

        # DÜZELTME BURADA: Global 'i' indeksini, yerel 'local_i' indeksine çeviriyoruz.
        local_i = i - prompt_len

        if token_id == mask_token_id:
            html_parts.append(
                f'<span style="background-color: #333; color: #fff; padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block;">[MASK]</span>')
        elif token_id == eos_token_id:
            html_parts.append(
                f'<span style="background-color: #8A2BE2; color: #fff; padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block; font-weight: bold;">[EOS]</span>')
        else:
            # Artık güvenli olan yerel indeksi kullanıyoruz.
            confidence_score = confidences[0, local_i].item()
            color = get_color_for_confidence(confidence_score)
            title = f"Güven: {confidence_score:.2f}"
            html_parts.append(
                f'<span title="{title}" style="background-color: {color}; color: white; padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block;">{token_str}</span>')

    full_html = f"**Adım {total_steps - step + 1}/{total_steps}**<br>" + "".join(html_parts)
    placeholder.markdown(full_html, unsafe_allow_html=True)


# ==============================================================================
# SADECE BU FONKSİYONU GÜNCELLEYİN
# ==============================================================================

def _run_reverse_process_st(sequence_so_far, masked_region_len, model, tokenizer, device, num_sampling_steps):
    """
    Kademeli Dondurma (Gradual Freezing) stratejisi ile güncellenmiş LLaDA ters difüzyon süreci.
    Bu versiyon, yüksek güvenle tahmin edilen token'ları kilitleyerek "yanıp sönme" (flickering) sorununu çözer.
    """
    mask_token_id = tokenizer.token_to_id('[MASK]')
    prompt_len = sequence_so_far.shape[1]

    # Başlangıç durumu: Üretilecek bölge tamamen maskeli
    current_xt_block = torch.full((1, masked_region_len), mask_token_id, device=device, dtype=torch.long)

    # Hangi token'ların bilindiğini (maskesiz olduğunu) takip etmek için bir maske.
    # Başlangıçta hepsi bilinmiyor (False).
    known_mask = torch.zeros_like(current_xt_block, dtype=torch.bool)

    # Difüzyon adımlarını (zamanı) tanımla: 1.0 -> ~0.0
    timesteps = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device)

    for i in range(num_sampling_steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]

        # Tam diziyi oluştur: [PROMPT] + [MEVCUT_GÜRÜLTÜLÜ_BLOK]
        full_sequence = torch.cat([sequence_so_far, current_xt_block], dim=1)

        # 1. Modeli kullanarak "temiz" halin (x0) tahminini yap
        with torch.no_grad():
            logits = model(full_sequence)

        probabilities = F.softmax(logits, dim=-1)
        # Sadece üretim bloğuna ait olasılıkları ve tahminleri al
        block_probabilities = probabilities[:, prompt_len:]
        confidences, predicted_x0_block = torch.max(block_probabilities, dim=-1)

        # Görselleştirme için mevcut durumu yield et
        yield ('viz',
               (num_sampling_steps - i, num_sampling_steps, full_sequence, prompt_len, tokenizer, block_probabilities))

        # Eğer son adımdaysak, döngüyü kır ve son tahmini döndür
        if t_next == 0:
            current_xt_block = predicted_x0_block
            break

        # 2. Kademeli Dondurma: Hangi token'ları açacağımızı belirle

        # Bu adımda kaç token'ın bilinir (maskesiz) olması gerektiğini hesapla
        num_known_target = math.ceil((1 - t_next) * masked_region_len)
        # Şu an kaç tanesinin bilindiğini hesapla
        num_known_current = known_mask.sum().item()
        # Bu adımda kaç tane YENİ token açmamız gerektiğini bul
        num_newly_unmasked = max(0, num_known_target - num_known_current)

        # Eğer açılacak yeni token yoksa, bir sonraki adıma geç
        if num_newly_unmasked == 0:
            continue

        # 3. En İyi Adayları Seçme

        # Sadece hala maskeli olan pozisyonların güven skorlarını dikkate al.
        # Zaten bilinen (kilitlenmiş) pozisyonların güvenini -sonsuz yaparak onları seçimden çıkar.
        candidate_confidences = confidences.clone()
        candidate_confidences[known_mask] = -torch.inf

        # En yüksek güvene sahip 'num_newly_unmasked' adet adayı seç
        _, indices_to_unmask = torch.topk(candidate_confidences, k=num_newly_unmasked, dim=-1)

        # 4. Durumu Güncelleme

        # Seçilen pozisyonlardaki token'ları modelin tahminleriyle doldur
        new_tokens = torch.gather(predicted_x0_block, 1, indices_to_unmask)
        current_xt_block.scatter_(1, indices_to_unmask, new_tokens)

        # Bu yeni pozisyonları artık "biliniyor" olarak işaretle (kilitle)
        known_mask.scatter_(1, indices_to_unmask, True)

    # Döngü bittiğinde son üretilen temiz bloğu döndür
    return current_xt_block


# GÜNCELLENDİ: Artık debug bilgisi de üretiyor
def generate_llada_st(prompt_text, model, tokenizer, device, block_size, num_sampling_steps, max_blocks):
    eos_token_id = tokenizer.token_to_id('[EOS]')
    special_tokens_ids = {tokenizer.token_to_id(t) for t in ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]', '[EOS]'] if
                          tokenizer.token_to_id(t) is not None}
    prompt_encoded = tokenizer.encode(prompt_text)
    context = torch.tensor(prompt_encoded.ids).unsqueeze(0).to(device)

    for block_num in range(max_blocks):
        reverse_process_generator = _run_reverse_process_st(context, block_size, model, tokenizer, device,
                                                            num_sampling_steps)
        final_block = None
        while True:
            try:
                type, data = next(reverse_process_generator)
                if type == 'viz':
                    yield ('viz', data)
            except StopIteration as e:
                final_block = e.value
                break

        if final_block is None: continue

        block_ids = final_block[0].cpu().tolist()

        # YENİ: Ham blok bilgisini debug için yield et
        decoded_block = tokenizer.decode(block_ids)
        yield ('debug', (block_num + 1, block_ids, decoded_block))

        eos_found = False
        if eos_token_id in block_ids:
            eos_index = block_ids.index(eos_token_id)
            block_ids = block_ids[:eos_index]
            eos_found = True

        ids_to_decode = [tid for tid in block_ids if tid not in special_tokens_ids]
        if ids_to_decode:
            new_text = tokenizer.decode(ids_to_decode)
            yield ('stream', new_text)

        if eos_found:
            break

        context = torch.cat([context, final_block], dim=1)


# ==============================================================================
# ADIM 3: STREAMLIT ARAYÜZÜNÜN ANA KODU (Debug bölümü eklendi)
# ==============================================================================

st.set_page_config(page_title="LLaDA Chatbot", layout="wide")
st.title("LLaDA Difüzyon Modeli ile Sohbet")

model, tokenizer, device, config = load_model_and_tokenizer()

with st.sidebar:
    st.header("Üretim Ayarları")
    block_size = st.slider("Blok Boyutu", 2, 32, 8, 2)
    num_sampling_steps = st.slider("Örnekleme Adımları (NFE)", 4, 128, 16, 4)
    max_blocks = st.slider("Maksimum Blok Sayısı", 5, 100, 50, 5)
    visualize_generation = st.toggle("Üretim Adımlarını Göster", value=True)
    visualization_delay = st.slider("Görselleştirme Gecikmesi (sn)", 0.0, 1.0, 0.1, 0.05)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! LLaDA modeliyle sohbet edebilirsiniz."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        viz_placeholder = st.empty()

        # YENİ: Debug alanı için açılır menü ve yer tutucu
        debug_log = []
        with st.expander("Geliştirici: Ham Üretim Günlüğü", expanded=False):
            debug_placeholder = st.empty()

        if not visualize_generation:
            viz_placeholder.status("Cevap üretiliyor...")

        history_prompt = ""
        for msg in st.session_state.messages[:-1]:
            role = "Kullanıcı" if msg["role"] == "user" else "Model"
            history_prompt += f"{role}: {msg['content']}\n"
        full_prompt_text = history_prompt + f"Kullanıcı: {prompt}\nModel: "


        def stream_response():
            generator = generate_llada_st(
                prompt_text=full_prompt_text,
                model=model, tokenizer=tokenizer, device=device,
                block_size=block_size, num_sampling_steps=num_sampling_steps,
                max_blocks=max_blocks
            )
            for type, data in generator:
                if type == 'viz' and visualize_generation:
                    render_visualization_step(viz_placeholder, data)
                    time.sleep(visualization_delay)
                elif type == 'stream':
                    yield data
                # YENİ: Debug bilgisini işle
                elif type == 'debug':
                    block_num, block_ids, decoded_block = data
                    debug_log.append(
                        f"**Blok {block_num}:**\n - ID'ler: `{block_ids}`\n - Çözümlenmiş: `{decoded_block}`")
                    debug_placeholder.markdown("\n\n".join(debug_log))


        full_response = st.write_stream(stream_response)
        viz_placeholder.empty()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
