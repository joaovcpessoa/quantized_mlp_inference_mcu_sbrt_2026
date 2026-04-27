import os
import time
import struct
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from codecarbon import EmissionsTracker

tracker    = EmissionsTracker(measure_power_secs=0.1, save_to_file=True)
data_dir   = r'C:\Users\jvt\Downloads\data'
epochs     = 10
lr         = 1e-3
batch_size = 256
n_mc       = 10
n_export   = 10
seed       = 0

INPUT_SIZE  = 784
HIDDEN_SIZE = 10
OUTPUT_SIZE = 10

torch.manual_seed(seed)
np.random.seed(seed)
DEVICE = torch.device('cpu')
print(f'Dispositivo: {DEVICE}')

MEAN = (0.1307,)
STD  = (0.3081,)

transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

def dataset_to_numpy(ds):
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X, y = next(iter(loader))
    return X.view(len(ds), -1).numpy(), y.numpy()

X_train_full, y_train_full = dataset_to_numpy(train_dataset)
X_test, y_test             = dataset_to_numpy(test_dataset)
print(f'Train: {X_train_full.shape} | Test: {X_test.shape}')

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def forward_numpy(self, x_np: np.ndarray):
        with torch.no_grad():
            x      = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z1     = self.fc1(x)
            h      = torch.relu(z1)
            logits = self.fc2(h)
        return z1.squeeze().cpu().numpy(), h.squeeze().cpu().numpy(), logits.squeeze().cpu().numpy()

def get_weights_numpy(model: MLP):
    W1 = model.fc1.weight.detach().cpu().numpy().T
    b1 = model.fc1.bias.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy().T
    b2 = model.fc2.bias.detach().cpu().numpy()
    return W1, b1, W2, b2

def train_model(X_tr, y_tr, epochs, seed=seed, verbose=True, tag=''):
    torch.manual_seed(seed)
    model     = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t    = torch.tensor(X_tr, dtype=torch.float32)
    y_t    = torch.tensor(y_tr, dtype=torch.long)
    loader = DataLoader(torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss, total_correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss    += loss.item() * len(xb)
            total_correct += (logits.argmax(1) == yb).sum().item()

        n = len(X_tr)
        if verbose and (ep + 1) % max(1, epochs // 4) == 0:
            print(f'[{tag}] Epoch {ep+1:3d} | Loss={total_loss/n:.4f} | Acc={total_correct/n:.4f}')
    return model

def accuracy_numpy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_t)
        preds  = logits.argmax(1).cpu().numpy()
    return float(np.mean(preds == y))

print(f'Monte Carlo - {n_mc} runs')
accs_mc = []
t0 = time.time()
try:
    for seed in range(n_mc):
        rng  = np.random.default_rng(seed)
        idx  = rng.permutation(len(X_train_full))
        X_s, y_s = X_train_full[idx], y_train_full[idx]
        n_tr     = int(0.8 * len(X_s))
        X_tr, y_tr = X_s[:n_tr], y_s[:n_tr]
        X_va, y_va = X_s[n_tr:], y_s[n_tr:]
        m   = train_model(X_tr, y_tr, epochs=epochs, seed=seed, verbose=False)
        acc = accuracy_numpy(m, X_va, y_va)
        accs_mc.append(acc)
        print(f'Run {seed+1:2d}/{n_mc} | Val Acc = {acc:.4f}')
finally:
    duration = time.time() - t0

accs_mc = np.array(accs_mc)
print(f'Precisao media : {accs_mc.mean():.4f}')
print(f'Desvio std     : {accs_mc.std():.4f}')
print(f'Min / Max      : {accs_mc.min():.4f} / {accs_mc.max():.4f}')
print(f'Tempo total    : {time.time()-t0:.1f}s')

model_final = train_model(X_train_full, y_train_full, epochs=epochs, seed=0, verbose=True, tag='FINAL')
t0 = time.time()
tracker.start()
test_acc            = accuracy_numpy(model_final, X_test, y_test)
emissions_inference = tracker.stop()
duration            = time.time() - t0

energy_kwh    = tracker.final_emissions_data.energy_consumed
energy_wh     = energy_kwh * 1000
energy_joules = energy_wh  * 3600

print('='*42)
print(f'Precisao final    : {test_acc:.4f}')
print(f'Tempo total       : {duration:.1f}s')
print(f'Energia consumida : {energy_joules:.2f} J  /  {energy_wh:.4f} Wh')
print(f'CO2 estimado      : {emissions_inference:.6f} kg')

_NP_WEIGHT = {8: np.int8,  16: np.int16, 32: np.int32}
_NP_BIAS   = {8: np.int32, 16: np.int32, 32: np.int64}
_C_WEIGHT  = {8: 'int8_t',  16: 'int16_t', 32: 'int32_t'}
_C_BIAS    = {8: 'int32_t', 16: 'int32_t', 32: 'int64_t'}
_C_ACCUM   = {8: 'int32_t', 16: 'int64_t', 32: 'int64_t'}

def calc_scale(arr: np.ndarray, bits: int) -> float:
    return float(np.max(np.abs(arr))) / (2**(bits - 1) - 1) + 1e-8

def quantize_multiplier(m: float):
    if m == 0.0:
        return 0, 0
    shift = 0
    while m < 0.5 and shift < 31:
        m *= 2
        shift += 1
    mult = min(int(np.round(m * (1 << 31))), (1 << 31) - 1)
    return mult, shift

def fp_requant(acc: int, mult: int, shift: int, out_min: int, out_max: int) -> int:
    x = (int(acc) * mult) >> 31
    x >>= shift
    return max(out_min, min(out_max, x))

def dequantize_logits(lq_int: np.ndarray, s_logits: float) -> np.ndarray:
    return lq_int.astype(np.float64) * s_logits

def quantize_model(model: MLP, X_calib: np.ndarray, bits: int) -> dict:
    print(f'\n{"="*42}')
    print(f'Quantizando modelo - INT{bits}')
    print(f'{"="*42}')

    W1, b1, W2, b2 = get_weights_numpy(model)
    s_W1 = calc_scale(W1, bits)
    s_W2 = calc_scale(W2, bits)

    calib_idx = np.random.choice(len(X_calib), size=min(2000, len(X_calib)), replace=False)
    h_all, logits_all, s_x_vals = [], [], []
    model.eval()
    for i in calib_idx:
        _, h, logits = model.forward_numpy(X_calib[i])
        h_all.append(h)
        logits_all.append(logits)
        s_x_vals.append(np.max(np.abs(X_calib[i])))

    s_x      = float(np.max(s_x_vals)) / (2**(bits - 1) - 1) + 1e-8
    s_h      = calc_scale(np.array(h_all),      bits)
    s_logits = calc_scale(np.array(logits_all), bits)

    clip_max = 2**(bits - 1) - 1
    clip_min = -(2**(bits - 1))

    W1_q = np.clip(np.round(W1 / s_W1), clip_min, clip_max).astype(_NP_WEIGHT[bits])
    W2_q = np.clip(np.round(W2 / s_W2), clip_min, clip_max).astype(_NP_WEIGHT[bits])
    b1_q = np.round(b1 / (s_x  * s_W1)).astype(_NP_BIAS[bits])
    b2_q = np.round(b2 / (s_h  * s_W2)).astype(_NP_BIAS[bits])

    M1 = (s_x * s_W1) / s_h
    M2 = (s_h * s_W2) / s_logits
    MULT1, SHIFT1 = quantize_multiplier(M1)
    MULT2, SHIFT2 = quantize_multiplier(M2)

    def calc_snr(orig: np.ndarray, quant: np.ndarray) -> float:
        noise = orig - quant
        pn    = np.mean(noise**2)
        return 10.0 * np.log10(np.mean(orig**2) / pn) if pn > 0 else np.inf

    def calc_rmse(orig: np.ndarray, quant: np.ndarray) -> float:
        return float(np.sqrt(np.mean((orig - quant)**2)))

    errors, snr_vals, rmse_vals = [], [], []
    for i in calib_idx[:200]:
        _, _, lf = model.forward_numpy(X_calib[i])

        x_q    = np.clip(np.round(X_calib[i] / s_x), clip_min, clip_max).astype(np.int64)
        z1_q   = x_q @ W1_q.astype(np.int64) + b1_q.astype(np.int64)
        h_raw  = np.array([fp_requant(int(v), MULT1, SHIFT1, clip_min, clip_max) for v in z1_q])
        h_q    = np.maximum(0, h_raw).astype(np.int64)
        z2_q   = h_q @ W2_q.astype(np.int64) + b2_q.astype(np.int64)
        lq_int = np.array([fp_requant(int(v), MULT2, SHIFT2, clip_min, clip_max) for v in z2_q])
        lq_dequant = dequantize_logits(lq_int, s_logits)
        errors.append(int(np.argmax(lf) != int(np.argmax(lq_int))))
        snr_vals.append(calc_snr(lf, lq_dequant))
        rmse_vals.append(calc_rmse(lf, lq_dequant))

    flash_kb = (W1_q.nbytes + W2_q.nbytes + b1_q.nbytes + b2_q.nbytes) / 1024
    print(f'Divergencia FP32 vs INT{bits} : {np.mean(errors)*100:.1f}%')
    print(f'SNR medio                     : {np.nanmean(snr_vals):.2f} dB')
    print(f'RMSE medio                    : {np.mean(rmse_vals):.6f}')
    print(f'Flash estimado (pesos+bias)   : ~{flash_kb:.1f} KB')

    return dict(
        bits=bits,
        W1_q=W1_q, W2_q=W2_q, b1_q=b1_q, b2_q=b2_q,
        s_x=s_x, s_h=s_h, s_logits=s_logits,
        MULT1=MULT1, SHIFT1=SHIFT1,
        MULT2=MULT2, SHIFT2=SHIFT2,
    )

def _c_array(name: str, arr: np.ndarray, c_type: str, per_line: int = 16) -> str:
    flat = [int(v) for v in arr.flatten()]
    rows = ["  " + ", ".join(map(str, flat[i:i + per_line]))
            for i in range(0, len(flat), per_line)]
    return (f"const {c_type} {name}[] PROGMEM = {{\n"
            + ",\n".join(rows) + "\n};\n\n")

def export_arduino_sketch(path: str, model: MLP, q: dict, X_te: np.ndarray, y_te: np.ndarray, n_export: int) -> None:
    bits = q['bits']
    N    = min(n_export, len(X_te))
    cw   = _C_WEIGHT[bits]
    cb   = _C_BIAS[bits]
    ca   = _C_ACCUM[bits]
    clip_max =  2**(bits - 1) - 1
    clip_min = -(2**(bits - 1))

    X_te_q = np.clip(np.round(X_te[:N] / q['s_x']), clip_min, clip_max).astype(_NP_WEIGHT[bits])

    header = f"""\
#include <Arduino.h>
#include <avr/pgmspace.h>
#include <string.h>

#define INPUT_SIZE  {INPUT_SIZE}
#define HIDDEN_SIZE {HIDDEN_SIZE}
#define OUTPUT_SIZE {OUTPUT_SIZE}
#define N_TEST      {N}

#define MULT1  ((int32_t){q['MULT1']}L)
#define SHIFT1 {q['SHIFT1']}
#define MULT2  ((int32_t){q['MULT2']}L)
#define SHIFT2 {q['SHIFT2']}
"""

    data_sec =  _c_array("W1_q",     q['W1_q'],                 cw)
    data_sec += _c_array("W2_q",     q['W2_q'],                 cw)
    data_sec += _c_array("b1_q",     q['b1_q'],                 cb)
    data_sec += _c_array("b2_q",     q['b2_q'],                 cb)
    data_sec += f"// {N} amostras de teste\n"
    data_sec += _c_array("X_test_q", X_te_q,                    cw)
    data_sec += _c_array("y_labels", y_te[:N].astype(np.uint8), "uint8_t")

    int64_helper = ""
    if bits == 32:
        int64_helper = """\
static int64_t read_int64_p(const int64_t *addr) {
  int64_t v;
  memcpy_P(&v, addr, sizeof(v));
  return v;
}

"""

    if bits == 8:
        requant_block = """\
static int8_t requant(int32_t acc, int32_t mult, int shift) {
  int64_t x = (int64_t)acc * mult;
  x >>= 31;
  x >>= shift;
  if (x >  127) x =  127;
  if (x < -128) x = -128;
  return (int8_t)x;
}

"""
    else:
        out_type = 'int16_t' if bits == 16 else 'int32_t'
        lo_clamp = '32767, -32768' if bits == 16 else '2147483647LL, -2147483648LL'
        hi_val, lo_val = lo_clamp.split(', ')
        requant_block = f"""\
static int64_t mulq31(int64_t a, int32_t m) {{
  int64_t a_hi = a >> 31;
  int64_t a_lo = a & 0x7FFFFFFFL;
  return (int64_t)a_hi * m + (((int64_t)a_lo * m) >> 31);
}}

static {out_type} requant(int64_t acc, int32_t mult, int shift) {{
  int64_t x = mulq31(acc, mult);
  x >>= shift;
  if (x >  {hi_val}) x =  {hi_val};
  if (x < {lo_val}) x = {lo_val};
  return ({out_type})x;
}}

"""

    if bits == 8:
        rw1 = "((int32_t)(int8_t) pgm_read_byte (&W1_q[i * HIDDEN_SIZE + j]))"
        rw2 = "((int32_t)(int8_t) pgm_read_byte (&W2_q[j * OUTPUT_SIZE + k]))"
        rb1 = "((int32_t)         pgm_read_dword(&b1_q[j]))"
        rb2 = "((int32_t)         pgm_read_dword(&b2_q[k]))"
        rx  = "((int8_t)          pgm_read_byte (&X_test_q[n * INPUT_SIZE + i]))"
    elif bits == 16:
        rw1 = "((int64_t)(int16_t)pgm_read_word (&W1_q[i * HIDDEN_SIZE + j]))"
        rw2 = "((int64_t)(int16_t)pgm_read_word (&W2_q[j * OUTPUT_SIZE + k]))"
        rb1 = "((int64_t)(int32_t)pgm_read_dword(&b1_q[j]))"
        rb2 = "((int64_t)(int32_t)pgm_read_dword(&b2_q[k]))"
        rx  = "((int16_t)         pgm_read_word (&X_test_q[n * INPUT_SIZE + i]))"
    else:
        rw1 = "((int64_t)(int32_t)pgm_read_dword(&W1_q[i * HIDDEN_SIZE + j]))"
        rw2 = "((int64_t)(int32_t)pgm_read_dword(&W2_q[j * OUTPUT_SIZE + k]))"
        rb1 = "read_int64_p(&b1_q[j])"
        rb2 = "read_int64_p(&b2_q[k])"
        rx  = "((int32_t)         pgm_read_dword(&X_test_q[n * INPUT_SIZE + i]))"

    infer_fn = f"""\
int mlp_predict(const {cw} *x_in) {{
  
  // Camada 1: z1 = ReLU( W1^T * x + b1 )  
  {ca} z1[HIDDEN_SIZE];
  for (int j = 0; j < HIDDEN_SIZE; j++) {{
    {ca} acc = {rb1};
    for (int i = 0; i < INPUT_SIZE; i++)
      acc += ({ca})x_in[i] * {rw1};
    // Requantiza e aplica ReLU
    {cw} v = requant(acc, MULT1, SHIFT1);
    z1[j] = (v > 0) ? ({ca})v : 0;
  }}

  // Camada 2: logits = W2^T * h + b2  
  {ca} logits[OUTPUT_SIZE];
  for (int k = 0; k < OUTPUT_SIZE; k++) {{
    {ca} acc = {rb2};
    for (int j = 0; j < HIDDEN_SIZE; j++)
      acc += z1[j] * {rw2};
    logits[k] = acc; // mantém em int32_t para argmax
  }}
  
  // Argmax
  int best = 0;
  for (int k = 1; k < OUTPUT_SIZE; k++)
    if (logits[k] > logits[best]) best = k;
  return best;
}}

"""

    setup_fn = f"""\
// Definições para o cálculo aproximado
const float VCC = 5.0;
const float I_ACTIVE = 0.020;
const float POWER_W = VCC * I_ACTIVE; 

void setup() {{
  Serial.begin(115200);
  Serial.println("Validacao MLP INT{bits}");

  int correct = 0;
  float total_joules = 0;
  unsigned long total_time_us = 0;
  
  for (int n = 0; n < N_TEST; n++) {{
    {cw} sample[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
      sample[i] = {rx};

    unsigned long t_start = micros();
    int pred = mlp_predict(sample);
    unsigned long t_end = micros();
    
    // Cálculos de métricas
    unsigned long duration_us = t_end - t_start;
    total_time_us += duration_us;
    float duration_sec = duration_us / 1000000.0;
    float energy_joules = POWER_W * duration_sec;
    total_joules += energy_joules;

    // Recupera o rótulo real para comparar
    int label = (int)pgm_read_byte(&y_labels[n]);
    if (pred == label) correct++;

    // Print unificado: Performance + Resultado da Classificação
    Serial.print("Amostra "); Serial.print(n);
    Serial.print(" | Pred: "); Serial.print(pred);
    Serial.print(" | Real: "); Serial.print(label);
    Serial.print(" | Tempo: "); Serial.print(duration_us); Serial.print("us");
    Serial.print(" | Energia: "); Serial.print(energy_joules, 10); Serial.println(" J");
  }}

  float avg_time_us = (float)total_time_us / N_TEST;
  float avg_joules = total_joules / N_TEST;

  Serial.println("------------------------------------------");
  Serial.print("Acuracia Final: "); Serial.print(correct); Serial.print("/"); Serial.println(N_TEST);
  
  Serial.print("Tempo Medio por Amostra: "); 
  Serial.print(avg_time_us); Serial.println(" us");

  Serial.print("Energia Media por Amostra: "); 
  Serial.print(avg_joules, 10); Serial.println(" J");
  
  Serial.print("Tempo Total: "); Serial.print(total_time_us / 1000.0); Serial.println(" ms");
  Serial.print("Energia Total: "); Serial.print(total_joules, 8); Serial.println(" J");
  Serial.println("------------------------------------------");
}}

void loop() {{}}
"""

    sketch = header + data_sec + int64_helper + requant_block + infer_fn + setup_fn
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sketch)

    flash_kb = (q['W1_q'].nbytes + q['W2_q'].nbytes +
                q['b1_q'].nbytes + q['b2_q'].nbytes) / 1024
    print(f'Sketch gerado      : {path}')
    print(f'Flash (pesos+bias) : ~{flash_kb:.1f} KB')
    print(f'Amostras de teste  : {N}')

for bits in (8, 16, 32):
    q = quantize_model(model_final, X_train_full, bits)
    export_arduino_sketch(path=f'mlp_int{bits}.ino', model=model_final, q=q, X_te=X_test, y_te=y_test, n_export=n_export)