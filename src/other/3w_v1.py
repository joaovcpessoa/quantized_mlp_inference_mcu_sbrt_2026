import time
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

import torch
import torch.nn as nn

from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.preprocessing import Windowing
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig

dataset_path = r'C:\Users\jvt\Downloads\data\3W'
random_seed = 2026
window_size = 1000
ds_config = ParquetDatasetConfig(path=dataset_path, clean_data=True, seed=random_seed, target_class=[1, 2])
ds = ParquetDataset(ds_config)

mlp_config = MLPConfig(
    input_size=window_size, hidden_sizes=(32, 16), output_size=3, 
    random_seed=random_seed, activation_function='relu', regularization=None
)

trainer_config = TrainerConfig(
    optimizer='adam', criterion='cross_entropy', batch_size=32, epochs=20, seed=random_seed, 
    config_model=mlp_config, learning_rate=0.001, cross_validation=False, shuffle_train=True
)

trainer = ModelTrainer(trainer_config)

windowing_config       = WindowingConfig(window="hann",   window_size=window_size, overlap=0.5, pad_last_window=True)
label_windowing_config = WindowingConfig(window="boxcar", window_size=window_size, overlap=0.5, pad_last_window=True)
windowing       = Windowing(windowing_config)
label_windowing = Windowing(label_windowing_config)
selected_col = 'T-TPT'
dfs = []

for event in ds:
    windowed_signal = windowing(event['signal'][selected_col])
    windowed_label  = label_windowing(event['label'])
    windowed_signal.drop(columns=['win'], inplace=True)
    windowed_signal['label'] = (windowed_label.drop(columns=["win"]).mode(axis=1)[0].astype(int))
    dfs.append(windowed_signal)

dfs_final = pd.concat(dfs, ignore_index=True, axis=0)

n_total = len(dfs_final)
n_train = int(0.8 * n_total)
rng_split = np.random.default_rng(random_seed)
idx = rng_split.permutation(n_total)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_all = dfs_final.iloc[:, :-1].values.astype(np.float32)
y_all = dfs_final['label'].values.astype(np.int64)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

trainer.train(x_train=pd.DataFrame(X_train), y_train=pd.Series(y_train))
assessment_config = ModelAssessmentConfig(metrics=['accuracy'], batch_size=32)
tracker = EmissionsTracker(measure_power_secs=0.1, save_to_file=True)
t0 = time.time()
tracker.start()
results = trainer.assess(pd.DataFrame(X_test), pd.Series(y_test.astype(int)), assessment_config=assessment_config)
tracker.stop()
emissions_inference = tracker.final_emissions_data.emissions
duration_inference  = time.time() - t0
energy_kwh    = tracker.final_emissions_data.energy_consumed
energy_wh     = energy_kwh * 1000
energy_joules = energy_wh  * 3600
n_samples_test = len(X_test)

print(f'Test Metrics: {results['metrics']}')
print(f'Tempo total de inferencia (FP32)  : {duration_inference:.3f} s')
print(f'Tempo medio por amostra (FP32)    : {duration_inference / n_samples_test * 1e6:.2f} µs')
print(f'Energia consumida (FP32)          : {energy_joules:.4f} J  /  {energy_wh:.6f} Wh')
print(f'Energia media por amostra (FP32)  : {energy_joules / n_samples_test * 1e6:.4f} µJ')
print(f'CO2 estimado                      : {emissions_inference:.6f} kg')
print('=' * 42)

model: MLP = trainer.model
model.eval()

_NP_WEIGHT = {8: np.int8,  16: np.int16, 32: np.int32}
_NP_BIAS   = {8: np.int32, 16: np.int32, 32: np.int64}
_C_WEIGHT  = {8: 'int8_t',  16: 'int16_t', 32: 'int32_t'}
_C_BIAS    = {8: 'int32_t', 16: 'int32_t', 32: 'int64_t'}
_C_ACCUM   = {8: 'int32_t', 16: 'int64_t', 32: 'int64_t'}

def get_all_linear_layers(mlp: MLP) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    for module in mlp.model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().T   # [in, out]
            b = module.bias.detach().cpu().numpy()       # [out]
            layers.append((W, b))
    return layers

def forward_numpy(mlp: MLP, x_np: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        hidden_acts: list[np.ndarray] = []
        for module in mlp.model:
            x = module(x)
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                hidden_acts.append(x.squeeze().cpu().numpy().copy())
        logits = x.squeeze().cpu().numpy().copy()
    return hidden_acts, logits

def calc_scale(arr: np.ndarray, bits: int) -> float:
    return float(np.max(np.abs(arr))) / (2 ** (bits - 1) - 1) + 1e-8

def quantize_multiplier(m: float) -> tuple[int, int]:
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

def dequantize(q_int: np.ndarray, scale: float) -> np.ndarray:
    return q_int.astype(np.float64) * scale

def quantize_model(mlp: MLP, X_calib: np.ndarray, bits: int) -> dict:
    print(f'{"=" * 42}')
    print(f'Quantizando modelo - INT{bits}')
    print(f'{"=" * 42}')

    layer_params = get_all_linear_layers(mlp)
    n_layers = len(layer_params)
    s_W = [calc_scale(W, bits) for W, _ in layer_params]
    calib_idx = np.random.choice(len(X_calib), size=min(2000, len(X_calib)), replace=False)

    x_abs_max: list[float] = []
    h_all: list[list[np.ndarray]] = [[] for _ in range(n_layers - 1)]  # hidden only
    logits_all: list[np.ndarray] = []

    mlp.eval()
    for i in calib_idx:
        x_abs_max.append(float(np.max(np.abs(X_calib[i]))))
        hidden_acts, logits = forward_numpy(mlp, X_calib[i])
        for l, h in enumerate(hidden_acts):
            h_all[l].append(h)
        logits_all.append(logits)

    clip_max = 2 ** (bits - 1) - 1
    clip_min = -(2 ** (bits - 1))

    s_x      = float(np.max(x_abs_max)) / clip_max + 1e-8
    s_h      = [calc_scale(np.array(h_list), bits) for h_list in h_all]
    s_logits = calc_scale(np.array(logits_all), bits)

    s_in  = [s_x] + s_h
    s_out = s_h + [s_logits]

    W_q, b_q, MULTS, SHIFTS = [], [], [], []
    for l, (W, b) in enumerate(layer_params):
        W_q.append(np.clip(np.round(W / s_W[l]), clip_min, clip_max).astype(_NP_WEIGHT[bits]))
        b_q.append(np.round(b / (s_in[l] * s_W[l])).astype(_NP_BIAS[bits]))
        M = (s_in[l] * s_W[l]) / s_out[l]
        mult, shift = quantize_multiplier(M)
        MULTS.append(mult)
        SHIFTS.append(shift)

    def calc_snr(orig: np.ndarray, quant: np.ndarray) -> float:
        noise = orig - quant
        pn = np.mean(noise ** 2)
        return 10.0 * np.log10(np.mean(orig ** 2) / pn) if pn > 0 else np.inf

    errors, snr_vals, rmse_vals = [], [], []
    for i in calib_idx[:200]:
        _, lf = forward_numpy(mlp, X_calib[i])

        h = np.clip(np.round(X_calib[i] / s_x), clip_min, clip_max).astype(np.int64)
        for l in range(n_layers):
            z = h @ W_q[l].astype(np.int64) + b_q[l].astype(np.int64)
            h_req = np.array(
                [fp_requant(int(v), MULTS[l], SHIFTS[l], clip_min, clip_max) for v in z]
            )
            if l < n_layers - 1:          # ReLU on hidden layers only
                h = np.maximum(0, h_req).astype(np.int64)
            else:
                h = h_req

        lq_dequant = dequantize(h, s_logits)
        errors.append(int(np.argmax(lf) != int(np.argmax(h))))
        snr_vals.append(calc_snr(lf, lq_dequant))
        rmse_vals.append(float(np.sqrt(np.mean((lf - lq_dequant) ** 2))))

    flash_kb = sum(W.nbytes + b.nbytes for W, b in zip(W_q, b_q)) / 1024
    print(f'Divergencia FP32 vs INT{bits} : {np.mean(errors) * 100:.1f}%')
    print(f'SNR medio                     : {np.nanmean(snr_vals):.2f} dB')
    print(f'RMSE medio                    : {np.mean(rmse_vals):.6f}')
    print(f'Flash estimado (pesos+bias)   : ~{flash_kb:.1f} KB')

    return dict(
        bits=bits, n_layers=n_layers, input_size=mlp_config.input_size, hidden_sizes=mlp_config.hidden_sizes,
        output_size=mlp_config.output_size, W_q=W_q, b_q=b_q, s_x=s_x, s_h=s_h, s_logits=s_logits, MULTS=MULTS, SHIFTS=SHIFTS
    )

def _c_array(name: str, arr: np.ndarray, c_type: str, per_line: int = 16) -> str:
    flat = [int(v) for v in arr.flatten()]
    rows = [
        "  " + ", ".join(map(str, flat[i: i + per_line]))
        for i in range(0, len(flat), per_line)
    ]
    return f"const {c_type} {name}[] PROGMEM = {{\n" + ",\n".join(rows) + "\n};\n\n"

def export_arduino_sketch(path: str, model: MLP, q: dict, X_te: np.ndarray, y_te: np.ndarray, n_export: int) -> None:
    bits       = q['bits']
    n_layers   = q['n_layers']
    input_size = q['input_size']
    hidden_sizes = q['hidden_sizes']   # e.g. (32, 16)
    output_size  = q['output_size']
    N            = min(n_export, len(X_te))
    cw, cb, ca   = _C_WEIGHT[bits], _C_BIAS[bits], _C_ACCUM[bits]
    clip_max     =  2 ** (bits - 1) - 1
    clip_min     = -(2 ** (bits - 1))

    # Layer sizes: [input, h1, h2, ..., output]
    layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

    X_te_q = np.clip(
        np.round(X_te[:N] / q['s_x']), clip_min, clip_max
    ).astype(_NP_WEIGHT[bits])

    # ── #define block ────────────────────────────────────────────────
    defines = f"""\
#include <Arduino.h>
#include <avr/pgmspace.h>
#include <string.h>

#define INPUT_SIZE  {input_size}
"""
    for l, h in enumerate(hidden_sizes):
        defines += f"#define HIDDEN{l + 1}     {h}\n"
    defines += f"#define OUTPUT_SIZE {output_size}\n"
    defines += f"#define N_TEST      {N}\n\n"

    for l in range(n_layers):
        defines += f"#define MULT{l + 1}  ((int32_t){q['MULTS'][l]}L)\n"
        defines += f"#define SHIFT{l + 1} {q['SHIFTS'][l]}\n"
    defines += "\n"

    # ── PROGMEM weight/bias arrays ───────────────────────────────────
    data_sec = ""
    for l in range(n_layers):
        data_sec += _c_array(f"W{l + 1}_q", q['W_q'][l], cw)
        data_sec += _c_array(f"b{l + 1}_q", q['b_q'][l], cb)
    data_sec += f"// {N} amostras de teste\n"
    data_sec += _c_array("X_test_q", X_te_q, cw)
    data_sec += _c_array("y_labels", y_te[:N].astype(np.uint8), "uint8_t")

    # ── int64 helper for INT32 mode ──────────────────────────────────
    int64_helper = ""
    if bits == 32:
        int64_helper = """\
static int64_t read_int64_p(const int64_t *addr) {
  int64_t v;
  memcpy_P(&v, addr, sizeof(v));
  return v;
}

"""

    # ── requant function ─────────────────────────────────────────────
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
        hi_val   = '32767'        if bits == 16 else '2147483647LL'
        lo_val   = '-32768'       if bits == 16 else '-2147483648LL'
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

    # ── PROGMEM read helpers (type-dependent) ────────────────────────
    def pgm_read_weight(arr_name: str, flat_idx: str) -> str:
        if bits == 8:
            return f"((int32_t)(int8_t) pgm_read_byte (&{arr_name}[{flat_idx}]))"
        elif bits == 16:
            return f"((int64_t)(int16_t)pgm_read_word (&{arr_name}[{flat_idx}]))"
        else:
            return f"((int64_t)(int32_t)pgm_read_dword(&{arr_name}[{flat_idx}]))"

    def pgm_read_bias(arr_name: str, idx: str) -> str:
        if bits == 8:
            return f"((int32_t)         pgm_read_dword(&{arr_name}[{idx}]))"
        elif bits == 16:
            return f"((int64_t)(int32_t)pgm_read_dword(&{arr_name}[{idx}]))"
        else:
            return f"read_int64_p(&{arr_name}[{idx}])"

    def pgm_read_input(arr_name: str, idx: str) -> str:
        if bits == 8:
            return f"((int8_t)          pgm_read_byte (&{arr_name}[{idx}]))"
        elif bits == 16:
            return f"((int16_t)         pgm_read_word (&{arr_name}[{idx}]))"
        else:
            return f"((int32_t)         pgm_read_dword(&{arr_name}[{idx}]))"

    # ── mlp_predict(): unrolled per-layer loops ──────────────────────
    # Variable names for each layer's output buffer: z1[], z2[], logits[]
    infer_body = f"int mlp_predict(const {cw} *x_in) {{\n\n"

    prev_buf  = "x_in"
    prev_size = input_size

    for l in range(n_layers):
        cur_size  = layer_sizes[l + 1]
        is_output = (l == n_layers - 1)
        W_name    = f"W{l + 1}_q"
        b_name    = f"b{l + 1}_q"
        mult_name = f"MULT{l + 1}"
        shift_name= f"SHIFT{l + 1}"

        if is_output:
            buf_name  = "logits"
            buf_decl  = f"  {ca} {buf_name}[OUTPUT_SIZE];\n"
            size_macro= "OUTPUT_SIZE"
        else:
            buf_name  = f"z{l + 1}"
            h_macro   = f"HIDDEN{l + 1}"
            buf_decl  = f"  {ca} {buf_name}[{h_macro}];\n"
            size_macro= h_macro

        if l == 0:
            layer_comment = f"  // Layer {l+1}: z = W{l+1}^T * x_in + b{l+1}"
        else:
            layer_comment = f"  // Layer {l+1}: z = W{l+1}^T * {prev_buf} + b{l+1}"

        if not is_output:
            layer_comment += f", {buf_name} = ReLU(requant(z))"
        else:
            layer_comment += "  (raw accumulator – argmax invariant to scale)"

        rw  = pgm_read_weight(W_name, f"i * {size_macro} + j")
        rb  = pgm_read_bias(b_name, "j")

        if l == 0:
            # x_in is a C array passed in, read directly
            rx = f"({ca})x_in[i]"
        else:
            rx = f"({ca}){prev_buf}[i]"

        infer_body += layer_comment + "\n"
        infer_body += buf_decl
        infer_body += f"  for (int j = 0; j < {size_macro}; j++) {{\n"
        infer_body += f"    {ca} acc = {rb};\n"
        infer_body += f"    for (int i = 0; i < {f'INPUT_SIZE' if l == 0 else f'HIDDEN{l}'}; i++)\n"
        infer_body += f"      acc += {rx} * {rw};\n"

        if not is_output:
            infer_body += f"    {cw} v = requant(acc, {mult_name}, {shift_name});\n"
            infer_body += f"    {buf_name}[j] = (v > 0) ? ({ca})v : 0;  // ReLU\n"
        else:
            infer_body += f"    {buf_name}[j] = acc;  // mantém acumulador para argmax\n"

        infer_body += "  }\n\n"
        prev_buf  = buf_name
        prev_size = cur_size

    infer_body += """\
  // Argmax
  int best = 0;
  for (int k = 1; k < OUTPUT_SIZE; k++)
    if (logits[k] > logits[best]) best = k;
  return best;
}

"""

    # ── setup() / loop() ─────────────────────────────────────────────
    rx_test = pgm_read_input("X_test_q", "n * INPUT_SIZE + i")
    setup_fn = f"""\
const float VCC      = 5.0;
const float I_ACTIVE = 0.020;
const float POWER_W  = VCC * I_ACTIVE;

void setup() {{
  Serial.begin(115200);
  Serial.println("Validacao MLP INT{bits}");

  int correct = 0;
  float total_joules  = 0;
  unsigned long total_time_us = 0;

  for (int n = 0; n < N_TEST; n++) {{
    {cw} sample[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
      sample[i] = {rx_test};

    unsigned long t_start = micros();
    int pred = mlp_predict(sample);
    unsigned long t_end   = micros();

    unsigned long duration_us = t_end - t_start;
    total_time_us += duration_us;
    float duration_sec  = duration_us / 1000000.0;
    float energy_joules = POWER_W * duration_sec;
    total_joules += energy_joules;

    int label = (int)pgm_read_byte(&y_labels[n]);
    if (pred == label) correct++;

    Serial.print("Amostra "); Serial.print(n);
    Serial.print(" | Pred: ");    Serial.print(pred);
    Serial.print(" | Real: ");    Serial.print(label);
    Serial.print(" | Tempo: ");   Serial.print(duration_us);   Serial.print("us");
    Serial.print(" | Energia: "); Serial.print(energy_joules, 10); Serial.println(" J");
  }}

  float avg_time_us = (float)total_time_us / N_TEST;
  float avg_joules  = total_joules / N_TEST;

  Serial.println("------------------------------------------");
  Serial.print("Acuracia Final: "); Serial.print(correct); Serial.print("/"); Serial.println(N_TEST);
  Serial.print("Tempo Medio por Amostra: "); Serial.print(avg_time_us); Serial.println(" us");
  Serial.print("Energia Media por Amostra: "); Serial.print(avg_joules, 10); Serial.println(" J");
  Serial.print("Tempo Total: "); Serial.print(total_time_us / 1000.0); Serial.println(" ms");
  Serial.print("Energia Total: "); Serial.print(total_joules, 8); Serial.println(" J");
  Serial.println("------------------------------------------");
}}

void loop() {{}}
"""
    sketch = defines + data_sec + int64_helper + requant_block + infer_body + setup_fn
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sketch)

    flash_kb = sum(W.nbytes + b.nbytes for W, b in zip(q['W_q'], q['b_q'])) / 1024
    print(f'Sketch gerado      : {path}')
    print(f'Flash (pesos+bias) : ~{flash_kb:.1f} KB')
    print(f'Amostras de teste  : {N}')

def select_balanced_export( X: np.ndarray, y: np.ndarray, n: int, seed: int = random_seed) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_out, y_out) with ≈ n//n_classes samples per class.

    Any remainder slots are filled from the majority class so the
    total is exactly min(n, len(X)).
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)
    n_cls   = len(classes)
    base    = n // n_cls
    extra   = n % n_cls

    chosen: list[int] = []
    for i, c in enumerate(classes):
        idx_c   = np.where(y == c)[0]
        n_pick  = base + (1 if i < extra else 0)
        n_pick  = min(n_pick, len(idx_c))
        chosen.extend(rng.choice(idx_c, size=n_pick, replace=False).tolist())

    chosen = np.array(chosen)
    rng.shuffle(chosen)
    return X[chosen], y[chosen]

n_export = 10
X_export, y_export = select_balanced_export(X_test, y_test, n_export)
print(f'\nAmostras exportadas - distribuição de classes: '
      f'{ {int(c): int((y_export == c).sum()) for c in np.unique(y_export)} }')

for bits in (8, 16, 32):
    q = quantize_model(model, X_train, bits)
    export_arduino_sketch(path=f'mlp_3w_int{bits}.ino', model=model, q=q, X_te=X_export, y_te=y_export, n_export=n_export)