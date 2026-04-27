import time

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from codecarbon import EmissionsTracker

from ThreeWToolkit.models.mlp import MLP, MLPConfig
from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.preprocessing import Windowing
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig

RANDOM_SEED = 2025

# PARTE 1: TREINO E INFERENCIA FP32

dataset_path = r'C:\Users\jvt\Downloads\data\3W'
ds_config = ParquetDatasetConfig(path=dataset_path, clean_data=True, seed=RANDOM_SEED, target_class=[1, 2])
ds = ParquetDataset(ds_config)

window_size = 1000

mlp_config = MLPConfig(
    input_size=window_size,
    hidden_sizes=(32, 16),
    output_size=3,
    random_seed=RANDOM_SEED,
    activation_function="relu",
    regularization=None,
)

trainer_config = TrainerConfig(
    optimizer="adam",
    criterion="cross_entropy",
    batch_size=32,
    epochs=20,
    seed=RANDOM_SEED,
    config_model=mlp_config,
    learning_rate=0.001,
    cross_validation=False,
    shuffle_train=True,
)

trainer = ModelTrainer(trainer_config)

windowing_config       = WindowingConfig(window="hann",   window_size=window_size, overlap=0.5, pad_last_window=True)
label_windowing_config = WindowingConfig(window="boxcar", window_size=window_size, overlap=0.5, pad_last_window=True)
windowing       = Windowing(windowing_config)
label_windowing = Windowing(label_windowing_config)

selected_col = "T-TPT"
dfs = []

for event in ds:
    windowed_signal = windowing(event["signal"][selected_col])
    windowed_label  = label_windowing(event["label"])
    windowed_signal.drop(columns=["win"], inplace=True)
    windowed_signal["label"] = (windowed_label.drop(columns=["win"]).mode(axis=1)[0].astype(int))
    dfs.append(windowed_signal)

dfs_final = pd.concat(dfs, ignore_index=True, axis=0)

# 80/20 split
n_total = len(dfs_final)
n_train = int(0.8 * n_total)
rng_split = np.random.default_rng(RANDOM_SEED)
idx = rng_split.permutation(n_total)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_all = dfs_final.iloc[:, :-1].values.astype(np.float32)
y_all = dfs_final["label"].values.astype(np.int64)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

trainer.train(x_train=pd.DataFrame(X_train), y_train=pd.Series(y_train))

assessment_config = ModelAssessmentConfig(metrics=["accuracy"], batch_size=32)

tracker = EmissionsTracker(measure_power_secs=0.1, save_to_file=True)
t0 = time.time()
tracker.start()
results = trainer.assess(
    pd.DataFrame(X_test), pd.Series(y_test.astype(int)),
    assessment_config=assessment_config,
)
emissions_inference = tracker.stop()
duration_inference  = time.time() - t0

energy_kwh    = tracker.final_emissions_data.energy_consumed
energy_wh     = energy_kwh * 1000
energy_joules = energy_wh  * 3600

n_samples_test = len(X_test)

print('=' * 42)
print(f"Test Metrics: {results['metrics']}")
print(f'Tempo total de inferencia (FP32)  : {duration_inference:.3f} s')
print(f'Tempo medio por amostra (FP32)    : {duration_inference / n_samples_test * 1e6:.2f} us')
print(f'Energia consumida (FP32)          : {energy_joules:.4f} J  /  {energy_wh:.6f} Wh')
print(f'Energia media por amostra (FP32)  : {energy_joules / n_samples_test * 1e6:.4f} uJ')
print(f'CO2 estimado                      : {emissions_inference:.6f} kg')
print('=' * 42)

model: MLP = trainer.model
model.eval()

# PARTE 2: QUANTIZACAO INTEIRA (INT8 / INT16 / INT32)

_NP_WEIGHT = {8: np.int8,  16: np.int16, 32: np.int32}
_NP_BIAS   = {8: np.int32, 16: np.int32, 32: np.int64}
_C_WEIGHT  = {8: 'int8_t',  16: 'int16_t', 32: 'int32_t'}
_C_BIAS    = {8: 'int32_t', 16: 'int32_t', 32: 'int64_t'}
_C_ACCUM   = {8: 'int32_t', 16: 'int64_t', 32: 'int64_t'}
_NP_SD     = {8: np.int8,   16: np.int16,  32: np.int32}
_C_BYTES   = {8: 1,         16: 2,         32: 4}

def get_all_linear_layers(mlp: MLP) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    for module in mlp.model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().T  # [in, out]
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
    print(f'\n{"=" * 42}')
    print(f'Quantizando modelo - INT{bits}')
    print(f'{"=" * 42}')

    layer_params = get_all_linear_layers(mlp)
    n_layers = len(layer_params)

    s_W = [calc_scale(W, bits) for W, _ in layer_params]

    calib_idx = np.random.choice(len(X_calib), size=min(2000, len(X_calib)), replace=False)

    x_abs_max: list[float] = []
    h_all: list[list[np.ndarray]] = [[] for _ in range(n_layers - 1)]
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
            if l < n_layers - 1:
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
        bits=bits,
        n_layers=n_layers,
        input_size=mlp_config.input_size,
        hidden_sizes=mlp_config.hidden_sizes,
        output_size=mlp_config.output_size,
        W_q=W_q, b_q=b_q,
        s_x=s_x, s_h=s_h, s_logits=s_logits,
        MULTS=MULTS, SHIFTS=SHIFTS,
    )


def _c_array(name: str, arr: np.ndarray, c_type: str, per_line: int = 16) -> str:
    flat = [int(v) for v in arr.flatten()]
    rows = [
        "  " + ", ".join(map(str, flat[i: i + per_line]))
        for i in range(0, len(flat), per_line)
    ]
    return f"const {c_type} {name}[] PROGMEM = {{\n" + ",\n".join(rows) + "\n};\n\n"

def export_binary_for_sd(
    path_X: str,
    path_y: str,
    X_te: np.ndarray,
    y_te: np.ndarray,
    q: dict,
) -> None:
    bits     = q['bits']
    clip_max =  2 ** (bits - 1) - 1
    clip_min = -(2 ** (bits - 1))

    X_q    = np.clip(np.round(X_te / q['s_x']), clip_min, clip_max).astype(_NP_SD[bits])
    X_q_le = X_q.astype(X_q.dtype.newbyteorder('<'))
    y_le   = y_te.astype(np.uint8)

    with open(path_X, 'wb') as f:
        f.write(X_q_le.tobytes())
    with open(path_y, 'wb') as f:
        f.write(y_le.tobytes())

    n    = len(X_te)
    sz_x = X_q_le.nbytes / 1024
    sz_y = y_le.nbytes / 1024
    print(f'  Binario X (INT{bits:<2}): {path_X}  ({sz_x:.1f} KB, {n} amostras)')
    print(f'  Binario y         : {path_y}  ({sz_y:.3f} KB)')

def export_arduino_sketch_sd(
    path: str,
    q: dict,
    n_test: int,
    sd_cs_pin: int = 10,
    i_mcu_ma: float = 20.0,
    i_sd_ma: float  = 100.0,
    vcc: float      = 5.0,
):
    bits         = q['bits']
    n_layers     = q['n_layers']
    input_size   = q['input_size']
    hidden_sizes = q['hidden_sizes']
    output_size  = q['output_size']

    cw = _C_WEIGHT[bits]
    ca = _C_ACCUM[bits]
    elem_bytes = _C_BYTES[bits]

    layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

    # -------- EXPORT BINÁRIOS --------
    for l in range(n_layers):
        q['W_q'][l].astype(_NP_SD[bits]).tofile(f"W{l+1}.BIN")
        q['b_q'][l].astype(_NP_BIAS[bits]).tofile(f"B{l+1}.BIN")

    print("Pesos exportados para SD!")

    # -------- CÓDIGO ARDUINO --------
    code = f"""
#include <Arduino.h>
#include <SdFat.h>

#define INPUT_SIZE {input_size}
#define OUTPUT_SIZE {output_size}
#define N_TEST {n_test}
#define SD_CS_PIN {sd_cs_pin}

SdFat sd;
SdFile fileX, fileY;

SdFile W_files[{n_layers}];
SdFile B_files[{n_layers}];

const float VCC = {vcc};
const float I_MCU = {i_mcu_ma} / 1000.0;
const float I_SD  = {i_sd_ma} / 1000.0;

const float P_INF = VCC * I_MCU;
const float P_SD  = VCC * (I_MCU + I_SD);

"""

    # Multipliers
    for l in range(n_layers):
        code += f"#define MULT{l} {q['MULTS'][l]}\n"
        code += f"#define SHIFT{l} {q['SHIFTS'][l]}\n"

    # -------- REQUANT --------
    if bits == 8:
        code += """
int8_t requant(int32_t acc, int32_t mult, int shift){
    int64_t x = (int64_t)acc * mult;
    x >>= 31;
    x >>= shift;
    if(x > 127) x = 127;
    if(x < -128) x = -128;
    return (int8_t)x;
}
"""
    else:
        code += """
int32_t requant(int64_t acc, int32_t mult, int shift){
    int64_t x = (acc * mult) >> 31;
    x >>= shift;
    return (int32_t)x;
}
"""

    # -------- INFERÊNCIA STREAMING --------
    code += f"""
int mlp_predict({cw} *input) {{

    static {ca} buffer1[{max(hidden_sizes + (output_size,))}];
    static {ca} buffer2[{max(hidden_sizes + (output_size,))}];

    {ca} *prev = ({ca}*)input;
    {ca} *curr = buffer1;

"""

    for l in range(n_layers):
        in_size  = layer_sizes[l]
        out_size = layer_sizes[l+1]

        code += f"""
    // Layer {l}
    for(int j=0;j<{out_size};j++){{

        {ca} acc;

        // ---- leitura bias ----
        B_files[{l}].seek(j * sizeof({ca}));
        B_files[{l}].read(&acc, sizeof({ca}));

        // ---- leitura coluna de W ----
        for(int i=0;i<{in_size};i++){{

            {cw} w;
            uint32_t offset = (i*{out_size} + j) * {elem_bytes};
            W_files[{l}].seek(offset);
            W_files[{l}].read(&w, {elem_bytes});

            acc += prev[i] * w;
        }}

"""

        if l < n_layers - 1:
            code += f"""
        int32_t v = requant(acc, MULT{l}, SHIFT{l});
        curr[j] = v > 0 ? v : 0;
"""
        else:
            code += "        curr[j] = acc;\n"

        code += "    }\n"

        # swap buffers
        code += """
    prev = curr;
    curr = (curr == buffer1) ? buffer2 : buffer1;
"""

    code += """
    int best = 0;
    for(int i=1;i<OUTPUT_SIZE;i++)
        if(prev[i] > prev[best]) best = i;

    return best;
}
"""

    # -------- SETUP + TELEMETRIA --------
    code += f"""
void setup(){{
    Serial.begin(115200);

    if(!sd.begin(SD_CS_PIN)){{
        Serial.println("SD FAIL");
        while(1);
    }}

    fileX.open("X_INT{bits}.BIN");
    fileY.open("Y_TEST.BIN");
"""

    for l in range(n_layers):
        code += f'    W_files[{l}].open("W{l+1}.BIN");\n'
        code += f'    B_files[{l}].open("B{l+1}.BIN");\n'

    code += f"""
    int correct=0;

    float E_sample=0, E_weight=0, E_compute=0;

    for(int n=0;n<N_TEST;n++){{

        {cw} x[INPUT_SIZE];
        uint8_t y;

        // ---- leitura amostra ----
        unsigned long t0 = micros();
        fileX.read(x, INPUT_SIZE*{elem_bytes});
        fileY.read(&y,1);
        unsigned long t1 = micros();

        float dt_sample = (t1-t0)*1e-6;
        E_sample += P_SD * dt_sample;

        // ---- inferência ----
        unsigned long t2 = micros();
        int pred = mlp_predict(x);
        unsigned long t3 = micros();

        float dt_compute = (t3-t2)*1e-6;
        E_compute += P_INF * dt_compute;

        // estimativa leitura pesos (aprox)
        float dt_weight = dt_compute * 0.6; // heurística
        E_weight += P_SD * dt_weight;

        if(pred==y) correct++;
    }}

    Serial.println("Done");
    Serial.print("Acc: "); Serial.println(correct);
    Serial.print("E_sample: "); Serial.println(E_sample);
    Serial.print("E_weight: "); Serial.println(E_weight);
    Serial.print("E_compute: "); Serial.println(E_compute);
}}

void loop(){{}}
"""

    with open(path, "w") as f:
        f.write(code)

    print(f"Sketch SD FULL gerado: {path}")

# --- PARTE 3: EXPORTACAO SD (conjunto de teste completo) ---

print('\n' + '=' * 42)
print('PARTE 3 - Exportacao SD (conjunto completo)')
print('=' * 42)

n_sd = len(X_test)
print(f'Total de amostras de teste para SD: {n_sd}')
print(f'Distribuicao de classes: '
      f'{ {int(c): int((y_test == c).sum()) for c in np.unique(y_test)} }')

y_bin_written = False

for bits in (8, 16, 32):
    print(f'\n-- INT{bits} --')
    q_sd = quantize_model(model, X_train, bits)

    path_X = f'X_INT{bits}.BIN'
    path_y = 'Y_TEST.BIN'

    if not y_bin_written:
        export_binary_for_sd(path_X, path_y, X_test, y_test, q_sd)
        y_bin_written = True
    else:
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -(2 ** (bits - 1))
        X_q_le = np.clip(
            np.round(X_test / q_sd['s_x']), clip_min, clip_max
        ).astype(_NP_SD[bits])
        X_q_le = X_q_le.astype(X_q_le.dtype.newbyteorder('<'))
        with open(path_X, 'wb') as f:
            f.write(X_q_le.tobytes())
        print(f'  Binario X (INT{bits:<2}): {path_X}  ({X_q_le.nbytes / 1024:.1f} KB)')

    export_arduino_sketch_sd(path=f'mlp_3w_int{bits}_sd.ino', q=q_sd, n_test=n_sd)

print('\n-- Arquivos para copiar na raiz do SD card (FAT32) --')
for bits in (8, 16, 32):
    print(f'    X_INT{bits}.BIN')
print('    Y_TEST.BIN')
print('\n-- Sketches Arduino gerados --')
for bits in (8, 16, 32):
    print(f'    mlp_3w_int{bits}_sd.ino')
print()
print('NOTA: Ajuste SD_CS_PIN no sketch conforme sua placa.')
print('      Arduino Mega: CS=53 (SPI hardware) ou 10 (SPI software).')
print('      Instale a biblioteca SdFat (Bill Greiman) via Library Manager.')