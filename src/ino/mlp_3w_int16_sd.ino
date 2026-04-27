
#include <Arduino.h>
#include <SdFat.h>

#define INPUT_SIZE 1000
#define OUTPUT_SIZE 3
#define N_TEST 3953
#define SD_CS_PIN 10

SdFat sd;
SdFile fileX, fileY;

SdFile W_files[3];
SdFile B_files[3];

const float VCC = 5.0;
const float I_MCU = 20.0 / 1000.0;
const float I_SD  = 100.0 / 1000.0;

const float P_INF = VCC * I_MCU;
const float P_SD  = VCC * (I_MCU + I_SD);

#define MULT0 1169775676
#define SHIFT0 19
#define MULT1 1346151473
#define SHIFT1 11
#define MULT2 1786355103
#define SHIFT2 15

int32_t requant(int64_t acc, int32_t mult, int shift){
    int64_t x = (acc * mult) >> 31;
    x >>= shift;
    return (int32_t)x;
}

int mlp_predict(int16_t *input) {

    static int64_t buffer1[32];
    static int64_t buffer2[32];

    int64_t *prev = (int64_t*)input;
    int64_t *curr = buffer1;


    // Layer 0
    for(int j=0;j<32;j++){

        int64_t acc;

        // ---- leitura bias ----
        B_files[0].seek(j * sizeof(int64_t));
        B_files[0].read(&acc, sizeof(int64_t));

        // ---- leitura coluna de W ----
        for(int i=0;i<1000;i++){

            int16_t w;
            uint32_t offset = (i*32 + j) * 2;
            W_files[0].seek(offset);
            W_files[0].read(&w, 2);

            acc += prev[i] * w;
        }


        int32_t v = requant(acc, MULT0, SHIFT0);
        curr[j] = v > 0 ? v : 0;
    }

    prev = curr;
    curr = (curr == buffer1) ? buffer2 : buffer1;

    // Layer 1
    for(int j=0;j<16;j++){

        int64_t acc;

        // ---- leitura bias ----
        B_files[1].seek(j * sizeof(int64_t));
        B_files[1].read(&acc, sizeof(int64_t));

        // ---- leitura coluna de W ----
        for(int i=0;i<32;i++){

            int16_t w;
            uint32_t offset = (i*16 + j) * 2;
            W_files[1].seek(offset);
            W_files[1].read(&w, 2);

            acc += prev[i] * w;
        }


        int32_t v = requant(acc, MULT1, SHIFT1);
        curr[j] = v > 0 ? v : 0;
    }

    prev = curr;
    curr = (curr == buffer1) ? buffer2 : buffer1;

    // Layer 2
    for(int j=0;j<3;j++){

        int64_t acc;

        // ---- leitura bias ----
        B_files[2].seek(j * sizeof(int64_t));
        B_files[2].read(&acc, sizeof(int64_t));

        // ---- leitura coluna de W ----
        for(int i=0;i<16;i++){

            int16_t w;
            uint32_t offset = (i*3 + j) * 2;
            W_files[2].seek(offset);
            W_files[2].read(&w, 2);

            acc += prev[i] * w;
        }

        curr[j] = acc;
    }

    prev = curr;
    curr = (curr == buffer1) ? buffer2 : buffer1;

    int best = 0;
    for(int i=1;i<OUTPUT_SIZE;i++)
        if(prev[i] > prev[best]) best = i;

    return best;
}

void setup(){
    Serial.begin(115200);

    if(!sd.begin(SD_CS_PIN)){
        Serial.println("SD FAIL");
        while(1);
    }

    fileX.open("X_INT16.BIN");
    fileY.open("Y_TEST.BIN");
    W_files[0].open("W1.BIN");
    B_files[0].open("B1.BIN");
    W_files[1].open("W2.BIN");
    B_files[1].open("B2.BIN");
    W_files[2].open("W3.BIN");
    B_files[2].open("B3.BIN");

    int correct=0;

    float E_sample=0, E_weight=0, E_compute=0;

    for(int n=0;n<N_TEST;n++){

        int16_t x[INPUT_SIZE];
        uint8_t y;

        // ---- leitura amostra ----
        unsigned long t0 = micros();
        fileX.read(x, INPUT_SIZE*2);
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
    }

    Serial.println("Done");
    Serial.print("Acc: "); Serial.println(correct);
    Serial.print("E_sample: "); Serial.println(E_sample);
    Serial.print("E_weight: "); Serial.println(E_weight);
    Serial.print("E_compute: "); Serial.println(E_compute);
}

void loop(){}
