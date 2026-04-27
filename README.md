<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']]
    },
    messageStyle: "none"
  });
</script>

# Informações gerais sobre o código

### Parte 1 - Treino e Inferência em ponto flutuante (FP32)

Esse projeto implementa e treina uma Rede Neural MLP (Multilayer Perceptron) simples para realizar tarefas de classificação para o conjunto de dados MNIST , composto por dígitos manuscristo e para o conjunto de dados 3W com informações sobre sensoreamento de poços de petróleo.

Então eu baixo e normalizo os pixels, ajustando os valores para que o cálculo matemático seja mais eficiente.

Além de treinar o modelo, o objetivo é validação estatística, usando Monte Carlo e sustentabilidade, medindo o consumo de energia e pegada de carbono.

Primeiro eu defino algumas variáveis básicas necessárias, depois defino uma rede neural clássica com três camadas:

- Entrada: 784 neurônios (uma imagem de 28x28 pixels "achatada")
- Oculta: 10 neurônios com função de ativação ReLU
- Saída: 10 neurônios (um para cada dígito de 0 a 9)

Em vez de treinar o modelo uma única vez, o código usa uma técnica inspirada em Monte Carlo para garantir que os resultados sejam confiáveis:

- Repete o processo 10 vezes (n_mc = 10)
- Em cada repetição, embaralha os dados e divide em:
    - 80% Treino: Para o modelo aprender
    - 20% Validação: Para testar se o modelo aprendeu corretamente
- Ao final, calcula a média e o desvio padrão da acurácia

Isso serve para saber se o modelo é estável ou se deu sorte em algum treinamento específico.

### Medição energética

Um diferencial desse código é o uso da biblioteca CodeCarbon. Enquanto o modelo treina, o EmissionsTracker fica "espiando" o hardware para calcular o tempo total de execução, a energia consumida (em Joules e Wh) e a estimativa de CO2 emitido para a atmosfera durante esse processamento.

Em cenários de Deep Learning para dispositivos móveis ou sistemas embarcados, medir a energia da inferência é quase tão importante quanto medir a acurácia. Uma coisa que eu poderia fazer é calcular apenas o custo do treino e separadamente o custo da inferência, assim eu treino uma relação de Custo de Operação vs. Custo de Criação.

No teste, estamos processando as imagens uma a uma (ou em pequenos lotes). Medir a energia aqui ajuda a entender a quanto custa para o processador classificar um único dígito. Se o modelo gasta muita energia para uma simples inferência de 10 mil imagens, ele pode ser inviável para rodar em um sensor que funciona a bateria, por exemplo.

Às vezes, um modelo "A" tem 99% de acurácia, mas gasta o dobro de energia que um modelo "B" com 98.5%. Ao medir a energia no teste, ganhamos dados para decidir: "Vale a pena gastar 50% mais bateria para ganhar 0.5% de precisão?"

#### Pergunta essencial: Como as equações são calculadas?

Não estamos aqui para usar mais uma "caixa preta". A biblioteca não "chuta" esses valores, ela utiliza uma hierarquia de medição baseada no hardware disponível. 

É possível consultar os detalhes técnicos na documentação oficial ou no [repositório do GitHub](https://github.com/mlco2/codecarbon)  do projeto, mas aqui está o resumo das fórmulas principais:

A energia total ($E$) é a soma do consumo dos três componentes principais:

$$E_{total} = E_{cpu} + E_{gpu} + E_{ram}$$

Para cada componente, a energia consumida em um intervalo de tempo é calculada como:

$$E (kWh) = \frac{P (Watts) \times t (horas)}{1000}$$

As equações variam conforme o componente.

- GPU: Usa ferramentas como o `nvidia-smi` para ler diretamente o consumo de energia em tempo real reportado pela placa
- CPU: Intel: Usa a interface RAPL (Running Average Power Limit), que fornece leituras de energia extremamente precisas direto do processador
    - AMD: Usa o driver amd_energy (no Linux).
    - Fallback (Se não houver acesso ao hardware): Ele usa o TDP (Thermal Design Power) do modelo da CPU multiplicado pela carga de uso (cpu_load) atual.
- RAM: Como a maioria dos sistemas não reporta o consumo da RAM individualmente, aplica uma estimativa padrão (geralmente 0.375 W por GB de RAM instalada).
    
Para converter Energia em Emissões, a equação é:

$$CO_{2}e = E_{total} \times CI$$

Onde $CI$ é a Intensidade de Carbono da rede elétrica local. O CodeCarbon detecta sua localização via IP e busca em uma base de dados qual é a mistura energética do seu país (ex: se é baseada em hidrelétricas, como no Brasil, o $CI$ é baixo; se for carvão, é alto).

Se quisermos fazer uma investigação mais profunda para futuras utilizações, dá para ver as equações exatas no código-fonte, basta procurar pelos módulos cpu.py, gpu.py e emissions.py dentro da pasta core no repositório oficial.

#### Complicações

`[codecarbon WARNING] No CPU tracking mode found. Falling back on CPU load mode.`

Esse log é muito interessante porque revela exatamente como a biblioteca se comporta quando encontra "obstáculos" de permissão no Windows e como ele monitora o hardware moderno.

Como estou no Ruindows, a lib tentou acessar as interfaces de energia da AMD e não conseguiu. Por segurança, ela não tem acesso direto aos sensores de Watts do meu Ryzen 7 5700X3D.

Então a lib muda para o `cpu_load`, ou seja, ela pega o TDP do processador (105W para o 5700X3D) e multiplica pela porcentagem de uso da CPU naquele momento. Se o seu código está usando 10% da CPU, ele estima que você está consumindo ~10.5W.

Outra dúvida é que mesmo que o meu código PyTorch esteja rodando cálculos apenas na CPU,  a lib monitora o sistema como um todo, não apenas o processo do Python. Minha RTX 3060 está ligada, enviando imagem para o monitor e mantendo a interface do Windows rodando. Então ele chegou a calcular o consumo medido de apenas 3.88W. Isso é o gasto de energia da placa apenas por estar ligada (estado de repouso). A lib assume que, se você iniciou um experimento de IA, ele deve reportar toda a energia consumida pelo hardware de computação disponível, mesmo que a carga de trabalho específica não o esteja utilizando intensamente.

Para a RAM, ele identificou que tenho 32GB. Como não há um sensor padrão de energia para pentes de memória no Windows, ele usou o modelo estatístico:
`RAM power estimation model`, aplicou uma constante (normalmente 0.375W por GB) para chegar nos 20.0W que aparecem no log.

Então se quisermos calcular isso com maior fidelidade, é interessante testarmos em um sistema Linux, caso contrário, teremos que subtrair esses valores para chegar no mais próximo do real.

Usar um container Docker não adianta. Você só adiciona mais abstração e não tem acesso aos sensores.

### Microcontrolador

Aqui para calcular a energia podemos usar os valores nominais. Qual o motivo? Sendo direto, não existe uma biblioteca C pura que consiga medir o consumo real de energia apenas via software. O Wokwi não possui sensores de corrente, como por exemplo INA219, que poderíamos utilizar para pegar os dados reais. O microcontrolador não tem um sensor de corrente interno. Ele sabe a que velocidade está operando, mas não sabe quanta eletricidade está puxando da fonte.

O que eu fiz foi, através da estimativa por ciclos de clock, como o Wokwi simula o tempo do processador com precisão, podemos usar o consumo médio de corrente do ATmega328P para calcular a energia. Um Arduino Uno consome cerca de 20mA operando a 5V. 

$$P = V \times I \implies 5V \times 0.02A = 0.1W$$

Multiplicamos essa potência fixa pelo tempo exato que o `mlp_predict` leva para rodar e *voilà*.

Abaixo vou deixar um log do código.

Resultado do python:

```txt
Dispositivo: cpu
Train: (60000, 784) | Test: (10000, 784)
Monte Carlo - 10 runs
Run  1/10 | Val Acc = 0.9165
Run  2/10 | Val Acc = 0.9122
Run  3/10 | Val Acc = 0.9157
Run  4/10 | Val Acc = 0.9155
Run  5/10 | Val Acc = 0.9126
Run  6/10 | Val Acc = 0.9120
Run  7/10 | Val Acc = 0.9188
Run  8/10 | Val Acc = 0.9177
Run  9/10 | Val Acc = 0.9123
Run 10/10 | Val Acc = 0.8999
==========================================
Precisao media : 0.9133
Desvio std     : 0.0050
Min / Max      : 0.8999 / 0.9188
Tempo total    : 45.8s
[FINAL] Epoch   2 | Loss=0.3619 | Acc=0.8975
[FINAL] Epoch   4 | Loss=0.2977 | Acc=0.9152
[FINAL] Epoch   6 | Loss=0.2768 | Acc=0.9219
[FINAL] Epoch   8 | Loss=0.2660 | Acc=0.9242
[FINAL] Epoch  10 | Loss=0.2562 | Acc=0.9281
==========================================
Precisao final    : 0.9249
Tempo total       : 1.5s
Energia consumida : 37.60 J  /  0.0104 Wh
CO2 estimado      : 0.000001 kg

==========================================
Quantizando modelo - INT8
==========================================
  Divergencia FP32 vs INT8 : 1.0%
  SNR medio                     : 30.91 dB
  RMSE medio                    : 0.153879
  Flash estimado (pesos+bias)   : ~6.3 KB
  Sketch gerado      : mlp_int8.ino
  Flash (pesos+bias) : ~6.3 KB
  Amostras de teste  : 20

==========================================
Quantizando modelo - INT16
==========================================
  Divergencia FP32 vs INT16 : 0.0%
  SNR medio                     : 79.38 dB
  RMSE medio                    : 0.000574
  Flash estimado (pesos+bias)   : ~12.5 KB
  Sketch gerado      : mlp_int16.ino
  Flash (pesos+bias) : ~12.5 KB
  Amostras de teste  : 20

==========================================
Quantizando modelo - INT32
==========================================
  Divergencia FP32 vs INT32 : 0.0%
  SNR medio                     : 133.19 dB
  RMSE medio                    : 0.000001
  Flash estimado (pesos+bias)   : ~25.0 KB
  Sketch gerado      : mlp_int32.ino
  Flash (pesos+bias) : ~25.0 KB
  Amostras de teste  : 20
```

Resultado para o Arduino Uno:

```txt
Validacao MLP INT8
Amostra 0 | Pred: 7 | Real: 7 | Tempo: 39708us | Energia: 0.0039707994 J
Amostra 1 | Pred: 2 | Real: 2 | Tempo: 39936us | Energia: 0.0039935998 J
Amostra 2 | Pred: 1 | Real: 1 | Tempo: 39988us | Energia: 0.0039987995 J
Amostra 3 | Pred: 0 | Real: 0 | Tempo: 39912us | Energia: 0.0039911996 J
Amostra 4 | Pred: 4 | Real: 4 | Tempo: 39968us | Energia: 0.0039967999 J
Amostra 5 | Pred: 1 | Real: 1 | Tempo: 39980us | Energia: 0.0039979996 J
Amostra 6 | Pred: 4 | Real: 4 | Tempo: 39952us | Energia: 0.0039951993 J
Amostra 7 | Pred: 9 | Real: 9 | Tempo: 39960us | Energia: 0.0039959998 J
Amostra 8 | Pred: 6 | Real: 5 | Tempo: 39932us | Energia: 0.0039931998 J
Amostra 9 | Pred: 9 | Real: 9 | Tempo: 39928us | Energia: 0.0039927995 J
------------------------------------------
Acuracia Final: 9/10
Tempo Medio por Amostra: 39926.40 us
Energia Media por Amostra: 0.0039926395 J
Tempo Total: 399.26 ms
Energia Total: 0.03992639 J
------------------------------------------

Validacao MLP INT32
Amostra 0 | Pred: 7 | Real: 7 | Tempo: 191404us | Energia: 0.0191403980 J
Amostra 1 | Pred: 2 | Real: 2 | Tempo: 191156us | Energia: 0.0191155986 J
Amostra 2 | Pred: 1 | Real: 1 | Tempo: 192312us | Energia: 0.0192311992 J
Amostra 3 | Pred: 0 | Real: 0 | Tempo: 190676us | Energia: 0.0190675983 J
Amostra 4 | Pred: 4 | Real: 4 | Tempo: 191672us | Energia: 0.0191671981 J
Amostra 5 | Pred: 1 | Real: 1 | Tempo: 192120us | Energia: 0.0192119979 J
Amostra 6 | Pred: 4 | Real: 4 | Tempo: 191512us | Energia: 0.0191511993 J
Amostra 7 | Pred: 9 | Real: 9 | Tempo: 191608us | Energia: 0.0191607971 J
Amostra 8 | Pred: 6 | Real: 5 | Tempo: 190992us | Energia: 0.0190991983 J
Amostra 9 | Pred: 9 | Real: 9 | Tempo: 190944us | Energia: 0.0190943984 J
------------------------------------------
Acuracia Final: 9/10
Tempo Medio por Amostra: 191439.59 us
Energia Media por Amostra: 0.0191439571 J
Tempo Total: 1914.40 ms
Energia Total: 0.19143958 J
------------------------------------------

Validacao MLP INT16
Amostra 0 | Pred: 7 | Real: 7 | Tempo: 187268us | Energia: 0.0187267990 J
Amostra 1 | Pred: 2 | Real: 2 | Tempo: 187432us | Energia: 0.0187432003 J
Amostra 2 | Pred: 1 | Real: 1 | Tempo: 187624us | Energia: 0.0187623977 J
Amostra 3 | Pred: 0 | Real: 0 | Tempo: 187360us | Energia: 0.0187359991 J
Amostra 4 | Pred: 4 | Real: 4 | Tempo: 187520us | Energia: 0.0187520008 J
Amostra 5 | Pred: 1 | Real: 1 | Tempo: 187592us | Energia: 0.0187591981 J
Amostra 6 | Pred: 4 | Real: 4 | Tempo: 187500us | Energia: 0.0187499990 J
Amostra 7 | Pred: 9 | Real: 9 | Tempo: 187508us | Energia: 0.0187508001 J
Amostra 8 | Pred: 6 | Real: 5 | Tempo: 187412us | Energia: 0.0187411975 J
Amostra 9 | Pred: 9 | Real: 9 | Tempo: 187404us | Energia: 0.0187403984 J
------------------------------------------
Acuracia Final: 9/10
Tempo Medio por Amostra: 187462.00 us
Energia Media por Amostra: 0.0187461986 J
Tempo Total: 1874.62 ms
Energia Total: 0.18746198 J
------------------------------------------
```

Para 10 amostras

| Métrica | INT8 | INT16 | INT32 |
| ------- | ---- | ----- | ----- |
| Acurácia              | 90%       | 90%         | 90%           |
| Tempo Médio/Amostra   | 39,93 ms  | 187,46 ms   | 191,44 ms     |
| Energia Média/Amostra | 0,00399 J | 0,01875 J   | 0,01914 J     |
| Tempo Total           | 399,26 ms | 1.874,62 ms | 1.914,40 ms   |
| Energia Total         | 0,0399 J  | 0,1874 J    | 0,1914 J      |
| Velocidade Relativa   | 4.8x      | 1.02x       | Referência (1x) |

1. Utilização de sensor

Eles raramente testam o dataset completo dentro do chip. Pelo que entendi de TinyML, o teste de 10.000 imagens serve para validar o modelo no CPU/GPU. No microcontrolador, o objetivo é a implantação real. Então eles podem utilizar um sensor de câmera acoplado ao Arduino capturaria uma imagem por vez. O modelo processa essa imagem única e descarta os dados para receber a próxima. Portanto, não há necessidade de guardar 10.000 imagens na memória; apenas o modelo (pesos) e o buffer da imagem atual precisam caber no chip.

2. Validação via Comunicação Serial

Utilização de um script de ponte:
- O hardware de treino, no caso a CPU, lê a imagem 1 do MNIST e envia os 784 bytes via cabo USB Serial para o Arduino
- O Arduino recebe os bytes, roda a predição e devolve apenas o número do resultado (ex: "7")
- A CPU compara com o rótulo real e passa para a imagem 2.

Isso permite testar milhões de amostras sem usar quase nada de Flash.

3. Uso de Hardware com Memória Externa

Para dispositivos que precisam operar isolados (offline) com muitos dados:
- Cartão SD: As imagens são lidas do SD, processadas e descartadas.
- Flash Externa (SPI Flash): Chips como o ESP32 ou o Raspberry Pi Pico permitem conectar memórias Flash externas de 4MB, 8MB ou até 16MB, onde você poderia armazenar uma parte muito maior do dataset.

A acurácia de 92.49% foi validada no CPU com 10.000 amostras. No Arduino, devido à limitação de 32KB de Flash do ATmega328P, foi utilizada uma amostra representativa de 20 imagens para validar a funcionalidade do hardware. O fato de o Arduino acertar as mesmas imagens que o CPU já prova que a implementação de quantização está correta.

Não sei se isso já tem algum valor. Vou tentar implementar o caso 3 e se tiver muita paciência e não tiver custo, tentar o 2.

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']]
    },
    messageStyle: "none"
  });
</script>

# Relatório

## Ato 1

### Escolha do hardware

Escolher o hardware é importante pois a performance final será determinada pelo backend de quantização do hardware escolhido. O backend, nesse contexto, é o "motor" que realmente executa os cálculos matemáticos no hardware. Sem um backend específico, o processador tentaria rodar números inteiros usando as mesmas instruções de ponto flutuante, o que não traria ganho nenhum de velocidade. O backend traduz as operações da sua rede neural para instruções de baixo nível que o seu processador entende de forma otimizada.

As principais funções são:

- Aceleração de Hardware: Utiliza instruções especiais do processador (como AVX-512 em Intel ou NEON em chips ARM) para processar vários dados de uma vez (SIMD)
- Gerenciamento de Memória: Organiza como os pesos da rede são carregados no cache para evitar gargalos

### Quando um modelo é considerado "Integer Only"?

Um modelo é considerado 'integer-only' quando todas as suas operações computacionais (multiplicações, adições, ativações, etc.) são realizadas usando aritmética de inteiros de baixa precisão. Isso significa que não há conversões intermediárias para ponto flutuante durante a inferência. O fluxo de dados é puramente de inteiros do início ao fim do modelo.

O backend `fbgemm` do PyTorch (para CPUs) e o `TensorRT` da NVIDIA são capazes de executar modelos de forma totalmente quantizada para certas arquiteturas e operações.

A quantização híbrida (simulada/mixed-precision) é uma abordagem mais comum e flexível, especialmente durante o processo de desenvolvimento e calibração. Neste cenário, algumas operações são realizadas em inteiros, mas conversões intermediárias ainda podem ocorrer em ponto flutuante.

Embora isso possa parecer flexível, já que permite quantizar apenas as partes mais intensivas computacionalmente do modelo, mantendo outras partes em FP32 para preservar a acurácia ou simplificar a implementação, não funcionria em hardware embarcado que não possui suporte para FP32.

Com base na estrutura do exemplo, dá para perceber que é um exemplo de quantização 'híbrida'. O PyTorch adota a abordagem híbrida por padrão em PTQ por várias razões:

- Facilidade de Uso: Simplifica a integração com o ecossistema PyTorch existente, onde muitas operações e o treinamento são em FP32.

- Compatibilidade: Garante que o modelo possa interagir com outras partes do código que esperam entradas/saídas em FP32.

- Acurácia: Minimiza a perda de acurácia ao permitir que operações sensíveis (ou não otimizadas para inteiros) permaneçam em FP32 ou sejam dequantizadas temporariamente.

- Backends: A capacidade de executar um modelo de forma totalmente 'integer-only' depende do backend de quantização (e.g., fbgemm, qnnpack) e do hardware subjacente. O PyTorch tenta otimizar para o backend disponível, mas a estrutura com QuantStub/DeQuantStub é a representação padrão para a quantização simulada.

Para obter um modelo verdadeiramente 'integer-only' no PyTorch, você precisaria garantir que:

1. Todas as operações são suportadas: Cada operação no seu grafo computacional tem uma implementação otimizada para inteiros no backend de quantização escolhido.
2. Fusão de Operações: Operações como Conv + ReLU ou Linear + ReLU são fundidas em uma única operação quantizada para evitar dequantizações intermediárias.
3. Remoção de DeQuantStub: A DeQuantStub final seria removida ou posicionada apenas no ponto onde a saída precisa ser consumida por um sistema que espera FP32. Para inferência puramente em hardware de inteiros, a saída pode permanecer em INT8.

## Ato II

Para inferência de modelos na AWS, o hardware proprietário de ponta é o AWS Inferentia, que possui os chips Inferentia1 e Inferentia2. Diz a documentação oficial que eles foram projetados especificamente para oferecer alto rendimento e baixa latência, otimizando o custo por inferência em comparação com GPUs tradicionais.

### AWS Inferentia2

O coração das instâncias Inf2 é o acelerador NeuronCore-v2. Diferente de uma CPU de propósito geral, ele é uma arquitetura otimizada para operações tensoriais, já que cada chip possui núcleos especializados que executam operações de álgebra linear de alta densidade, utiliza memória de alta largura de banda para evitar gargalos durante o carregamento dos pesos do modelo e ainda permite a comunicação direta entre chips para modelos robustos, como LLMs, reduzindo a dependência da CPU principal.

### Representação Numérica e Aritmética

Assim como quase toda a arquitetura moderna, esse hardware utiliza *two's complement* para representar números inteiros com sinal.

O compilador da AWS, AWS Neuron SDK, permite um controle fino sobre como a precisão é tratada. No contexto de Deep Learning, a maioria das operações ocorre em FP16, BF16 ou INT8.

O que esperamos no final de um produto escalar é truncamento, mas o padrão do Neuron SDK é arredondamento para o valor mais próximo (*round-to-nearest-even*).

### Testando no Ambiente

Para validar o comportamento exato da aritmética de ponto fixo no chip preciso realizar um experimento, para isso é necessário um ambiente com o AWS Neuron SDK instalado. O foco será comparar o comportamento do acelerador NeuronDevice contra o da CPU padrão.

Escrevi um script baseado nas documentações oficiais que utiliza o framework `PyTorch` com a extensão `torch-neuronx`. Vou tentar forçar uma situação onde a diferença entre truncamento e arredondamento seja visível em operações de ponto flutuante e verificar o comportamento de sinal para o complemento a 2.

```python
import torch
import torch_neuronx
import numpy as np

val = 1.7
tensor_cpu = torch.tensor([val], dtype=torch.float32)

class RoundModel(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.int32) # Conversão explícita

model = RoundModel()
neuron_model = torch_neuronx.trace(model, tensor_cpu)
output_cpu = model(tensor_cpu)
output_neuron = neuron_model(tensor_cpu)

print(f'Valor Original:   {val}')
print(f'Resultado CPU:    {output_cpu.item()}')
print(f'Resultado Neuron: {output_neuron.item()}')

neg_val = -1
tensor_neg = torch.tensor([neg_val], dtype=torch.int32)
res_neg = neuron_model(tensor_neg.to(torch.float32))

print(f'Input negativo: {neg_val} -> Output: {res_neg.item()}')
```

Esse teste não valeu de nada na prática. Assim como o teste que fiz em CPU.
O problema é que Python abstrai demais o hardware. Para ver o que de fato ocorre, precisamos explorar em C ou até Assembly.

Minha ideia é forçar um caso onde o resultado exato não cabe em float32, outro para deixar a FPU arredondar e um para observar o resultado nos bits. Quero ver o arredondamento IEEE-754 acontecendo dentro da FPU.

Float32 tem 23 bits de mantissa (~7 dígitos decimais). Depois de certo ponto, somar $1$ não muda mais o número porque a mantissa não consegue representar.

```c
#include <stdio.h>

int main() {
    float x = 16777216.0f;  // 2^24
    float y = 1.0f;

    float result = x + y;

    printf("x = %.0f\n", x);
    printf("x + 1 = %.0f\n", result);
}

// Resultado:
// x = 16777216
// x + 1 = 16777216
// Matematicamente deveria ser: 16777217
// Mas a FPU faz: resultado exato -> arredondamento para float32
```

Agora vamos ver o *bit pattern*.

```c
#include <stdio.h>
#include <stdint.h>

int main() {
    float x = 16777216.0f;
    float y = 1.0f;

    float result = x + y;

    uint32_t *bits = (uint32_t*)&result;

    printf("Resultado: %.0f\n", result);
    printf("Bits: 0x%X\n", *bits);
}

// Resultado:
// Resultado: 16777216
// Bits: 0x4B800000
// Isso mostra o valor já arredondado pela FPU
```

Aqui vamos ver a FPU descartar completamente o $1e-8$ porque não há bits suficientes na mantissa para representar esse incremento mantendo a magnitude do $1.0$.

```c
#include <stdio.h>

int main() {
    float a = 1.0f;
    float b = 1e-8f;

    float c = a + b;

    printf("1.0 + 1e-8 = %.10f\n", c);
}

// Resultado:
// 1.0000000000
// A FPU faz: 1.00000001 -> round-to-nearest -> 1.00000000
```

Vou tentar ler o registrador da FPU.

```c
#include <stdio.h>
#include <fenv.h>

int main() {
    int mode = fegetround();

    if(mode == FE_TONEAREST)
        printf("Round to nearest\n");
}

// Resultado:
// Round to nearest
```

Agora vou tentar trocar o modo da FPU. Normalmente, quando o resultado de uma operação matemática não pode ser representado exatamente em binário, a CPU usa um padrão chamado 'arredondamento para o par mais próximo', como vimos.

Ao usar a biblioteca <fenv.h> e a função `fesetround` (FE_DOWNWARD), você está forçando a unidade de ponto flutuante do processador a mudar seu comportamento: todo resultado de uma operação deve ser arredondado para o número representável imediatamente abaixo (ou igual) ao valor exato.

A operação 1.9f + 1.0f resulta em 2.9.

O problema é que 2.9 não possui uma representação binária exata (ele é uma dízima periódica em binário).

Sem a alteração, o sistema arredondaria 2.9 para o valor binário mais próximo, que é algo como 2.900000095....

Com FE_DOWNWARD, o sistema ignora o "excesso" e trava no número representável imediatamente anterior a 2.9.

```c
#include <stdio.h>
#include <fenv.h>

int main() {
    fesetround(FE_DOWNWARD);
    float x = 1.9f + 1.0f;
    printf("%f\n", x);
}

// Resultado:
// 2.900000
```

O compilador gera instruções SIMD como:

```
addss
mulss
vfmadd132ps
```

---

## Objective of the Experiment

The goal is to compare an MLP implemented with:
- Floating-point (FP32)
- Fixed-point (Q-format, e.g., Q15, Q7)

Evaluating:
- Inference time
- Memory usage
- Numerical error
- Classification accuracy
- Energy

## MLP Model Definition

Use a small but realistic network:

Example MLP:
- Input: 64 features
- Hidden layer 1: 32 neurons (ReLU)
- Hidden layer 2: 16 neurons (ReLU)
- Output: 5 classes (Softmax)

Mathematically:

$$h_1 = \text{ReLU}(W_1x + b_1)$$
$$h_2 = \text{ReLU}(W_2x + b_2)$$
$$y = \text{Softmax}(W_3h_2 + b_3)$$

## Fixed-Point Representation

Use uniform quantization:

$$x_q = \text{round}(x \cdot 2^n)$$
​
$$x \approx \frac{x_q}{2^n}$$ 

Where $n$ = number of fractional bits.

Test formats:

| Format | Word length | Fraction bits    | 
| ------ | ----------- | ---------------- | 
| Q7	 | 8 bits	   | 7                | 
| Q15	 | 16 bits     | 15               | 
| Q31	 | 32 bits     | 31               |

## Quantization Strategy

You must define scaling per layer:

Per-layer scaling:

$$S=\frac{2^{n−1}-1}{max(∣x∣)}$$

$$x_q=round(S \cdot x)$$

This avoids overflow and is standard in quantized neural networks.

## What You Should Measure

### Accuracy

$$ Accuracy = \frac{\text{correct predictions}}{\text{total samples}} $$
​
### Numerical Error

Compare float output vs fixed output:

$$RMSE=\sqrt{\frac{1}{N} \sum (y_{float} - y_{fixed})^2}$$

$$SNR=10log10 (\frac{\sum y^2_{float}}{\sum (y_{float} - y_{fixed})^2})$$
 
### Inference Time

$$ Total time = \frac{\text{Total time}}{\text{Number of samples}} $$

### Memory Usage

$$ Memory = \text{weights} + \text{activations}$$


## Experimental Procedure (Step-by-Step)

Step 1 — Train MLP in Float
Train normally using PyTorch/Keras.
Step 2 — Export Weights
Save weights and biases.
Step 3 — Quantize Weights
Quantize:
- Weights
- Biases
- Activations
Step 4 — Implement Fixed-Point Inference
Important: Only inference is fixed-point, not training.
Step 5 — Run Tests
For each format:
- FP32
- INT16
- INT8
Measure:
- Accuracy
- RMSE
- SNR
- Time
- Memory

## Suggested Results Table (For Paper)
Format	Accuracy	RMSE	SNR (dB)	Time (ms)	Memory (KB)
FP32	98.2%	0	∞	2.1	120
Q31	98.1%	0.002	52	1.6	60
Q15	97.5%	0.01	38	1.2	30
Q7	92.0%	0.08	22	0.8	15
This type of table is very common in embedded ML papers.

## Datasets You Can Use: MNIST

## Tools You Can Use
Tool	Purpose
Python + NumPy	Simulation
PyTorch	Train MLP
fxpmath	Fixed-point
CMSIS-NN	Embedded implementation
TensorFlow Lite	Quantization
STM32 / ESP32	Real hardware test

# 3W

## Boxplot

O boxplot, também conhecido como diagrama de caixa, é uma ferramenta estatística utilizada para representar a distribuição de um conjunto de dados e identificar valores discrepantes, chamados *outliers*. Ele é extremamente útil porque condensa várias informações sobre a dispersão e a simetria dos dados em um único gráfico visualmente simples.

Permite visualizar rapidamente cinco medidas estatísticas principais, conhecidas como o resumo dos cinco números:

1.  Mínimo: O menor valor dos dados (excluindo os outliers)
2.  Primeiro Quartil ($Q_1$): O ponto onde $25\%$ dos dados estão abaixo dele
3.  Mediana ($Q_2$): O valor central que divide os dados ao meio ($50\%$)
4.  Terceiro Quartil ($Q_3$): O ponto onde $75\%$ dos dados estão abaixo dele
5.  Máximo: O maior valor dos dados (excluindo os outliers)

### Principais utilidades

- **Identificar Outliers:** Pontos que ficam muito fora da curva (além dos "bigodes" do gráfico) são facilmente visualizados como pontos isolados. Isso ajuda a decidir se esses dados são erros de medição ou casos raros importantes.
- **Analisar a Dispersão:** A altura da "caixa" representa o **Intervalo Interquartil ($IQR = Q_3 - Q_1$)**, que mostra onde estão concentrados os $50\%$ centrais dos seus dados. Quanto maior a caixa, mais dispersos os dados.
- **Verificar a Simetria:** Se a linha da mediana estiver no centro da caixa, os dados são simétricos. Se estiver mais próxima do topo ou da base, há uma **assimetria** (os dados tendem mais para um lado).
- **Comparar Grupos:** É uma das melhores formas de comparar várias categorias lado a lado. Por exemplo, comparar o salário de diferentes departamentos de uma empresa para ver qual tem maior variação ou média.

### Estrutura do Gráfico

* **A Caixa:** Contém a maior parte dos dados ($50\%$)
* **A Linha Central:** Indica a mediana
* **Os Bigodes (Whiskers):** Linhas que se estendem da caixa até os valores máximo e mínimo
* **Pontos Isolados:** Representam os valores atípicos (outliers)

## Normalização Z-SCORE

A normalização Z-score (também chamada de padronização) é uma técnica estatística usada para colocar diferentes conjuntos de dados na mesma "escala", permitindo que você compare maçãs com laranjas de forma justa. Em termos simples, ela transforma seus dados para que a média seja 0 e o desvio padrão seja 1.

O Z-score de um valor indica quantos desvios padrão ele está acima ou abaixo da média. A equação é:

$$z = \frac{x - \mu}{\sigma}$$

Onde:
- $x$: O valor original.
- $\mu$ (mu): A média do conjunto de dados.
- $\sigma$ (sigma): O desvio padrão.

Imagine dois amigos, Ana e João, que estudam em faculdades diferentes com sistemas de notas distintos:

1.  Ana tirou 85 em uma prova de Cálculo 1. Nessa sala, a média foi 75 e o desvio padrão foi 5.
2.  João tirou 90 em uma prova de Álgebra Linear 1. Nessa sala, a média foi 85 e o desvio padrão foi 10.

Olhando apenas para as notas brutas, parece que João foi melhor (AL é mais difícil hahaha). Porém, vamos usar o z-score para ver quem se destacou mais em relação à sua própria turma:

Cálculo da Ana:
$$z_{Ana} = \frac{85 - 75}{5} = \frac{10}{5} = 2,0$$
A Ana está 2 desvios padrão acima da média.

Cálculo do João:
$$z_{João} = \frac{90 - 85}{10} = \frac{5}{10} = 0,5$$
O João está apenas 0,5 desvios padrão acima da média.

Embora a nota nominal do João seja maior, a Ana teve um desempenho superior em relação aos seus colegas de classe.

Isso nos ajuda a estabelecer uma comparação justa entre unidades diferentes, ajuda na identificação de *outliers*, já que um z-score muito alto ou muito baixo geralmente indica um *outlier*. Além disso, muitos algoritmos (SVM e KNN) funcionam muito melhor quando os dados estão normalizados, pois evitam que uma variável com números grandes "atropele" variáveis com números menores.

## Desvio padrão

O desvio padrão é uma medida que indica o quanto os dados de um conjunto estão "espalhados" em relação à média. Enquanto a média te diz onde está o "centro" dos seus dados, o desvio padrão diz se esses dados estão todos agrupados perto desse centro ou se estão amplamente distribuídos.

O desvio padrão ($\sigma$) é a raiz quadrada da variância:

$$\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$$

Onde:
- $x_i$: cada valor individual.
- $\mu$: a média dos valores.
- $N$: o número total de valores.

Basicamente, ele calcula a distância de cada ponto até a média, eleva ao quadrado para eliminar valores negativos, tira a média dessas distâncias e volta para a unidade original com a raiz quadrada.

Imagine dois sensores de pressão no dataset 3W:

- Sensor A: Registra pressões de $99, 100$ e $101$. A média é 100 e o desvio padrão é baixo (os dados são estáveis).
- Sensor B: Registra pressões de $50, 100$ e $150$. A média também é 100, mas o desvio padrão é alto (os dados são voláteis).

Um desvio padrão alto em um sensor de poço pode indicar que o sinal está sofrendo muitas oscilações, o que é um comportamento típico de quando algo está saindo do normal (uma anomalia).

Em muitos casos, os dados seguem uma distribuição normal (sino). Nessa situação, o desvio padrão $\sigma$ nos dá previsibilidade:

- **$68\%$** dos dados estão a uma distância de $1$ desvio padrão da média.
- **$95\%$** dos dados estão a uma distância de $2$ desvios padrões da média.
- **$99.7\%$** dos dados estão a uma distância de $3$ desvios padrões da média.